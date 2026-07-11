"""Modal A100 decode-overhead harness for Esme-214M-Chat: profile and oracle-gated bench.

At 214M the engine is CPU-bound in decode: the GPU finishes each step's kernels in ~1-2 ms
while Python orchestration takes tens of ms. This harness measures where that per-step wall
time goes and tracks it across overhead-reduction changes. The main commands run the default
bf16 CUDA Esme attention path, greedy, prefix caching off:

* ``--command profile`` — diagnostic only, never a speed claim. Per batch size it reports
  (a) plain decode wall per step (no instrumentation), (b) GPU busy per step and the top ops
  by CPU dispatch time from a short ``torch.profiler`` slice, and (c) a ``cProfile`` run
  attributing Python time to engine/model/cache functions.
* ``--command bench`` — the measured number. Per batch size it times the engine like the
  pinned three-way benchmark (fresh engine per iteration, warmup + measured iterations,
  median wall) and gates every row on the fp32 ``PretrainBundleModel.logits()`` oracle with
  reference policy v2. Raw timing is kept; public tok/s remains gated.

    modal run scripts/modal_esme_decode_profile.py --command profile
    modal run scripts/modal_esme_decode_profile.py --command bench
    modal run scripts/modal_esme_decode_profile.py --command bench --batch-sizes 8,32,128
    modal run scripts/modal_esme_decode_profile.py --command capture --batch-sizes 8,64,256
    modal run scripts/modal_esme_decode_profile.py --command serve-smoke

``--command capture`` is the decode-graph report: in ONE container it runs the sync-debug
probe, launch counts, and the oracle-gated same-GPU ablation across the eager window and
the manual piecewise CUDA-graph runner.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import modal

from scripts.modal_esme_bundle import (
    ESME_BUNDLE_MOUNT,
    REMOTE_BUNDLE_PATH,
    VOLUME_NAME,
    local_bundle_path,
    stage_bundle,
)
from scripts.modal_flash_image import FLASH_IMAGE, REPO_ROOT

BLOCK_SIZE = 128
MAX_NEW_TOKENS = 64
WARMUP_ITERS = 1
MEASURED_ITERS = 3
# How many decode steps the torch.profiler slice covers — enough to average out per-step
# jitter, short enough that profiler overhead stays a few seconds.
PROFILER_STEPS = 16
# Batch-size buckets the decode-graph runner captures. Includes 256 so the largest bench
# batch replays from graphs instead of silently falling back to the eager window.
CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# The serving eval's phase0 sampled workload settings, so the sampled profile attributes the
# same configuration the HTTP measurement pays for.
SAMPLED_SETTINGS = {"temperature": 0.8, "top_p": 0.95, "top_k": 32, "seed": 17}

app = modal.App("llm-infer-esme-decode-profile")

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _parse_batch_sizes(raw: str) -> list[int]:
    sizes = [int(part) for part in raw.split(",") if part.strip()]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError(f"batch sizes must be positive integers; got {raw!r}")
    return sizes


def _build_engine(runtime, num_requests: int, device: str, config: dict | None = None):
    """A fresh engine with the benchmark's block sizing and all requests queued (greedy).

    ``config`` overrides engine knobs for the same-GPU ablation: ``decode_window_size``
    (int), ``planned_decode`` (bool, masks the capability flag), and ``sampling``
    (a :class:`SamplingParams` applied to every request; None keeps greedy).
    """
    from dataclasses import replace

    from llm_infer.benchmarks.esme_paged import build_requests
    from llm_infer.serving import InferenceEngine, Request

    config = config or {}
    sampling = config.get("sampling")
    capabilities = runtime.capabilities
    if config.get("planned_decode") is False:
        capabilities = replace(capabilities, planned_decode=False)
    window_kwargs = (
        {"decode_window_size": config["decode_window_size"]}
        if "decode_window_size" in config
        else {}
    )
    requests = build_requests(runtime.tokenizer, num_requests)
    needed = sum(math.ceil((len(req.prompt_ids) + MAX_NEW_TOKENS) / BLOCK_SIZE) for req in requests)
    engine = InferenceEngine(
        runtime.model,
        block_size=BLOCK_SIZE,
        num_blocks=needed + max(4, len(requests)),
        device=device,
        capabilities=capabilities,
        profiler=config.get("profiler"),
        **window_kwargs,
    )
    for req in requests:
        engine.add_request(
            Request(
                req.request_id,
                list(req.prompt_ids),
                MAX_NEW_TOKENS,
                runtime.eos_token_ids,
                **({"sampling": sampling} if sampling is not None else {}),
            )
        )
    return engine, requests


def _run_prefill_step(engine) -> None:
    """Advance the engine through its first step (prefill + first sampled token)."""
    engine.step()


def _decode_wall(runtime, num_requests: int, config: dict | None = None) -> dict:
    """Un-instrumented decode timing: wall per decode step after prefill, one synced run."""
    import torch

    engine, _ = _build_engine(runtime, num_requests, "cuda", config)
    _run_prefill_step(engine)
    torch.cuda.synchronize()

    steps = 0
    tokens = 0
    start = time.perf_counter()
    while engine.scheduler.has_work():
        result = engine.step()
        steps += 1
        tokens += sum(len(ids) for ids in result.tokens.values())
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - start
    return {
        "decode_steps": steps,
        "decode_tokens": tokens,
        "wall_s": wall_s,
        "wall_ms_per_step": wall_s * 1000.0 / steps if steps else None,
        "decode_tokens_per_second": tokens / wall_s if wall_s > 0 else None,
    }


def _torch_profile(runtime, num_requests: int, config: dict | None = None) -> dict:
    """GPU busy time vs CPU dispatch over a short decode slice, via torch.profiler.

    CUDA *event* spans include GPU idle gaps while the CPU is still issuing work, so they
    cannot separate busy from starved on a CPU-bound engine. Kineto's per-kernel times can:
    ``self_cuda_time_total`` sums actual kernel execution. CPU times here carry profiler
    overhead — use them for attribution, never as a wall claim.

    A "step" below is one scheduler pass. With the deferred window (default size 8) one pass
    runs a whole window, so per-token launch counts are the per-step counts divided by the
    window size; the eager-vs-graphs comparison holds either way because both run the same
    pass shape.
    """
    import torch

    engine, _ = _build_engine(runtime, num_requests, "cuda", config)
    _run_prefill_step(engine)
    torch.cuda.synchronize()

    steps = 0
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        while engine.scheduler.has_work() and steps < PROFILER_STEPS:
            engine.step()
            steps += 1
        torch.cuda.synchronize()

    events = prof.key_averages()
    gpu_busy_ms = sum(evt.self_device_time_total for evt in events) / 1000.0
    cpu_total_ms = sum(evt.self_cpu_time_total for evt in events) / 1000.0
    launch_calls = sum(evt.count for evt in events if evt.key == "cudaLaunchKernel")
    graph_launches = sum(evt.count for evt in events if evt.key == "cudaGraphLaunch")
    top_cpu = sorted(events, key=lambda evt: evt.self_cpu_time_total, reverse=True)[:15]
    return {
        "profiled_steps": steps,
        "gpu_busy_ms_per_step": gpu_busy_ms / steps if steps else None,
        "profiler_cpu_ms_per_step": cpu_total_ms / steps if steps else None,
        "cuda_launch_kernel_per_step": launch_calls / steps if steps else None,
        "cuda_graph_launch_per_step": graph_launches / steps if steps else None,
        "top_ops_by_self_cpu": [
            {
                "op": evt.key,
                "calls": evt.count,
                "self_cpu_ms": evt.self_cpu_time_total / 1000.0,
                "self_cuda_ms": evt.self_device_time_total / 1000.0,
            }
            for evt in top_cpu
        ],
    }


def _sync_op_isolation() -> dict:
    """Sync-debug warning counts for the primitive ops the window flush is built from.

    The pass-level probe counts warnings but the c10 message never names the caller; this
    attributes them. Each case runs alone under ``set_sync_debug_mode("warn")`` with a
    recording warnings context, so the count per op is exact. Pinned allocation appears
    twice because the caching host allocator may behave differently cold vs warm.
    """
    import warnings

    import torch

    device_matrix = torch.randint(0, 100, (8, 8), dtype=torch.long, device="cuda")
    lookup = torch.tensor([1, 2], dtype=torch.long, device="cuda")
    host = torch.empty((8, 8), dtype=torch.long, pin_memory=True)
    event = torch.cuda.Event()
    repeats = torch.full((8,), 3, dtype=torch.long, device="cuda")
    batch_arange = torch.arange(8, device="cuda")

    cases = {
        "empty_pinned_cold": lambda: torch.empty((16, 16), dtype=torch.long, pin_memory=True),
        "empty_pinned_warm": lambda: torch.empty((16, 16), dtype=torch.long, pin_memory=True),
        "copy_nonblocking_d2h_pinned": lambda: host.copy_(device_matrix, non_blocking=True),
        "eos_mask_on_device": lambda: (device_matrix.unsqueeze(-1) == lookup).any(dim=-1),
        "event_record": lambda: event.record(),
        "event_synchronize": lambda: event.synchronize(),
        "repeat_interleave_output_size": lambda: torch.repeat_interleave(
            batch_arange, repeats, output_size=24
        ),
        "tolist_pinned_host": lambda: host.tolist(),
        "argmax_device": lambda: torch.argmax(device_matrix, dim=-1),
        "stack_device": lambda: torch.stack([device_matrix[0], device_matrix[1]]),
    }
    torch.cuda.synchronize()
    counts: dict[str, int] = {}
    torch.cuda.set_sync_debug_mode("warn")
    try:
        for name, run_case in cases.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_case()
            counts[name] = len(caught)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return counts


def _sync_first_site(runtime, num_requests: int) -> dict:
    """Python stacks of every synchronizing op in one decode pass.

    Sync-debug warnings are emitted synchronously in the calling thread, so the stack at
    ``showwarning`` time names the exact call site — unlike the c10 message, which is the
    same generic line for every sync source.
    """
    import traceback
    import warnings

    import torch

    engine, _ = _build_engine(runtime, num_requests, "cuda")
    _run_prefill_step(engine)
    torch.cuda.synchronize()

    sites: list[str] = []

    def record_site(message, category, filename, lineno, file=None, line=None):
        del message, category, filename, lineno, file, line
        stack = traceback.format_stack(limit=14)
        sites.append("".join(stack[:-1]))  # drop the record_site frame itself

    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = record_site
            engine.step()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return {"sync_sites": sites}


def _sync_probe(runtime, num_requests: int) -> dict:
    """Host-sync inventory of the decode loop under ``torch.cuda.set_sync_debug_mode``.

    Each engine step (one scheduler pass = one whole deferred window) runs with sync-debug
    warnings recorded. The contract being checked: the first window is fully sync-free
    (steps, staging flush, everything), and later windows show only the one staged-copy
    ``event.synchronize`` per window boundary, consumed a window behind.
    """
    import warnings

    import torch

    engine, _ = _build_engine(runtime, num_requests, "cuda")
    _run_prefill_step(engine)
    torch.cuda.synchronize()

    per_pass: list[list[str]] = []
    torch.cuda.set_sync_debug_mode("warn")
    try:
        for _ in range(3):
            if not engine.scheduler.has_work():
                break
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                engine.step()
            per_pass.append([str(warning.message) for warning in caught])
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    return {
        "sync_warnings_per_pass": [len(messages) for messages in per_pass],
        "messages": per_pass,
    }


def _python_profile(runtime, num_requests: int, config: dict | None = None) -> dict:
    """Function-level Python attribution of the decode loop via cProfile (diagnostic only)."""
    import cProfile
    import pstats

    engine, _ = _build_engine(runtime, num_requests, "cuda", config)
    _run_prefill_step(engine)

    profiler = cProfile.Profile()
    profiler.enable()
    while engine.scheduler.has_work():
        engine.step()
    profiler.disable()

    stats = pstats.Stats(profiler)
    total_s = stats.total_tt  # type: ignore[attr-defined]
    rows = []
    for (filename, lineno, name), (_cc, ncalls, tottime, cumtime, _callers) in sorted(
        stats.stats.items(),  # type: ignore[attr-defined]
        key=lambda item: item[1][2],
        reverse=True,
    )[:25]:
        rows.append(
            {
                "function": f"{Path(filename).name}:{lineno}:{name}",
                "ncalls": ncalls,
                "tottime_s": tottime,
                "cumtime_s": cumtime,
            }
        )
    return {"profile_total_s": total_s, "top_functions_by_tottime": rows}


def _phase_profile(runtime, num_requests: int, config: dict | None = None) -> dict:
    """One diagnostic generation projected to the Phase 0 timing buckets."""
    import torch

    from llm_infer.profiling import TimingProfiler, attach_host_method_profile

    profiler = TimingProfiler("cuda")
    build_config = {**(config or {}), "profiler": profiler}
    engine, _ = _build_engine(runtime, num_requests, "cuda", build_config)
    attach_host_method_profile(
        profiler,
        engine,
        "window_flushing",
        ("_stage_window_flush", "_consume_window_flush"),
    )
    engine.run()
    torch.cuda.synchronize()
    return profiler.summary().as_dict()


def _reference_by_request(oracle_runtime, requests) -> dict[str, list[int]]:
    """fp32 oracle greedy outputs per request, computed once per unique prompt.

    The workload cycles a small prompt pool, so large batches repeat prompts; greedy oracle
    outputs are identical per prompt and need computing only once each.
    """
    from llm_infer.model.decode import greedy_decode

    by_prompt: dict[tuple[int, ...], list[int]] = {}
    for req in requests:
        if req.prompt_ids not in by_prompt:
            by_prompt[req.prompt_ids] = greedy_decode(
                oracle_runtime.model,
                list(req.prompt_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_ids=set(oracle_runtime.eos_token_ids),
            )
    return {req.request_id: list(by_prompt[req.prompt_ids]) for req in requests}


def _bench_batch(
    oracle_runtime, flash_runtime, num_requests: int, config: dict | None = None
) -> dict:
    """One oracle-gated flash-engine row: median wall over fresh-engine iterations."""
    import statistics

    import torch

    from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
    from llm_infer.benchmarks.reference_policy import build_system_evidence_record
    from llm_infer.benchmarks.report import total_output_tokens

    def decode_once() -> dict[str, list[int]]:
        engine, _ = _build_engine(flash_runtime, num_requests, "cuda", config)
        return engine.run()

    for _ in range(WARMUP_ITERS):
        decode_once()
        torch.cuda.synchronize()
    per_iter: list[float] = []
    outputs: dict[str, list[int]] = {}
    for _ in range(MEASURED_ITERS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = decode_once()
        torch.cuda.synchronize()
        per_iter.append(time.perf_counter() - start)
    median_s = statistics.median(per_iter)

    _, requests = _build_engine(flash_runtime, num_requests, "cuda")
    reference = _reference_by_request(oracle_runtime, requests)
    agreement = tie_tolerant_agreement(
        oracle_runtime.model, requests, outputs, reference, oracle_runtime.eos_token_ids
    )
    tokens = total_output_tokens(outputs, oracle_runtime.eos_token_ids)
    return {
        "batch_size": num_requests,
        "agreement": {
            "exact": agreement.exact,
            "tie": agreement.tie,
            "nontie": agreement.nontie,
            "total": agreement.total,
            "ties_sample": agreement.ties_sample,
            "divergences_sample": agreement.divergences_sample,
        },
        "median_seconds": median_s,
        "per_iter_seconds": per_iter,
        "total_output_tokens": tokens,
        **build_system_evidence_record(
            agreement=agreement,
            median_seconds=median_s,
            total_tokens=tokens,
        ),
    }


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def profile_decode(batch_sizes: list[int]) -> str:
    """Decode-overhead profile per batch size: wall/step, GPU busy/step, Python attribution."""
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.model.runtime import load_model_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    results = []
    for size in batch_sizes:
        wall = _decode_wall(runtime, size)
        kineto = _torch_profile(runtime, size)
        python = _python_profile(runtime, size)
        phases = _phase_profile(runtime, size)
        results.append(
            {
                "batch_size": size,
                "wall": wall,
                "torch_profiler": kineto,
                "phase_profile": phases,
                **python,
            }
        )
        print(
            f"[profile] batch={size}: wall/step {wall['wall_ms_per_step']:.2f} ms, "
            f"gpu busy/step {kineto['gpu_busy_ms_per_step']:.2f} ms, "
            f"decode tok/s {wall['decode_tokens_per_second']:.1f}"
        )
    return json.dumps(
        {
            "results": results,
            "attention_backend": type(runtime.model.backend).__name__,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def profile_sampled_decode(batch_sizes: list[int]) -> str:
    """Greedy-vs-sampled decode attribution per batch size, in one container.

    Sampled requests run the serving eval's phase0 settings (temperature 0.8, top-p 0.95,
    top-k 32, seed 17), which today disable the deferred decode window and take the per-row
    Python sampler. The greedy wall row anchors the comparison on the same GPU, so the
    sampled-minus-greedy delta is the engine-side cost Phase 5 has to remove. Diagnostic
    only, never a speed claim.
    """
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving import SamplingParams

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    sampled_config = {"sampling": SamplingParams(**SAMPLED_SETTINGS)}
    results = []
    for size in batch_sizes:
        # Throwaway generation at this shape first: kernel JIT and per-shape plan warmup must
        # not land in the greedy wall, or the greedy-vs-sampled delta reads backwards.
        _decode_wall(runtime, size)
        greedy_wall = _decode_wall(runtime, size)
        sampled_wall = _decode_wall(runtime, size, sampled_config)
        kineto = _torch_profile(runtime, size, sampled_config)
        python = _python_profile(runtime, size, sampled_config)
        phases = _phase_profile(runtime, size, sampled_config)
        results.append(
            {
                "batch_size": size,
                "greedy_wall": greedy_wall,
                "sampled_wall": sampled_wall,
                "sampled_torch_profiler": kineto,
                "sampled_phase_profile": phases,
                **python,
            }
        )
        print(
            f"[sampled-profile] batch={size}: "
            f"greedy wall/step {greedy_wall['wall_ms_per_step']:.2f} ms, "
            f"sampled wall/step {sampled_wall['wall_ms_per_step']:.2f} ms, "
            f"greedy tok/s {greedy_wall['decode_tokens_per_second']:.1f}, "
            f"sampled tok/s {sampled_wall['decode_tokens_per_second']:.1f}"
        )
    return json.dumps(
        {
            "results": results,
            "sampling": SAMPLED_SETTINGS,
            "attention_backend": type(runtime.model.backend).__name__,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def bench_decode(batch_sizes: list[int]) -> str:
    """Oracle-gated flash-engine bench per batch size (fresh engine per iter, median wall)."""
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.model.runtime import load_model_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    flash_runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    rows = []
    for size in batch_sizes:
        row = _bench_batch(oracle_runtime, flash_runtime, size)
        rows.append(row)
        tps = row["tokens_per_second"]
        print(
            f"[bench] batch={size}: median {row['median_seconds']:.3f} s, "
            f"tokens {row['total_output_tokens']}, "
            f"tok/s {f'{tps:.1f}' if tps else 'NOT REPORTED (diverged)'}"
        )
    return json.dumps(
        {
            "rows": rows,
            "attention_backend": type(flash_runtime.model.backend).__name__,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


# Same-GPU ablation configs: one container measures all of them back to back, so the
# comparison is free of Modal's machine-to-machine variance (GPU SKU, clocks, host CPU).
# ``decode_graphs`` is a model-level toggle (the runner is built once and reused), applied
# by the harness around each row rather than by ``_build_engine``.
ABLATION_CONFIGS: dict[str, dict] = {
    "per-step (window=1)": {"decode_window_size": 1},
    "window, classic decode_many": {"planned_decode": False},
    "window + planned buffers (default)": {},
    "window + planned + cuda graphs": {"decode_graphs": True},
}


def _graph_runner(flash_runtime):
    """Capture the decode-graph buckets once (outside any timed region) and return the runner."""
    start = time.perf_counter()
    runner = flash_runtime.model.enable_decode_graphs(CAPTURE_SIZES)
    flash_runtime.model.decode_graphs = None  # rows opt in explicitly
    print(f"[graphs] captured buckets {CAPTURE_SIZES} in {time.perf_counter() - start:.1f} s")
    return runner


def _runner_for(config: dict, graph_runner):
    return graph_runner if config.get("decode_graphs") else None


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def ablate_decode(batch_sizes: list[int]) -> str:
    """Oracle-gated A/B of engine decode configs on ONE GPU — the attribution measurement."""
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.model.runtime import load_model_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    flash_runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    graph_runner = _graph_runner(flash_runtime)
    rows = []
    for size in batch_sizes:
        for label, config in ABLATION_CONFIGS.items():
            flash_runtime.model.decode_graphs = _runner_for(config, graph_runner)
            row = _bench_batch(oracle_runtime, flash_runtime, size, config)
            flash_runtime.model.decode_graphs = None
            row["config_label"] = label
            rows.append(row)
            tps = row["tokens_per_second"]
            print(
                f"[ablate] batch={size} | {label}: "
                f"median {row['median_seconds']:.3f} s, "
                f"tok/s {f'{tps:.1f}' if tps else 'NOT REPORTED (diverged)'}"
            )
    return json.dumps(
        {
            "rows": rows,
            "attention_backend": type(flash_runtime.model.backend).__name__,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=30 * 60,
)
def sync_report(batch_sizes: list[int]) -> str:
    """Sync-attribution probe only: op isolation, first-site traceback, per-pass counts."""
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.model.runtime import load_model_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    flash_runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    ops = _sync_op_isolation()
    for name, count in ops.items():
        print(f"[sync-ops] {name}: {count}")
    size = batch_sizes[0]
    first_site = _sync_first_site(flash_runtime, size)
    for index, site in enumerate(first_site["sync_sites"]):
        print(f"[sync-site] sync {index}:\n{site}")
    probe = _sync_probe(flash_runtime, size)
    print(f"[sync] eager-window: warnings per pass {probe['sync_warnings_per_pass']}")
    return json.dumps(
        {
            "op_isolation": ops,
            **first_site,
            "pass_probe": probe,
            "attention_backend": type(flash_runtime.model.backend).__name__,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    timeout=30 * 60,
)
def flashinfer_graph_probe() -> str:
    """Capture only FlashInfer ``run`` and re-plan fixed buffers across page boundaries."""
    import flashinfer
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.kernels.flashinfer_graph import (
        FlashInferDecodeShape,
        build_graph_wrapper,
        page_metadata_for_lengths,
        plan_wrapper,
        run_wrapper_into,
    )
    from llm_infer.kernels.flashinfer_paged import FlashInferPagedAttention

    expected_version = "0.6.14"
    actual_version = getattr(flashinfer, "__version__", "unknown")
    if actual_version != expected_version:
        raise RuntimeError(
            f"FlashInfer graph probe requires {expected_version}; got {actual_version}"
        )
    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    wrapper_class = FlashInferPagedAttention._decode_wrapper_class(flashinfer)
    shape = FlashInferDecodeShape(
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=64,
        page_size=64,
        dtype=torch.bfloat16,
    )
    lengths_to_probe = [63, 64, 65, 127, 128, 129]
    results: list[dict[str, object]] = []

    for batch_size in (1, 8):
        max_length = max(lengths_to_probe)
        fixed_metadata = page_metadata_for_lengths(
            [max_length] * batch_size,
            page_size=shape.page_size,
            device="cuda",
        )
        # FlashInfer requires the workspace to be zero-initialized before its first use.
        workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        graph_wrapper = build_graph_wrapper(wrapper_class, workspace, fixed_metadata)
        torch.manual_seed(1234 + batch_size)
        query = torch.randn(
            batch_size,
            shape.num_qo_heads,
            shape.head_dim,
            dtype=shape.dtype,
            device="cuda",
        )
        paged_kv = torch.randn(
            int(fixed_metadata.indices.numel()),
            2,
            shape.page_size,
            shape.num_kv_heads,
            shape.head_dim,
            dtype=shape.dtype,
            device="cuda",
        )
        graph_output = torch.empty_like(query)
        fixed_pointers = {
            "workspace": workspace.data_ptr(),
            "fixed_indptr": fixed_metadata.indptr.data_ptr(),
            "fixed_indices": fixed_metadata.indices.data_ptr(),
            "fixed_last_page_len": fixed_metadata.last_page_len.data_ptr(),
            "query": query.data_ptr(),
            "output": graph_output.data_ptr(),
            "paged_kv": paged_kv.data_ptr(),
        }
        initial_metadata = page_metadata_for_lengths(
            [lengths_to_probe[0]] * batch_size,
            page_size=shape.page_size,
            device="cuda",
        )
        plan_wrapper(graph_wrapper, initial_metadata, shape)
        for _ in range(3):
            run_wrapper_into(graph_wrapper, query, paged_kv, graph_output)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run_wrapper_into(graph_wrapper, query, paged_kv, graph_output)
        torch.cuda.synchronize()
        memory_after_capture = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }

        reference_workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        ordinary_wrapper = wrapper_class(
            reference_workspace,
            "NHD",
            use_tensor_cores=False,
            backend="auto",
        )
        plan_wrapper(ordinary_wrapper, initial_metadata, shape)
        reference_warmup_output = torch.empty_like(query)
        run_wrapper_into(ordinary_wrapper, query, paged_kv, reference_warmup_output)
        torch.cuda.synchronize()
        del reference_warmup_output
        memory_before_replans = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }
        comparisons: list[dict[str, object]] = []
        for length in lengths_to_probe:
            metadata = page_metadata_for_lengths(
                [length] * batch_size,
                page_size=shape.page_size,
                device="cuda",
            )
            allocated_before_plan = torch.cuda.memory_allocated()
            reserved_before_plan = torch.cuda.memory_reserved()
            plan_wrapper(graph_wrapper, metadata, shape)
            graph.replay()
            torch.cuda.synchronize()
            allocated_after_plan = torch.cuda.memory_allocated()
            reserved_after_plan = torch.cuda.memory_reserved()

            plan_wrapper(ordinary_wrapper, metadata, shape)
            reference_output = torch.empty_like(query)
            run_wrapper_into(ordinary_wrapper, query, paged_kv, reference_output)
            torch.cuda.synchronize()
            difference = (graph_output.float() - reference_output.float()).abs()
            max_error = float(difference.max().item())
            is_close = bool(torch.allclose(graph_output, reference_output, rtol=1e-2, atol=1e-2))
            comparisons.append(
                {
                    "length": length,
                    "pages_per_request": -(-length // shape.page_size),
                    "allclose": is_close,
                    "max_abs_error": max_error,
                    "allocated_before_replan": allocated_before_plan,
                    "allocated_after_replan": allocated_after_plan,
                    "allocated_replan_delta": allocated_after_plan - allocated_before_plan,
                    "reserved_before_replan": reserved_before_plan,
                    "reserved_after_replan": reserved_after_plan,
                    "reserved_replan_delta": reserved_after_plan - reserved_before_plan,
                    "ordinary_wrapper_id": id(ordinary_wrapper),
                }
            )
            del reference_output

        torch.cuda.synchronize()
        memory_after_replans = {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }
        final_metadata = page_metadata_for_lengths(
            [lengths_to_probe[-1]] * batch_size,
            page_size=shape.page_size,
            device="cuda",
        )
        plan_wrapper(graph_wrapper, final_metadata, shape)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as replay_profile:
            for _ in range(4):
                graph.replay()
            torch.cuda.synchronize()
        replay_events = replay_profile.key_averages()
        plan_wrapper(ordinary_wrapper, final_metadata, shape)
        ordinary_output = torch.empty_like(query)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as ordinary_profile:
            run_wrapper_into(ordinary_wrapper, query, paged_kv, ordinary_output)
            torch.cuda.synchronize()
        ordinary_events = ordinary_profile.key_averages()
        pointers_after_replans = {
            "workspace": workspace.data_ptr(),
            "fixed_indptr": fixed_metadata.indptr.data_ptr(),
            "fixed_indices": fixed_metadata.indices.data_ptr(),
            "fixed_last_page_len": fixed_metadata.last_page_len.data_ptr(),
            "query": query.data_ptr(),
            "output": graph_output.data_ptr(),
            "paged_kv": paged_kv.data_ptr(),
        }
        results.append(
            {
                "batch_size": batch_size,
                "capture_count": 1,
                "replan_count": len(lengths_to_probe) + 1,
                "plan_count_total": len(lengths_to_probe) + 2,
                "warmup_run_count": 3,
                "wrapper_id": id(graph_wrapper),
                "pointers": fixed_pointers,
                "pointers_after_replans": pointers_after_replans,
                "fixed_pointers_unchanged": fixed_pointers == pointers_after_replans,
                "memory_after_capture": memory_after_capture,
                "memory_before_replans": memory_before_replans,
                "memory_after_replans": memory_after_replans,
                "steady_replan_memory_delta": {
                    key: memory_after_replans[key] - memory_before_replans[key]
                    for key in memory_before_replans
                },
                "comparisons": comparisons,
                "all_replays_match": all(row["allclose"] for row in comparisons),
                "launch_profile": {
                    "graph_replay": {
                        "replays": 4,
                        "cuda_graph_launch": sum(
                            event.count for event in replay_events if event.key == "cudaGraphLaunch"
                        ),
                        "cuda_launch_kernel": sum(
                            event.count
                            for event in replay_events
                            if event.key == "cudaLaunchKernel"
                        ),
                    },
                    "ordinary_run": {
                        "runs": 1,
                        "cuda_graph_launch": sum(
                            event.count
                            for event in ordinary_events
                            if event.key == "cudaGraphLaunch"
                        ),
                        "cuda_launch_kernel": sum(
                            event.count
                            for event in ordinary_events
                            if event.key == "cudaLaunchKernel"
                        ),
                    },
                },
            }
        )

    return json.dumps(
        {
            "probe": "flashinfer-fixed-buffer-run-capture",
            "flashinfer_version": actual_version,
            "expected_flashinfer_version": expected_version,
            "plan_contract": "plan once per token step outside capture; capture wrapper.run only",
            "shape": {
                "num_qo_heads": shape.num_qo_heads,
                "num_kv_heads": shape.num_kv_heads,
                "head_dim": shape.head_dim,
                "page_size": shape.page_size,
                "dtype": str(shape.dtype),
            },
            "lengths": lengths_to_probe,
            "results": results,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=3 * 60 * 60,
)
def grouped_layer_ab(
    grouped_layers: int,
    correctness_only: bool = False,
    batch_sizes: list[int] | None = None,
) -> str:
    """A/B of piecewise decode versus one cache-owned grouped graph.

    ``correctness_only`` runs one batch-8 pair and records outputs without publishing a fresh
    timing result. It exists to close a parity question in an older performance record.
    ``batch_sizes`` defaults to the exact low-batch protocol (1 and 8); pass larger sizes for
    the high-batch regression check that gates any serving adoption of the grouped runner.
    """
    import statistics

    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_paged import (
        HEADLINE_PROMPTS,
        build_requests,
        requests_at_context_length,
    )
    from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
    from llm_infer.benchmarks.reference_policy import (
        build_reference_only_record,
        build_system_evidence_record,
        normalized_outputs_match,
    )
    from llm_infer.model.decode import greedy_decode
    from llm_infer.model.grouped_decode_graph import EngineOwnedGroupedDecodeGraphRunner
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    if grouped_layers not in (2, 4):
        raise ValueError(f"grouped_layers must be 2 or 4; got {grouped_layers}")
    esme_bundles.reload()
    if batch_sizes is None:
        batch_sizes = [8] if correctness_only else [1, 8]
    context_length = 256
    max_new_tokens = 128
    warmup_pairs = 0 if correctness_only else 2
    measured_pairs = 1 if correctness_only else 10
    eos = frozenset()
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    reference_by_prompt: dict[tuple[int, ...], list[int]] = {}
    rows: list[dict[str, object]] = []
    candidate_label = f"{grouped_layers}_layer_group"

    def p95(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[math.ceil(0.95 * len(ordered)) - 1]

    def reference_for(requests) -> dict[str, list[int]]:
        for request in requests:
            if request.prompt_ids not in reference_by_prompt:
                reference_by_prompt[request.prompt_ids] = greedy_decode(
                    oracle_runtime.model,
                    list(request.prompt_ids),
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=set(eos),
                )
        return {
            request.request_id: list(reference_by_prompt[request.prompt_ids])
            for request in requests
        }

    for batch_size in batch_sizes:
        runtimes = {
            label: load_model_runtime(
                "esme",
                bundle_path=Path(REMOTE_BUNDLE_PATH),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for label in ("piecewise", candidate_label)
        }
        requests_by_mode = {
            label: requests_at_context_length(
                build_requests(runtime.tokenizer, batch_size, HEADLINE_PROMPTS),
                context_length,
            )
            for label, runtime in runtimes.items()
        }

        def build_engine(
            label: str,
            runtimes=runtimes,
            requests_by_mode=requests_by_mode,
            batch_size=batch_size,
        ):
            runtime = runtimes[label]
            requests = requests_by_mode[label]
            needed = sum(
                math.ceil((len(request.prompt_ids) + max_new_tokens) / 64) for request in requests
            )
            return InferenceEngine(
                runtime.model,
                block_size=64,
                num_blocks=needed + max(8, batch_size),
                device="cuda",
                capabilities=runtime.capabilities,
            )

        engines = {label: build_engine(label) for label in runtimes}
        baseline_model = runtimes["piecewise"].model
        allocated_before_baseline = torch.cuda.memory_allocated()
        capture_start = time.perf_counter()
        baseline_model.enable_decode_graphs(capture_sizes=(batch_size,))
        baseline_capture_seconds = time.perf_counter() - capture_start
        baseline_capture_memory = torch.cuda.memory_allocated() - allocated_before_baseline
        candidate_runner: EngineOwnedGroupedDecodeGraphRunner | None = None
        timings = {"piecewise": [], candidate_label: []}
        token_counts = {"piecewise": [], candidate_label: []}
        last_outputs: dict[str, dict[str, list[int]]] = {}

        def decode_once(
            label: str,
            runtimes=runtimes,
            requests_by_mode=requests_by_mode,
            engines=engines,
            batch_size=batch_size,
        ) -> tuple[float, int, dict[str, list[int]]]:
            nonlocal candidate_runner
            runtime = runtimes[label]
            requests = requests_by_mode[label]
            engine = engines[label]
            live_requests = []
            for request in requests:
                live = Request(
                    request.request_id,
                    list(request.prompt_ids),
                    max_new_tokens,
                    eos,
                )
                live_requests.append(live)
                engine.add_request(live)
            outputs = {request.request_id: [] for request in requests}
            first = engine.step()
            for request_id, tokens in first.tokens.items():
                outputs[request_id].extend(int(token) for token in tokens)
            if label == candidate_label and candidate_runner is None:
                tables = [request.block_table for request in live_requests]
                capture_plan = runtime.model.open_decode_window(engine.cache, tables, budget=8)
                if capture_plan is None:
                    raise RuntimeError("two-layer capture could not open a planned window")
                candidate_start = time.perf_counter()
                candidate_runner = EngineOwnedGroupedDecodeGraphRunner(
                    runtime.model,
                    engine.cache,
                    batch_size,
                    grouped_layers=grouped_layers,
                    mode="graph",
                    capture_plan=capture_plan,
                )
                candidate_runner.total_capture_seconds = time.perf_counter() - candidate_start
                runtime.model.decode_graphs = candidate_runner
            torch.cuda.synchronize()
            decode_tokens = 0
            start = time.perf_counter()
            while engine.scheduler.has_work():
                result = engine.step()
                for request_id, tokens in result.tokens.items():
                    values = [int(token) for token in tokens]
                    outputs[request_id].extend(values)
                    decode_tokens += len(values)
            torch.cuda.synchronize()
            return time.perf_counter() - start, decode_tokens, outputs

        for pair in range(warmup_pairs + measured_pairs):
            order = (
                ("piecewise", candidate_label) if pair % 2 == 0 else (candidate_label, "piecewise")
            )
            for label in order:
                elapsed, tokens, outputs = decode_once(label)
                if pair >= warmup_pairs:
                    timings[label].append(elapsed)
                    token_counts[label].append(tokens)
                    last_outputs[label] = outputs

        launch_profiles: dict[str, dict[str, object]] = {}
        for label, engine in () if correctness_only else engines.items():
            requests = requests_by_mode[label]
            for request in requests:
                engine.add_request(
                    Request(
                        request.request_id,
                        list(request.prompt_ids),
                        max_new_tokens,
                        eos,
                    )
                )
            engine.step()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profile:
                passes = 0
                while engine.scheduler.has_work() and passes < 2:
                    engine.step()
                    passes += 1
                torch.cuda.synchronize()
            events = profile.key_averages()
            launch_profiles[label] = {
                "profiled_scheduler_passes": passes,
                "cuda_graph_launch": sum(
                    event.count for event in events if event.key == "cudaGraphLaunch"
                ),
                "cuda_launch_kernel": sum(
                    event.count for event in events if event.key == "cudaLaunchKernel"
                ),
            }

        reference = reference_for(requests_by_mode["piecewise"])
        batch_rows: list[dict[str, object]] = []
        agreements = {}
        for label in ("piecewise", candidate_label):
            agreement = tie_tolerant_agreement(
                oracle_runtime.model,
                requests_by_mode[label],
                last_outputs[label],
                reference,
                eos,
            )
            agreements[label] = agreement
            median_s = statistics.median(timings[label])
            tokens = token_counts[label][-1]
            row = {
                "batch_size": batch_size,
                "context_length": context_length,
                "runner": label,
                "agreement": dataclasses.asdict(agreement),
                "decode_token_counts": token_counts[label],
                "total_decode_tokens": tokens,
            }
            if correctness_only:
                row.update(
                    {
                        **build_reference_only_record(agreement),
                        "measurement_status": "not_measured",
                        "raw_tokens_per_second": None,
                        "tokens_per_second": None,
                    }
                )
            else:
                row.update(
                    {
                        "decode_median_seconds": median_s,
                        "decode_p95_seconds": p95(timings[label]),
                        "decode_per_iter_seconds": timings[label],
                        **build_system_evidence_record(
                            agreement=agreement,
                            median_seconds=median_s,
                            total_tokens=tokens,
                        ),
                        "launch_profile": launch_profiles[label],
                    }
                )
            if label == "piecewise":
                row["capture"] = {
                    "seconds": baseline_capture_seconds,
                    "allocator_delta_bytes": baseline_capture_memory,
                }
            else:
                assert candidate_runner is not None
                row["capture"] = {
                    "seconds": candidate_runner.total_capture_seconds,
                    "group_seconds": candidate_runner.capture_seconds,
                    "group_memory_bytes": candidate_runner.capture_memory_bytes,
                    "group_owned_memory_bytes": candidate_runner.owned_group_memory_bytes,
                    "cache_id": id(candidate_runner.cache),
                    "cache_kv_pointer": candidate_runner._cache_kv_pointer,
                }
            batch_rows.append(row)
            rows.append(row)

        baseline_row, candidate_row = batch_rows
        parity_status = (
            "exact"
            if normalized_outputs_match(
                last_outputs["piecewise"], last_outputs[candidate_label], eos
            )
            else "review_required"
        )
        relative_claim_eligible = (
            parity_status == "exact"
            and token_counts["piecewise"][-1] == token_counts[candidate_label][-1]
        )
        baseline_raw_tps = baseline_row["raw_tokens_per_second"]
        candidate_raw_tps = candidate_row["raw_tokens_per_second"]
        comparison: dict[str, object] = {
            "policy_version": 2,
            "batch_size": batch_size,
            "context_length": context_length,
            "runner": "candidate_vs_baseline",
            "parity_status": parity_status,
            "candidate_matches_baseline_exact": parity_status == "exact",
            "relative_claim_eligible": relative_claim_eligible,
        }
        if correctness_only:
            target_request = "esme-007"
            target_step = 22
            candidate_evidence = next(
                (
                    item
                    for item in agreements[candidate_label].numerical_evidence
                    if item.get("request") == target_request and item.get("step") == target_step
                ),
                None,
            )
            if candidate_evidence is None:
                raise RuntimeError(
                    "batch-8 correctness check did not reproduce esme-007 step-22 evidence"
                )
            comparison.update(
                {
                    "measurement_status": "not_measured",
                    "performance_claim_in_record": False,
                    "headline_eligible": False,
                    "outputs": {
                        "baseline": last_outputs["piecewise"],
                        "candidate": last_outputs[candidate_label],
                    },
                    "reference_outputs": reference,
                    "target_diagnostic": {
                        "request": target_request,
                        "step": target_step,
                        "baseline_token": last_outputs["piecewise"][target_request][target_step],
                        "candidate_token": last_outputs[candidate_label][target_request][
                            target_step
                        ],
                        "fp32_token": reference[target_request][target_step],
                        "baseline_matches_candidate": (
                            last_outputs["piecewise"][target_request][target_step]
                            == last_outputs[candidate_label][target_request][target_step]
                        ),
                        "fp32_evidence": candidate_evidence,
                    },
                }
            )
        else:
            raw_decode_median_ratio = (
                candidate_row["decode_median_seconds"] / baseline_row["decode_median_seconds"]
            )
            comparison.update(
                {
                    "raw_decode_median_ratio": raw_decode_median_ratio,
                    "decode_median_ratio": (
                        raw_decode_median_ratio if relative_claim_eligible else None
                    ),
                    "tokens_per_second_ratio": (
                        candidate_raw_tps / baseline_raw_tps
                        if relative_claim_eligible
                        and candidate_raw_tps is not None
                        and baseline_raw_tps is not None
                        else None
                    ),
                }
            )
        rows.append(comparison)

    return json.dumps(
        {
            "probe": "engine-owned-grouped-layer-decode-graph",
            "policy_version": 2,
            "measurement_status": "not_measured" if correctness_only else "measured",
            "rows": rows,
            "experiment": {
                "batch_sizes": batch_sizes,
                "context_length": context_length,
                "max_new_tokens": max_new_tokens,
                "warmup_pairs": warmup_pairs,
                "measured_pairs": measured_pairs,
                "grouped_layer_count": grouped_layers,
                "grouped_layers": list(range(grouped_layers)),
                "expected_graph_launches_per_token": 31 - grouped_layers,
                "planning": "outside capture every token",
            },
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=3 * 60 * 60,
)
def bucket_policy_report(probe_batches: list[int]) -> str:
    """The grouped bucket-policy record: off-bucket A/B, capture budget, workspace sharing.

    Three questions, one container:

    1. **Exact-size vs padded grouped buckets.** For each off-bucket batch N, A/B today's
       real fallback (piecewise, padded up to the next bucket) against an exact-N grouped
       runner — the ceiling any padded-grouped design could reach. A small gap means
       exact-size-only wins on simplicity; a large gap prices the pad-row scratch design.
    2. **Capture budget.** Per-bucket capture seconds and owned memory for the full
       power-of-two ladder, so the serving default set is chosen against a startup budget.
    3. **Workspace sharing.** Build the ladder twice (per-runner vs one shared 128 MiB
       FlashInfer workspace) and gate a mixed-bucket decode on the shared engine against
       the fp32 oracle — deciding ~200 MiB versus ~1.1 GiB for an 8-bucket set.
    """
    import statistics

    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_paged import (
        HEADLINE_PROMPTS,
        build_requests,
        requests_at_context_length,
    )
    from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
    from llm_infer.benchmarks.reference_policy import (
        build_system_evidence_record,
        normalized_outputs_match,
    )
    from llm_infer.model.decode import greedy_decode
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    context_length = 256
    max_new_tokens = 128
    warmup_pairs = 2
    measured_pairs = 6
    piecewise_buckets = (16, 32, 64, 128, 256)
    ladder = (1, 2, 4, 8, 16, 32, 64, 128)
    eos = frozenset()
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    reference_by_prompt: dict[tuple[int, ...], list[int]] = {}

    def reference_for(requests) -> dict[str, list[int]]:
        for request in requests:
            if request.prompt_ids not in reference_by_prompt:
                reference_by_prompt[request.prompt_ids] = greedy_decode(
                    oracle_runtime.model,
                    list(request.prompt_ids),
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=set(eos),
                )
        return {
            request.request_id: list(reference_by_prompt[request.prompt_ids])
            for request in requests
        }

    def build_engine(runtime, requests, batch_size: int, **grouped_kwargs) -> InferenceEngine:
        needed = sum(
            math.ceil((len(request.prompt_ids) + max_new_tokens) / 64) for request in requests
        )
        return InferenceEngine(
            runtime.model,
            block_size=64,
            num_blocks=needed + max(8, batch_size),
            device="cuda",
            capabilities=runtime.capabilities,
            **grouped_kwargs,
        )

    def decode_once(engine, requests) -> tuple[float, int, dict[str, list[int]]]:
        """Add every request, absorb the first (prefill) step, then time the decode drain."""
        for request in requests:
            engine.add_request(
                Request(request.request_id, list(request.prompt_ids), max_new_tokens, eos)
            )
        outputs = {request.request_id: [] for request in requests}
        first = engine.step()
        for request_id, tokens in first.tokens.items():
            outputs[request_id].extend(int(token) for token in tokens)
        torch.cuda.synchronize()
        decode_tokens = 0
        start = time.perf_counter()
        while engine.scheduler.has_work():
            result = engine.step()
            for request_id, tokens in result.tokens.items():
                values = [int(token) for token in tokens]
                outputs[request_id].extend(values)
                decode_tokens += len(values)
        torch.cuda.synchronize()
        return time.perf_counter() - start, decode_tokens, outputs

    rows: list[dict[str, object]] = []
    for batch_size in probe_batches:
        labels = ("piecewise_padded", "grouped_exact")
        runtimes = {
            label: load_model_runtime(
                "esme",
                bundle_path=Path(REMOTE_BUNDLE_PATH),
                dtype=torch.bfloat16,
                device="cuda",
            )
            for label in labels
        }
        requests_by_label = {
            label: requests_at_context_length(
                build_requests(runtime.tokenizer, batch_size, HEADLINE_PROMPTS),
                context_length,
            )
            for label, runtime in runtimes.items()
        }
        # Both systems keep the identical piecewise fallback; the candidate adds one
        # exact-batch grouped runner on top, which is exactly the serving dispatch chain.
        for label in labels:
            runtimes[label].model.enable_decode_graphs(capture_sizes=piecewise_buckets)
        engines = {
            "piecewise_padded": build_engine(
                runtimes["piecewise_padded"],
                requests_by_label["piecewise_padded"],
                batch_size,
            ),
            "grouped_exact": build_engine(
                runtimes["grouped_exact"],
                requests_by_label["grouped_exact"],
                batch_size,
                grouped_decode_graphs=True,
                grouped_capture_sizes=(batch_size,),
            ),
        }
        timings = {label: [] for label in labels}
        token_counts = {label: [] for label in labels}
        last_outputs: dict[str, dict[str, list[int]]] = {}
        for pair in range(warmup_pairs + measured_pairs):
            order = labels if pair % 2 == 0 else tuple(reversed(labels))
            for label in order:
                elapsed, tokens, outputs = decode_once(engines[label], requests_by_label[label])
                if pair >= warmup_pairs:
                    timings[label].append(elapsed)
                    token_counts[label].append(tokens)
                    last_outputs[label] = outputs

        grouped_runner = engines["grouped_exact"].grouped_decode_runners[batch_size]
        if grouped_runner.steps_handled == 0:
            raise RuntimeError(f"grouped runner never fired at exact batch {batch_size}")
        reference = reference_for(requests_by_label["piecewise_padded"])
        batch_rows: list[dict[str, object]] = []
        for label in labels:
            agreement = tie_tolerant_agreement(
                oracle_runtime.model,
                requests_by_label[label],
                last_outputs[label],
                reference,
                eos,
            )
            median_s = statistics.median(timings[label])
            tokens = token_counts[label][-1]
            row = {
                "batch_size": batch_size,
                "context_length": context_length,
                "runner": label,
                "padded_bucket": (
                    next((b for b in piecewise_buckets if b >= batch_size), None)
                    if label == "piecewise_padded"
                    else batch_size
                ),
                "agreement": dataclasses.asdict(agreement),
                "decode_median_seconds": median_s,
                "decode_per_iter_seconds": timings[label],
                "total_decode_tokens": tokens,
                **build_system_evidence_record(
                    agreement=agreement,
                    median_seconds=median_s,
                    total_tokens=tokens,
                ),
            }
            if label == "grouped_exact":
                row["capture"] = {
                    "seconds": grouped_runner.total_capture_seconds,
                    "group_seconds": grouped_runner.capture_seconds,
                    "group_owned_memory_bytes": grouped_runner.owned_group_memory_bytes,
                    "steps_handled": grouped_runner.steps_handled,
                }
            batch_rows.append(row)
            rows.append(row)

        baseline_row, candidate_row = batch_rows
        parity_exact = normalized_outputs_match(
            last_outputs["piecewise_padded"], last_outputs["grouped_exact"], eos
        )
        relative_claim_eligible = (
            parity_exact
            and token_counts["piecewise_padded"][-1] == token_counts["grouped_exact"][-1]
        )
        raw_ratio = candidate_row["decode_median_seconds"] / baseline_row["decode_median_seconds"]
        rows.append(
            {
                "policy_version": 2,
                "batch_size": batch_size,
                "runner": "grouped_exact_vs_piecewise_padded",
                "parity_status": "exact" if parity_exact else "review_required",
                "relative_claim_eligible": relative_claim_eligible,
                "raw_decode_median_ratio": raw_ratio,
                "decode_median_ratio": raw_ratio if relative_claim_eligible else None,
            }
        )
        print(
            f"[bucket-policy] batch={batch_size}: grouped/piecewise wall ratio "
            f"{raw_ratio:.3f} (parity {'exact' if parity_exact else 'REVIEW'})"
        )
        del engines, runtimes
        torch.cuda.empty_cache()

    # Workspace sharing: capture the ladder twice, then gate a mixed-bucket decode on the
    # shared engine so wrapper interleaving across runners is exercised, not assumed.
    sharing: dict[str, object] = {"ladder": list(ladder)}
    sharing_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    sharing_runtime.model.enable_decode_graphs(capture_sizes=piecewise_buckets)
    ladder_requests = requests_at_context_length(
        build_requests(sharing_runtime.tokenizer, max(ladder), HEADLINE_PROMPTS),
        context_length,
    )
    for shared in (False, True):
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        start = time.perf_counter()
        engine = build_engine(
            sharing_runtime,
            ladder_requests,
            max(ladder),
            grouped_decode_graphs=True,
            grouped_capture_sizes=ladder,
            grouped_shared_workspace=shared,
        )
        torch.cuda.synchronize()
        capture_s = time.perf_counter() - start
        capture_bytes = torch.cuda.memory_allocated() - allocated_before
        key = "shared_workspace" if shared else "per_runner_workspace"
        sharing[key] = {
            "total_capture_seconds": capture_s,
            "total_allocator_delta_bytes": capture_bytes,
            "per_bucket": {
                str(size): {
                    "capture_seconds": runner.total_capture_seconds,
                    "group_owned_memory_bytes": runner.owned_group_memory_bytes,
                }
                for size, runner in engine.grouped_decode_runners.items()
            },
        }
        if shared:
            # Mixed-bucket parity: run the full ladder batch, then a smaller one, through
            # the same shared-workspace engine and require exact fp32 reference agreement.
            parity: dict[str, object] = {}
            for size in (max(ladder), 8):
                requests = ladder_requests[:size]
                _, _, outputs = decode_once(engine, requests)
                agreement = tie_tolerant_agreement(
                    oracle_runtime.model,
                    requests,
                    outputs,
                    reference_for(requests),
                    eos,
                )
                runner = engine.grouped_decode_runners[size]
                parity[str(size)] = {
                    "agreement": dataclasses.asdict(agreement),
                    "steps_handled": runner.steps_handled,
                }
                if runner.steps_handled == 0:
                    raise RuntimeError(f"shared-workspace runner never fired at batch {size}")
            sharing[key]["mixed_bucket_parity"] = parity
        del engine
        torch.cuda.empty_cache()

    return json.dumps(
        {
            "probe": "grouped-bucket-policy",
            "policy_version": 2,
            "measurement_status": "measured",
            "rows": rows,
            "workspace_sharing": sharing,
            "experiment": {
                "probe_batches": probe_batches,
                "piecewise_buckets": list(piecewise_buckets),
                "context_length": context_length,
                "max_new_tokens": max_new_tokens,
                "warmup_pairs": warmup_pairs,
                "measured_pairs": measured_pairs,
                "grouped_layer_count": 4,
            },
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=30 * 60,
)
def serve_smoke() -> str:
    """Build the HTTP app far enough to run FlashInfer warmup and decode-graph capture."""
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serve import build_app_from_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    start = time.perf_counter()
    build_app_from_runtime(
        runtime,
        block_size=64,
        num_blocks=64,
        device="cuda",
        decode_graph_buckets=(1, 2),
    )
    startup_s = time.perf_counter() - start
    return json.dumps(
        {
            "app_built": True,
            "startup_s": startup_s,
            "decode_graph_buckets": [1, 2],
            "attention_backend": type(runtime.model.backend).__name__,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def capture_report(batch_sizes: list[int]) -> str:
    """The one-container decode-graph report: sync probe, launch counts, oracle-gated A/B.

    Everything runs on ONE GPU in one process so every comparison — sync warnings,
    launches/step, eager-window vs captured tok/s — is like-for-like (Modal containers vary
    ±20% machine to machine on this CPU-bound loop).
    """
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.kernels.flashinfer_paged import FlashInferPagedAttention
    from llm_infer.model.runtime import load_model_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    flash_runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
        attention_backend=FlashAttnPagedAttention(),
    )
    model = flash_runtime.model
    graph_runner = _graph_runner(flash_runtime)
    flashinfer_meta: dict[str, object] = {"available": False}
    flashinfer_runtime = None
    flashinfer_graph_runner = None
    try:
        flashinfer_runtime = load_model_runtime(
            "esme",
            bundle_path=Path(REMOTE_BUNDLE_PATH),
            dtype=torch.bfloat16,
            device="cuda",
            attention_backend=FlashInferPagedAttention(),
        )
        flashinfer_graph_runner = _graph_runner(flashinfer_runtime)
        flashinfer_meta["available"] = True
    except RuntimeError as exc:
        flashinfer_meta["error"] = str(exc)
        print(f"[flashinfer] unavailable: {exc}")

    sync = {}
    for label, active in (("eager-window", None), ("cuda-graphs", graph_runner)):
        model.decode_graphs = active
        sync[label] = _sync_probe(flash_runtime, 8)
        print(f"[sync] {label}: warnings per pass {sync[label]['sync_warnings_per_pass']}")

    launch_configs = [
        ("eager-window", flash_runtime, None),
        ("cuda-graphs", flash_runtime, graph_runner),
    ]
    if flashinfer_runtime is not None and flashinfer_graph_runner is not None:
        launch_configs.append(
            ("cuda graphs + flashinfer paged decode", flashinfer_runtime, flashinfer_graph_runner)
        )
    launches = []
    for size in batch_sizes:
        for label, row_runtime, active in launch_configs:
            row_runtime.model.decode_graphs = active
            profile = _torch_profile(row_runtime, size)
            row_runtime.model.decode_graphs = None
            launches.append({"batch_size": size, "config_label": label, **profile})
            print(
                f"[launches] batch={size} | {label}: "
                f"cudaLaunchKernel/pass {profile['cuda_launch_kernel_per_step']:.0f}, "
                f"graphLaunch/pass {profile['cuda_graph_launch_per_step']:.0f}, "
                f"gpu busy/pass {profile['gpu_busy_ms_per_step']:.2f} ms"
            )

    bench_rows = []
    for size in batch_sizes:
        for label, config in ABLATION_CONFIGS.items():
            model.decode_graphs = _runner_for(config, graph_runner)
            row = _bench_batch(oracle_runtime, flash_runtime, size, config)
            model.decode_graphs = None
            row["config_label"] = label
            bench_rows.append(row)
            tps = row["tokens_per_second"]
            print(
                f"[bench] batch={size} | {label}: median {row['median_seconds']:.3f} s, "
                f"tok/s {f'{tps:.1f}' if tps else 'NOT REPORTED (diverged)'}"
            )
        if flashinfer_runtime is not None and flashinfer_graph_runner is not None:
            flashinfer_runtime.model.decode_graphs = flashinfer_graph_runner
            row = _bench_batch(oracle_runtime, flashinfer_runtime, size, {"decode_graphs": True})
            flashinfer_runtime.model.decode_graphs = None
            row["config_label"] = "window + planned + cuda graphs + flashinfer paged decode"
            bench_rows.append(row)
            tps = row["tokens_per_second"]
            print(
                f"[bench] batch={size} | {row['config_label']}: "
                f"median {row['median_seconds']:.3f} s, "
                f"tok/s {f'{tps:.1f}' if tps else 'NOT REPORTED (diverged)'}"
            )
    return json.dumps(
        {
            "sync_probe": sync,
            "launches": launches,
            "rows": bench_rows,
            "capture_sizes": list(CAPTURE_SIZES),
            "flashinfer": flashinfer_meta,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(command: str = "bench", batch_sizes: str = "1,8,64,256", bundle_path: str = "") -> None:
    """Stage the bundle, run the selected command on the A100, write the JSON record."""
    if command not in (
        "profile",
        "sampled-profile",
        "bench",
        "ablate",
        "capture",
        "sync",
        "serve-smoke",
        "flashinfer-graph-probe",
        "two-layer-group-ab",
        "four-layer-group-ab",
        "four-layer-parity",
        "bucket-policy",
    ):
        raise ValueError(
            "command must be 'profile', 'sampled-profile', 'bench', 'ablate', 'capture', "
            "'sync', 'serve-smoke', 'flashinfer-graph-probe', 'two-layer-group-ab', "
            "'four-layer-group-ab', 'four-layer-parity', or 'bucket-policy', "
            f"got {command!r}"
        )
    sizes = _parse_batch_sizes(batch_sizes)
    if command != "flashinfer-graph-probe":
        local_bundle = local_bundle_path(bundle_path)
        stage_bundle(esme_bundles, local_bundle, label="esme-decode")

    if command == "flashinfer-graph-probe":
        print("[esme-decode] FlashInfer fixed-buffer graph probe: exact batches 1,8")
        record = json.loads(flashinfer_graph_probe.remote())
    elif command == "two-layer-group-ab":
        print(f"[esme-decode] cache-owned two-layer graph A/B: exact batches {sizes}")
        record = json.loads(grouped_layer_ab.remote(2, False, sizes))
    elif command == "four-layer-group-ab":
        print(f"[esme-decode] cache-owned four-layer graph A/B: exact batches {sizes}")
        record = json.loads(grouped_layer_ab.remote(4, False, sizes))
    elif command == "four-layer-parity":
        print("[esme-decode] cache-owned four-layer graph: batch-8 correctness only")
        record = json.loads(grouped_layer_ab.remote(4, True))
    elif command == "bucket-policy":
        print(f"[esme-decode] grouped bucket-policy record: off-bucket batches {sizes}")
        record = json.loads(bucket_policy_report.remote(sizes))
    elif command == "profile":
        print(
            f"[esme-decode] {command}: batches {sizes}, {MAX_NEW_TOKENS} new tokens, greedy, "
            f"bf16 default attention, prefix caching off"
        )
        record = json.loads(profile_decode.remote(sizes))
    elif command == "sampled-profile":
        print(
            f"[esme-decode] {command}: batches {sizes}, {MAX_NEW_TOKENS} new tokens, "
            f"greedy vs sampled {SAMPLED_SETTINGS}, bf16 default attention, prefix caching off"
        )
        record = json.loads(profile_sampled_decode.remote(sizes))
    elif command == "ablate":
        record = json.loads(ablate_decode.remote(sizes))
    elif command == "capture":
        record = json.loads(capture_report.remote(sizes))
    elif command == "sync":
        record = json.loads(sync_report.remote(sizes))
    elif command == "serve-smoke":
        record = json.loads(serve_smoke.remote())
    else:
        record = json.loads(bench_decode.remote(sizes))
    record["config"] = {
        "command": command,
        "model": "Esme-214M-Chat",
        "batch_sizes": (
            [8]
            if command == "four-layer-parity"
            else [1, 8]
            if command == "flashinfer-graph-probe"
            else sizes
        ),
        "max_new_tokens": (
            None
            if command == "flashinfer-graph-probe"
            else 128
            if command
            in ("two-layer-group-ab", "four-layer-group-ab", "four-layer-parity", "bucket-policy")
            else MAX_NEW_TOKENS
        ),
        "warmup": (
            3
            if command == "flashinfer-graph-probe"
            else 0
            if command == "four-layer-parity"
            else 2
            if command in ("two-layer-group-ab", "four-layer-group-ab", "bucket-policy")
            else WARMUP_ITERS
        ),
        "iters": (
            1
            if command in ("flashinfer-graph-probe", "four-layer-parity")
            else 10
            if command in ("two-layer-group-ab", "four-layer-group-ab")
            else 6
            if command == "bucket-policy"
            else MEASURED_ITERS
        ),
        "backend": (
            "FlashInfer BatchDecodeWithPagedKVCacheWrapper (bf16)"
            if command == "flashinfer-graph-probe"
            else (
                f"Esme FlashInfer piecewise vs cache-owned "
                f"{'four' if command in ('four-layer-group-ab', 'four-layer-parity') else 'two'}"
                "-layer graph (bf16)"
                if command in ("two-layer-group-ab", "four-layer-group-ab", "four-layer-parity")
                else f"{record.get('attention_backend', 'comparison')} (bf16)"
            )
        ),
        "reference": (
            "fresh ordinary FlashInfer wrapper planned for the exact page metadata"
            if command == "flashinfer-graph-probe"
            else "none (diagnostic attribution only, never a speed claim)"
            if command == "sampled-profile"
            else "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant"
        ),
        "repro_command": (
            f"modal run scripts/modal_esme_decode_profile.py --command {command}"
            if command
            in (
                "flashinfer-graph-probe",
                "two-layer-group-ab",
                "four-layer-group-ab",
                "four-layer-parity",
            )
            else (
                f"modal run scripts/modal_esme_decode_profile.py --command {command} "
                f"--batch-sizes {batch_sizes}"
            )
        ),
    }

    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-decode-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-decode] wrote {out_path}")
