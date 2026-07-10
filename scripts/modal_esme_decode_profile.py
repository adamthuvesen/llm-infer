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
  the audited tie-tolerant rule. A row that diverges beyond genuine ties reports no tok/s.

    modal run scripts/modal_esme_decode_profile.py --command profile
    modal run scripts/modal_esme_decode_profile.py --command bench
    modal run scripts/modal_esme_decode_profile.py --command bench --batch-sizes 8,32,128
    modal run scripts/modal_esme_decode_profile.py --command capture --batch-sizes 8,64,256
    modal run scripts/modal_esme_decode_profile.py --command serve-smoke

``--command capture`` is the decode-graph report: in ONE container it runs the sync-debug
probe, launch counts, and the oracle-gated same-GPU ablation across the eager window, the
manual piecewise CUDA-graph runner, and the torch.compile runner (opaque paged-attention
custom op, ``mode='reduce-overhead'``), plus a recompile audit over the compiled rows.
"""

from __future__ import annotations

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
    (int) and ``planned_decode`` (bool, masks the capability flag).
    """
    from dataclasses import replace

    from llm_infer.benchmarks.esme_paged import build_requests
    from llm_infer.serving import InferenceEngine, Request

    config = config or {}
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
            Request(req.request_id, list(req.prompt_ids), MAX_NEW_TOKENS, runtime.eos_token_ids)
        )
    return engine, requests


def _run_prefill_step(engine) -> None:
    """Advance the engine through its first step (prefill + first sampled token)."""
    engine.step()


def _decode_wall(runtime, num_requests: int) -> dict:
    """Un-instrumented decode timing: wall per decode step after prefill, one synced run."""
    import torch

    engine, _ = _build_engine(runtime, num_requests, "cuda")
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


def _python_profile(runtime, num_requests: int) -> dict:
    """Function-level Python attribution of the decode loop via cProfile (diagnostic only)."""
    import cProfile
    import pstats

    engine, _ = _build_engine(runtime, num_requests, "cuda")
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


def _phase_profile(runtime, num_requests: int) -> dict:
    """One diagnostic generation projected to the Phase 0 timing buckets."""
    import torch

    from llm_infer.profiling import TimingProfiler, attach_host_method_profile

    profiler = TimingProfiler("cuda")
    engine, _ = _build_engine(runtime, num_requests, "cuda", {"profiler": profiler})
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
        "matches_reference": agreement.all_ties_or_exact,
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
        "tokens_per_second": tokens / median_s if agreement.all_ties_or_exact else None,
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
# ``decode_graphs``/``decode_compile`` are model-level toggles (each runner is built once
# and reused), applied by the harness around each row rather than by ``_build_engine``.
ABLATION_CONFIGS: dict[str, dict] = {
    "per-step (window=1)": {"decode_window_size": 1},
    "window, classic decode_many": {"planned_decode": False},
    "window + planned buffers (default)": {},
    "window + planned + cuda graphs": {"decode_graphs": True},
    "window + planned + torch.compile": {"decode_compile": True},
}


def _graph_runner(flash_runtime):
    """Capture the decode-graph buckets once (outside any timed region) and return the runner."""
    start = time.perf_counter()
    runner = flash_runtime.model.enable_decode_graphs(CAPTURE_SIZES)
    flash_runtime.model.decode_graphs = None  # rows opt in explicitly
    print(f"[graphs] captured buckets {CAPTURE_SIZES} in {time.perf_counter() - start:.1f} s")
    return runner


def _compile_runner(flash_runtime):
    """Compile+warm the torch.compile runner once; returns (runner, startup metadata).

    Tries ``mode='reduce-overhead'`` (compiler-managed CUDA graphs) first. The runner's
    enable-time parity check raises when the torch build CUDA-graph-captures through the
    opaque attention op; the fallback then recompiles without compiler-managed graphs so
    the row still measures honest Inductor fusion instead of decoding garbage.
    """
    import torch._dynamo

    torch._logging.set_logs(recompiles=True)  # any mid-bench recompile shows in the log
    start = time.perf_counter()
    try:
        runner = flash_runtime.model.enable_decode_compile(CAPTURE_SIZES)
        mode = "reduce-overhead"
    except RuntimeError as exc:
        print(f"[compile] reduce-overhead failed the enable-time parity check: {exc}")
        print("[compile] falling back to mode=None (Inductor fusion, no cudagraphs)")
        torch._dynamo.reset()
        runner = flash_runtime.model.enable_decode_compile(CAPTURE_SIZES, mode=None)
        mode = "none (fallback)"
    flash_runtime.model.decode_graphs = None  # rows opt in explicitly
    elapsed = time.perf_counter() - start
    print(
        f"[compile] compiled+warmed buckets {CAPTURE_SIZES} in {elapsed:.1f} s "
        f"(mode={mode}, warmup_s={runner.warmup_s:.1f})"
    )
    return runner, {"mode": mode, "compile_and_warmup_s": elapsed}


def _runner_for(config: dict, graph_runner, compile_runner):
    if config.get("decode_graphs"):
        return graph_runner
    if config.get("decode_compile"):
        return compile_runner
    return None


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
    compile_runner, compile_meta = _compile_runner(flash_runtime)
    rows = []
    for size in batch_sizes:
        for label, config in ABLATION_CONFIGS.items():
            flash_runtime.model.decode_graphs = _runner_for(config, graph_runner, compile_runner)
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
            "compile": compile_meta,
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
    compile_runner, compile_meta = _compile_runner(flash_runtime)
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
        ("torch-compile", flash_runtime, compile_runner),
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

    # Recompile audit: from here to the end of the bench rows, the compiled path must not
    # build a single new Dynamo graph — recompiles in steady state are the failure mode the
    # tensors-not-ints design exists to prevent (reasons would show via TORCH_LOGS).
    from llm_infer.model.decode_compile import dynamo_counters_snapshot

    counters_before = dynamo_counters_snapshot()
    bench_rows = []
    for size in batch_sizes:
        for label, config in ABLATION_CONFIGS.items():
            model.decode_graphs = _runner_for(config, graph_runner, compile_runner)
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
    counters_after = dynamo_counters_snapshot()
    recompile_audit = {
        "before_bench": counters_before,
        "after_bench": counters_after,
        "new_graphs_during_bench": counters_after["unique_graphs"]
        - counters_before["unique_graphs"],
    }
    print(f"[compile] recompile audit: {recompile_audit}")

    return json.dumps(
        {
            "sync_probe": sync,
            "launches": launches,
            "rows": bench_rows,
            "capture_sizes": list(CAPTURE_SIZES),
            "compile": {**compile_meta, "recompile_audit": recompile_audit},
            "flashinfer": flashinfer_meta,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(command: str = "bench", batch_sizes: str = "1,8,64,256", bundle_path: str = "") -> None:
    """Stage the bundle, run the selected command on the A100, write the JSON record."""
    if command not in ("profile", "bench", "ablate", "capture", "sync", "serve-smoke"):
        raise ValueError(
            "command must be 'profile', 'bench', 'ablate', 'capture', 'sync', "
            f"or 'serve-smoke', got {command!r}"
        )
    sizes = _parse_batch_sizes(batch_sizes)
    local_bundle = local_bundle_path(bundle_path)
    stage_bundle(esme_bundles, local_bundle, label="esme-decode")

    print(
        f"[esme-decode] {command}: batches {sizes}, {MAX_NEW_TOKENS} new tokens, greedy, "
        f"bf16 default attention, prefix caching off"
    )
    if command == "profile":
        record = json.loads(profile_decode.remote(sizes))
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
        "batch_sizes": sizes,
        "max_new_tokens": MAX_NEW_TOKENS,
        "warmup": WARMUP_ITERS,
        "iters": MEASURED_ITERS,
        "backend": f"{record.get('attention_backend', 'comparison')} (bf16)",
        "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant",
        "repro_command": (
            f"modal run scripts/modal_esme_decode_profile.py --command {command} "
            f"--batch-sizes {batch_sizes}"
        ),
    }

    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-decode-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-decode] wrote {out_path}")
