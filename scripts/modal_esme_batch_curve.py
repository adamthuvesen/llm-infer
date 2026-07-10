"""Modal A100 batch-size curve for Esme-214M-Chat: llm_infer vs the naive-HF floor.

The centerpiece figure behind ``assets/fig-esme-batch-curve.svg``: engine throughput as the
batch grows from 8 to 256 concurrent chat requests, against the measured naive
HF-sequential floor. Same workload family as the headline benchmark (HEADLINE_PROMPTS
pool, up to 256 new tokens, greedy, prefix caching off) and the same gate: every reported
row must agree with the fp32 ``PretrainBundleModel.logits()`` oracle under the audited
reference policy v2; raw timing is retained while public tok/s stays gated.

Same-container methodology (cross-container A100 variance is ±20% for this CPU-bound
decode; see docs/benchmark.md): every llm_infer row and every HF floor row runs in ONE
flash-image container, back to back, at every batch level. The floor's anchor level runs
the full warmup+3-iteration protocol; the larger levels run one measured iteration each —
one HF iteration at batch b is itself b sequential requests, and the anchor's iteration
spread is under 1%. The vLLM ceiling rows exist in the harness for private measurement
(``--skip-vllm`` omits them); they are not part of the published story.

The llm_infer batch-256 row also records peak KV-pool usage — the paged-KV evidence for
how the right side of the curve fits in memory.

    modal run scripts/modal_esme_batch_curve.py --command smoke   # tiny shape, 1 iter
    modal run scripts/modal_esme_batch_curve.py --command curve   # the published sweep
"""

from __future__ import annotations

import dataclasses
import json
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
from scripts.modal_flash_image import (
    FLASH_IMAGE,
    IGNORE,
    REMOTE_ROOT,
    REPO_ROOT,
)

ESME_HF_DIR = "esme-214m-chat-hf"
REMOTE_HF_PATH = f"{ESME_BUNDLE_MOUNT}/{ESME_HF_DIR}"
BLOCK_SIZE = 128
# Decode-graph buckets for the llm_infer rows: the default spread plus 256, so the curve's
# largest batch replays from graphs instead of silently falling back to the eager window.
CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

app = modal.App("llm-infer-esme-batch-curve")

# Same vLLM image recipe as scripts/modal_esme_three_way.py: vLLM pulls its own torch/CUDA
# stack, so it cannot share the flash image (and therefore cannot share its container).
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
    .env(
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=IGNORE)
    .workdir(REMOTE_ROOT)
    .run_commands("pip install --no-deps -e .")
)

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=FLASH_IMAGE, volumes={ESME_BUNDLE_MOUNT: esme_bundles}, timeout=30 * 60)
def convert_bundle_to_hf() -> str:
    """Convert the staged bundle to the HF Qwen3 checkpoint on the volume (remote, has torch).

    Mirrors the three-way harness's conversion step; each app owns its copy because Modal
    functions bind to their app. CPU-only key remap; ``commit()`` publishes the files.
    """
    from scripts.convert_esme_to_hf import convert

    config = convert(Path(REMOTE_BUNDLE_PATH), Path(REMOTE_HF_PATH), max_position_embeddings=1024)
    esme_bundles.commit()
    layers = config["num_hidden_layers"]
    return f"converted HF Qwen3 checkpoint -> {REMOTE_HF_PATH} (layers={layers})"


class _PeakBlocks:
    """Allocator observer that tracks the KV pool's peak used-block count."""

    def __init__(self) -> None:
        self.peak = 0

    def __call__(self, event) -> None:  # noqa: ANN001 - BlockPoolEvent, imported remotely
        self.peak = max(self.peak, event.num_used)


@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def bench_vllm_batches(batch_sizes: list[int], max_new_tokens: int, warmup: int, iters: int) -> str:
    """vLLM ceiling rows for every batch size, all in this one container."""
    import torch

    from llm_infer.benchmarks import gpu_snapshot
    from llm_infer.benchmarks.esme_paged import HEADLINE_PROMPTS, _time, build_requests
    from llm_infer.benchmarks.esme_three_way import build_vllm_llm, vllm_decode_closure
    from llm_infer.model.runtime import load_model_runtime

    esme_bundles.reload()
    runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cpu"
    )
    # ONE engine for all batch sizes: a second LLM() in this process fails startup because
    # the first engine's memory reservation is never returned. The prompt pool is shared
    # across batch sizes, so one max_model_len covers every row.
    pool_requests = build_requests(runtime.tokenizer, max(batch_sizes), HEADLINE_PROMPTS)
    max_model_len = max(len(req.prompt_ids) for req in pool_requests) + max_new_tokens
    llm, vllm_config = build_vllm_llm(Path(REMOTE_HF_PATH), max_model_len=max_model_len)
    rows = []
    for size in batch_sizes:
        requests = build_requests(runtime.tokenizer, size, HEADLINE_PROMPTS)
        decode_once = vllm_decode_closure(
            llm,
            requests,
            max_new_tokens=max_new_tokens,
            eos_token_ids=runtime.eos_token_ids,
        )
        median_s, outputs = _time(decode_once, warmup=warmup, iters=iters, sync=False)
        rows.append(
            {
                "batch_size": size,
                "median_seconds": float(median_s),
                "outputs": {k: [int(x) for x in v] for k, v in outputs.items()},
            }
        )
        print(f"[curve/vllm] batch={size}: median {median_s:.3f} s")
    return json.dumps({"rows": rows, "config": vllm_config, "gpu": gpu_snapshot()})


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=3 * 60 * 60,
)
def sweep_baseline(
    batch_sizes: list[int],
    context_lengths: list[int],
    max_new_tokens: int,
    warmup: int,
    iters: int,
) -> str:
    """Reference-gated Phase 0 matrix with cold startup and persistent timing separated."""
    import math
    import statistics

    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_paged import (
        HEADLINE_PROMPTS,
        build_requests,
        requests_at_context_length,
        single_request_prompt_coverage,
    )
    from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
    from llm_infer.benchmarks.reference_policy import (
        QUALIFIED_REFERENCE_STATUSES,
        build_reference_only_record,
        build_system_evidence_record,
    )
    from llm_infer.benchmarks.report import total_output_tokens
    from llm_infer.model.decode import greedy_decode
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.profiling import TimingProfiler, attach_host_method_profile
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    engine_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    # This diagnostic measures exactly ``max_new_tokens`` decode steps at every synthetic
    # context length. Natural EOS would turn some repeated-token contexts into one-token rows.
    eos = frozenset()
    capture_s = enable_decode_graphs_if_cuda(engine_runtime.model, CAPTURE_SIZES)
    reference_by_prompt: dict[tuple[int, ...], list[int]] = {}

    def p95(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[math.ceil(0.95 * len(ordered)) - 1]

    def matrix_requests(size: int, context_length: int):
        return requests_at_context_length(
            build_requests(engine_runtime.tokenizer, size, HEADLINE_PROMPTS), context_length
        )

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

    def agreement_record(requests, outputs):
        agreement = tie_tolerant_agreement(
            oracle_runtime.model, requests, outputs, reference_for(requests), eos
        )
        return (
            {
                "exact": agreement.exact,
                "tie": agreement.tie,
                "nontie": agreement.nontie,
                "total": agreement.total,
                "ties_sample": agreement.ties_sample,
                "divergences_sample": agreement.divergences_sample,
                "review_required": agreement.review_required,
                "failed": agreement.failed,
                "numerical_evidence": agreement.numerical_evidence,
            },
            agreement,
        )

    rows: list[dict] = []
    for context_length in context_lengths:
        for size in batch_sizes:
            requests = matrix_requests(size, context_length)
            needed = sum(
                math.ceil((len(request.prompt_ids) + max_new_tokens) / BLOCK_SIZE)
                for request in requests
            )
            num_blocks = needed + max(4, len(requests))
            peak = _PeakBlocks()

            torch.cuda.synchronize()
            startup_start = time.perf_counter()
            engine = InferenceEngine(
                engine_runtime.model,
                block_size=BLOCK_SIZE,
                num_blocks=num_blocks,
                device="cuda",
                capabilities=engine_runtime.capabilities,
            )
            torch.cuda.synchronize()
            engine_kv_startup_s = time.perf_counter() - startup_start

            def persistent_once(
                requests=requests, engine=engine
            ) -> dict[str, list[int]]:
                for request in requests:
                    engine.add_request(
                        Request(
                            request.request_id,
                            list(request.prompt_ids),
                            max_new_tokens,
                            eos,
                        )
                    )
                return engine.run()

            for _ in range(warmup):
                persistent_once()
                torch.cuda.synchronize()
            per_iter: list[float] = []
            outputs: dict[str, list[int]] = {}
            for _ in range(iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = persistent_once()
                torch.cuda.synchronize()
                per_iter.append(time.perf_counter() - start)

            agreement, agreement_assessment = agreement_record(requests, outputs)
            tokens = total_output_tokens(outputs, eos)
            expected_tokens = size * max_new_tokens
            if tokens != expected_tokens:
                raise AssertionError(
                    f"baseline row produced {tokens} tokens, expected {expected_tokens}"
                )
            median_s = statistics.median(per_iter)
            policy = build_system_evidence_record(
                agreement=agreement_assessment, median_seconds=median_s, total_tokens=tokens
            )

            profiler = TimingProfiler("cuda")
            profile_engine = InferenceEngine(
                engine_runtime.model,
                block_size=BLOCK_SIZE,
                num_blocks=num_blocks,
                device="cuda",
                capabilities=engine_runtime.capabilities,
                profiler=profiler,
            )
            # KV peak is diagnostic. Keep allocator callbacks out of steady-state timing.
            profile_engine.cache.allocator.observer = peak
            attach_host_method_profile(
                profiler,
                profile_engine,
                "window_flushing",
                ("_stage_window_flush", "_consume_window_flush"),
            )
            for request in requests:
                profile_engine.add_request(
                    Request(
                        request.request_id,
                        list(request.prompt_ids),
                        max_new_tokens,
                        eos,
                    )
                )
            profile_engine.run()
            torch.cuda.synchronize()
            phase_profile = profiler.summary().as_dict()

            model = engine_runtime.model
            kv_bytes_per_token = (
                model.num_layers
                * 2
                * model.num_kv_heads
                * model.head_dim
                * model.dtype.itemsize
            )
            row = {
                "system": "llm_infer_persistent",
                "batch_size": size,
                "context_length": context_length,
                "workload_request_ids": [request.request_id for request in requests],
                "reference_scope": "measured_workload_only",
                "agreement": agreement,
                "engine_kv_startup_seconds": engine_kv_startup_s,
                "steady_state_median_seconds": median_s,
                "steady_state_p95_seconds": p95(per_iter),
                "steady_state_per_iter_seconds": per_iter,
                "total_output_tokens": tokens,
                **policy,
                "phase_profile": phase_profile,
                "kv_pool": {
                    "block_size": BLOCK_SIZE,
                    "num_blocks": num_blocks,
                    "peak_used_blocks": peak.peak,
                    "peak_kv_bytes": peak.peak * BLOCK_SIZE * kv_bytes_per_token,
                    "kv_bytes_per_token": kv_bytes_per_token,
                },
            }
            rows.append(row)
            tps = row["tokens_per_second"]
            print(
                f"[baseline/llm_infer] context={context_length} batch={size}: "
                f"startup {engine_kv_startup_s:.3f} s, steady {median_s:.3f} s, "
                f"tok/s {f'{tps:.1f}' if tps else 'NOT REPORTED (diverged)'}"
            )

    coverage_rows: list[dict[str, object]] = []
    for context_length, request in single_request_prompt_coverage(
        engine_runtime.tokenizer, tuple(context_lengths), HEADLINE_PROMPTS
    ):
        needed = math.ceil((len(request.prompt_ids) + max_new_tokens) / BLOCK_SIZE)
        coverage_engine = InferenceEngine(
            engine_runtime.model,
            block_size=BLOCK_SIZE,
            num_blocks=needed + 2,
            device="cuda",
            capabilities=engine_runtime.capabilities,
        )
        coverage_engine.add_request(
            Request(
                request.request_id,
                list(request.prompt_ids),
                max_new_tokens,
                eos,
            )
        )
        outputs = coverage_engine.run()
        agreement, agreement_assessment = agreement_record([request], outputs)
        policy = build_reference_only_record(agreement_assessment)
        coverage_rows.append(
            {
                "context_length": context_length,
                "request_id": request.request_id,
                "prompt": request.prompt,
                "agreement": agreement,
                **policy,
            }
        )

    coverage_by_context = {
        context_length: all(
            row["reference_status"] in QUALIFIED_REFERENCE_STATUSES
            for row in coverage_rows
            if row["context_length"] == context_length
        )
        for context_length in context_lengths
    }
    for row in rows:
        row["all_prompt_single_request_coverage_passed"] = coverage_by_context[
            row["context_length"]
        ]

    return json.dumps(
        {
            "rows": rows,
            "single_request_prompt_coverage": coverage_rows,
            "decode_graphs": {"capture_sizes": list(CAPTURE_SIZES), "capture_s": capture_s},
            "attention_backend": type(engine_runtime.model.backend).__name__,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=2 * 60 * 60,
)
def sweep(
    llm_infer_batches: list[int],
    hf_specs: list[list[int]],
    max_new_tokens: int,
    warmup: int,
    iters: int,
    vllm_rows: list[dict],
) -> str:
    """All llm_infer and HF-floor rows in ONE container, plus the oracle gate for every row."""
    import math
    import statistics

    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_paged import HEADLINE_PROMPTS, build_requests
    from llm_infer.benchmarks.esme_three_way import (
        run_hf_sequential_esme,
        tie_tolerant_agreement,
    )
    from llm_infer.benchmarks.reference_policy import build_system_evidence_record
    from llm_infer.benchmarks.report import normalize_at_eos, total_output_tokens
    from llm_infer.model.decode import greedy_decode
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    engine_runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    eos = oracle_runtime.eos_token_ids
    # The llm_infer rows run what serving runs: decode-window CUDA graphs, captured once up
    # front so no capture cost lands inside a timed iteration. The oracle stays eager fp32.
    capture_s = enable_decode_graphs_if_cuda(engine_runtime.model, CAPTURE_SIZES)
    print(
        f"[curve] {type(engine_runtime.model.backend).__name__}: "
        f"captured {CAPTURE_SIZES} in {capture_s:.1f} s"
    )

    # fp32 oracle greedy decode once per unique prompt, shared by every row in this record.
    reference_by_prompt: dict[tuple[int, ...], list[int]] = {}

    def reference_for(requests) -> dict[str, list[int]]:
        for req in requests:
            if req.prompt_ids not in reference_by_prompt:
                reference_by_prompt[req.prompt_ids] = greedy_decode(
                    oracle_runtime.model,
                    list(req.prompt_ids),
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=set(eos),
                )
        return {req.request_id: list(reference_by_prompt[req.prompt_ids]) for req in requests}

    def gated_row(system: str, size: int, requests, outputs, per_iter: list[float]) -> dict:
        reference = reference_for(requests)
        agreement = tie_tolerant_agreement(oracle_runtime.model, requests, outputs, reference, eos)
        tokens = total_output_tokens(outputs, eos)
        median_s = statistics.median(per_iter)
        return {
            "system": system,
            "batch_size": size,
            "agreement": dataclasses.asdict(agreement),
            "median_seconds": median_s,
            "per_iter_seconds": per_iter,
            "total_output_tokens": tokens,
            **build_system_evidence_record(
                agreement=agreement, median_seconds=median_s, total_tokens=tokens
            ),
        }

    rows: list[dict] = []

    # llm_infer rows: fresh engine per iteration, serving defaults (window + planned buffers
    # + decode graphs; the runner is model-owned, so every fresh engine replays the same
    # captured buckets).
    kv_evidence: dict[str, object] = {}
    for size in llm_infer_batches:
        requests = build_requests(engine_runtime.tokenizer, size, HEADLINE_PROMPTS)
        needed = sum(
            math.ceil((len(req.prompt_ids) + max_new_tokens) / BLOCK_SIZE) for req in requests
        )
        num_blocks = needed + max(4, len(requests))
        peak = _PeakBlocks()

        def decode_once(
            requests=requests, num_blocks=num_blocks, peak=peak
        ) -> dict[str, list[int]]:
            engine = InferenceEngine(
                engine_runtime.model,
                block_size=BLOCK_SIZE,
                num_blocks=num_blocks,
                device="cuda",
                capabilities=engine_runtime.capabilities,
            )
            engine.cache.allocator.observer = peak
            for req in requests:
                engine.add_request(
                    Request(req.request_id, list(req.prompt_ids), max_new_tokens, eos)
                )
            return engine.run()

        for _ in range(warmup):
            decode_once()
            torch.cuda.synchronize()
        per_iter: list[float] = []
        outputs: dict[str, list[int]] = {}
        for _ in range(iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = decode_once()
            torch.cuda.synchronize()
            per_iter.append(time.perf_counter() - start)

        row = gated_row("llm_infer", size, requests, outputs, per_iter)
        model = engine_runtime.model
        kv_bytes_per_token = (
            model.num_layers * 2 * model.num_kv_heads * model.head_dim * model.dtype.itemsize
        )
        row["kv_pool"] = {
            "block_size": BLOCK_SIZE,
            "num_blocks": num_blocks,
            "peak_used_blocks": peak.peak,
            "peak_kv_bytes": peak.peak * BLOCK_SIZE * kv_bytes_per_token,
            "kv_bytes_per_token": kv_bytes_per_token,
        }
        kv_evidence[str(size)] = row["kv_pool"]
        rows.append(row)
        tps = row["tokens_per_second"]
        print(
            f"[curve/llm_infer] batch={size}: median {row['median_seconds']:.3f} s, "
            f"tok/s {f'{tps:.1f}' if tps else 'NOT REPORTED (diverged)'}, "
            f"peak KV blocks {peak.peak}/{num_blocks}"
        )

    # HF floor rows, same container: full protocol at the first size, flatness checks after.
    for spec in hf_specs:
        size, hf_iters = spec
        requests = build_requests(engine_runtime.tokenizer, size, HEADLINE_PROMPTS)
        decode_once = run_hf_sequential_esme(
            Path(REMOTE_HF_PATH),
            requests,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos,
            device="cuda",
        )
        hf_warmup = 1 if hf_iters > 1 else 0
        for _ in range(hf_warmup):
            decode_once()
            torch.cuda.synchronize()
        per_iter = []
        outputs = {}
        for _ in range(hf_iters):
            torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = decode_once()
            torch.cuda.synchronize()
            per_iter.append(time.perf_counter() - start)
        row = gated_row("hf_sequential", size, requests, outputs, per_iter)
        row["flatness_check"] = hf_iters == 1
        rows.append(row)
        tps = row["tokens_per_second"]
        print(
            f"[curve/hf] batch={size}: median {row['median_seconds']:.3f} s, "
            f"tok/s {f'{tps:.1f}' if tps else 'NOT REPORTED (diverged)'}"
        )

    # vLLM ceiling rows measured in their own container; gate their tokens here with the
    # same oracle so a diverging ceiling row reports no tok/s either.
    for vllm_row in vllm_rows:
        size = vllm_row["batch_size"]
        requests = build_requests(engine_runtime.tokenizer, size, HEADLINE_PROMPTS)
        outputs = {k: normalize_at_eos(v, eos) for k, v in vllm_row["outputs"].items()}
        row = gated_row("vllm", size, requests, outputs, [float(vllm_row["median_seconds"])])
        row["separate_container"] = True
        rows.append(row)
        tps = row["tokens_per_second"]
        print(
            f"[curve/vllm-gate] batch={size}: "
            f"tok/s {f'{tps:.1f}' if tps else 'NOT REPORTED (diverged)'}"
        )

    return json.dumps(
        {
            "rows": rows,
            "kv_evidence": kv_evidence,
            "decode_graphs": {"capture_sizes": list(CAPTURE_SIZES), "capture_s": capture_s},
            "attention_backend": type(engine_runtime.model.backend).__name__,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=2 * 60 * 60,
)
def sweep_flashinfer_compare(
    batch_sizes: list[int],
    max_new_tokens: int,
    warmup: int,
    iters: int,
) -> str:
    """Headline workload A/B: current llm_infer graph path vs FlashInfer paged decode."""
    import math
    import statistics

    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_paged import HEADLINE_PROMPTS, build_requests
    from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
    from llm_infer.benchmarks.reference_policy import build_system_evidence_record
    from llm_infer.benchmarks.report import total_output_tokens
    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.kernels.flashinfer_paged import FlashInferPagedAttention
    from llm_infer.model.decode import greedy_decode
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    runtimes = [
        (
            "llm_infer",
            load_model_runtime(
                "esme",
                bundle_path=Path(REMOTE_BUNDLE_PATH),
                dtype=torch.bfloat16,
                device="cuda",
                attention_backend=FlashAttnPagedAttention(),
            ),
        ),
        (
            "llm_infer_flashinfer_paged",
            load_model_runtime(
                "esme",
                bundle_path=Path(REMOTE_BUNDLE_PATH),
                dtype=torch.bfloat16,
                device="cuda",
                attention_backend=FlashInferPagedAttention(),
            ),
        ),
    ]
    graph_captures: dict[str, float | None] = {}
    for system, runtime in runtimes:
        capture_s = enable_decode_graphs_if_cuda(runtime.model, CAPTURE_SIZES)
        graph_captures[system] = capture_s
        print(f"[flashinfer-curve] {system}: captured {CAPTURE_SIZES} in {capture_s:.1f} s")

    eos = oracle_runtime.eos_token_ids
    reference_by_prompt: dict[tuple[int, ...], list[int]] = {}

    def reference_for(requests) -> dict[str, list[int]]:
        for req in requests:
            if req.prompt_ids not in reference_by_prompt:
                reference_by_prompt[req.prompt_ids] = greedy_decode(
                    oracle_runtime.model,
                    list(req.prompt_ids),
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=set(eos),
                )
        return {req.request_id: list(reference_by_prompt[req.prompt_ids]) for req in requests}

    def gated_row(
        system: str, size: int, runtime, requests, outputs, per_iter: list[float]
    ) -> dict:
        reference = reference_for(requests)
        agreement = tie_tolerant_agreement(
            oracle_runtime.model, requests, outputs, reference, eos
        )
        tokens = total_output_tokens(outputs, eos)
        median_s = statistics.median(per_iter)
        model = runtime.model
        kv_bytes_per_token = (
            model.num_layers * 2 * model.num_kv_heads * model.head_dim * model.dtype.itemsize
        )
        return {
            "system": system,
            "batch_size": size,
            "agreement": dataclasses.asdict(agreement),
            "median_seconds": median_s,
            "per_iter_seconds": per_iter,
            "total_output_tokens": tokens,
            **build_system_evidence_record(
                agreement=agreement, median_seconds=median_s, total_tokens=tokens
            ),
            "kv_bytes_per_token": kv_bytes_per_token,
        }

    rows: list[dict] = []
    for size in batch_sizes:
        for system, runtime in runtimes:
            requests = build_requests(runtime.tokenizer, size, HEADLINE_PROMPTS)
            needed = sum(
                math.ceil((len(req.prompt_ids) + max_new_tokens) / BLOCK_SIZE)
                for req in requests
            )
            num_blocks = needed + max(4, len(requests))
            peak = _PeakBlocks()

            def decode_once(
                runtime=runtime,
                requests=requests,
                num_blocks=num_blocks,
                peak=peak,
            ) -> dict[str, list[int]]:
                engine = InferenceEngine(
                    runtime.model,
                    block_size=BLOCK_SIZE,
                    num_blocks=num_blocks,
                    device="cuda",
                    capabilities=runtime.capabilities,
                )
                engine.cache.allocator.observer = peak
                for req in requests:
                    engine.add_request(
                        Request(req.request_id, list(req.prompt_ids), max_new_tokens, eos)
                    )
                return engine.run()

            for _ in range(warmup):
                decode_once()
                torch.cuda.synchronize()
            per_iter: list[float] = []
            outputs: dict[str, list[int]] = {}
            for _ in range(iters):
                torch.cuda.synchronize()
                start = time.perf_counter()
                outputs = decode_once()
                torch.cuda.synchronize()
                per_iter.append(time.perf_counter() - start)

            row = gated_row(system, size, runtime, requests, outputs, per_iter)
            row["kv_pool"] = {
                "block_size": BLOCK_SIZE,
                "num_blocks": num_blocks,
                "peak_used_blocks": peak.peak,
                "peak_kv_bytes": peak.peak * BLOCK_SIZE * row["kv_bytes_per_token"],
            }
            rows.append(row)
            tps = row["tokens_per_second"]
            print(
                f"[flashinfer-curve] batch={size} | {system}: "
                f"median {row['median_seconds']:.3f} s, "
                f"tok/s {f'{tps:.1f}' if tps else 'NOT REPORTED (diverged)'}, "
                f"agreement exact/tie/nontie "
                f"{row['agreement']['exact']}/{row['agreement']['tie']}/"
                f"{row['agreement']['nontie']}"
            )

    return json.dumps(
        {
            "rows": rows,
            "decode_graphs": {"capture_sizes": list(CAPTURE_SIZES), "capture_s": graph_captures},
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(command: str = "curve", bundle_path: str = "", skip_vllm: bool = False) -> None:
    """Stage + convert the bundle, run the (optional) vLLM rows, then the one-container sweep."""
    if command == "smoke":
        llm_infer_batches = [2, 4]
        context_lengths: list[int] = []
        hf_specs = [[2, 1]]
        vllm_batches = [2]
        max_new_tokens, warmup, iters = 8, 0, 1
    elif command == "curve":
        llm_infer_batches = [1, 8, 16, 32, 64, 128, 256]
        context_lengths = []
        hf_specs = [[8, 3], [16, 1], [32, 1], [64, 1], [128, 1], [256, 1]]
        vllm_batches = [1, 8, 64, 256]
        max_new_tokens, warmup, iters = 256, 1, 3
    elif command == "baseline":
        llm_infer_batches = [1, 8, 64, 256]
        context_lengths = [32, 256, 768]
        hf_specs = []
        vllm_batches = []
        max_new_tokens, warmup, iters = 128, 2, 10
    elif command == "baseline-smoke":
        llm_infer_batches = [1]
        context_lengths = [32]
        hf_specs = []
        vllm_batches = []
        max_new_tokens, warmup, iters = 8, 0, 1
    elif command == "flashinfer-smoke":
        llm_infer_batches = [8]
        context_lengths = []
        hf_specs = []
        vllm_batches = []
        max_new_tokens, warmup, iters = 16, 1, 1
    elif command == "flashinfer-curve":
        llm_infer_batches = [1, 8, 16, 32, 64, 128, 256]
        context_lengths = []
        hf_specs = []
        vllm_batches = []
        max_new_tokens, warmup, iters = 256, 1, 3
    else:
        raise ValueError(
            "command must be 'smoke', 'curve', 'baseline', 'baseline-smoke', "
            "'flashinfer-smoke', "
            f"or 'flashinfer-curve', got {command!r}"
        )

    local_bundle = local_bundle_path(bundle_path)
    stage_bundle(esme_bundles, local_bundle, label="esme-curve")
    if command.startswith("flashinfer-"):
        print(f"[esme-curve] flashinfer compare: llm_infer {llm_infer_batches}")
        sweep_res = json.loads(
            sweep_flashinfer_compare.remote(
                llm_infer_batches, max_new_tokens, warmup, iters
            )
        )
        record = {
            "rows": sweep_res["rows"],
            "decode_graphs": sweep_res["decode_graphs"],
            "gpu": {"sweep_container": sweep_res["gpu"], "vllm_container": None},
            "versions": sweep_res["versions"],
            "config": {
                "command": command,
                "model": "Esme-214M-Chat",
                "prompt_pool": "headline",
                "max_new_tokens": max_new_tokens,
                "warmup": warmup,
                "iters": iters,
                "block_size": BLOCK_SIZE,
                "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant",
                "same_container": {
                    "llm_infer_and_flashinfer": True,
                    "hf": False,
                    "vllm": False,
                },
                "repro_command": (
                    f"modal run scripts/modal_esme_batch_curve.py --command {command}"
                ),
            },
        }
        out_dir = REPO_ROOT / "bench-results"
        out_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        out_path = out_dir / f"esme-batch-curve-{command}-{stamp}.json"
        out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"[esme-curve] wrote {out_path}")
        return

    if command in ("baseline", "baseline-smoke"):
        print(
            f"[baseline] llm_infer matrix: batches {llm_infer_batches}, "
            f"contexts {context_lengths}"
        )
        sweep_res = json.loads(
            sweep_baseline.remote(
                llm_infer_batches,
                context_lengths,
                max_new_tokens,
                warmup,
                iters,
            )
        )
        record = {
            "rows": sweep_res["rows"],
            "single_request_prompt_coverage": sweep_res["single_request_prompt_coverage"],
            "decode_graphs": sweep_res["decode_graphs"],
            "gpu": sweep_res["gpu"],
            "versions": sweep_res["versions"],
            "config": {
                "command": command,
                "model": "Esme-214M-Chat",
                "batch_sizes": llm_infer_batches,
                "context_lengths": context_lengths,
                "max_new_tokens": max_new_tokens,
                "warmup": warmup,
                "iters": iters,
                "block_size": BLOCK_SIZE,
                "attention_backend": sweep_res.get("attention_backend"),
                "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant",
                "ignore_eos": True,
                "timing": {
                    "engine_kv_startup": "one synchronized engine construction per matrix row",
                    "steady_state": (
                        f"persistent engine, {warmup} warmups then {iters} measured runs"
                    ),
                    "phase_profile": "one diagnostic run outside steady-state timing; nested spans",
                },
                "same_container": {"llm_infer_matrix": True},
                "repro_command": (
                    f"modal run scripts/modal_esme_batch_curve.py --command {command}"
                ),
            },
        }
        out_dir = REPO_ROOT / "bench-results"
        out_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        suffix = "-smoke" if command == "baseline-smoke" else ""
        out_path = out_dir / f"esme-measurement-baseline{suffix}-{stamp}.json"
        out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"[baseline] wrote {out_path}")
        return

    print("[esme-curve] converting bundle -> HF Qwen3 checkpoint (remote) ...")
    print(f"[esme-curve] {convert_bundle_to_hf.remote()}")

    if skip_vllm:
        vllm_res = {"rows": [], "config": {"skipped": True}, "gpu": None}
        print("[esme-curve] vLLM rows skipped (--skip-vllm)")
    else:
        print(f"[esme-curve] vLLM rows (own container): batches {vllm_batches}")
        vllm_res = json.loads(
            bench_vllm_batches.remote(vllm_batches, max_new_tokens, warmup, iters)
        )
    print(f"[esme-curve] one-container sweep: llm_infer {llm_infer_batches}, HF {hf_specs}")
    sweep_res = json.loads(
        sweep.remote(llm_infer_batches, hf_specs, max_new_tokens, warmup, iters, vllm_res["rows"])
    )

    record = {
        "rows": sweep_res["rows"],
        "kv_evidence": sweep_res["kv_evidence"],
        "decode_graphs": sweep_res["decode_graphs"],
        "gpu": {"sweep_container": sweep_res["gpu"], "vllm_container": vllm_res["gpu"]},
        "versions": sweep_res["versions"],
        "config": {
            "command": command,
            "model": "Esme-214M-Chat",
            "prompt_pool": "headline",
            "max_new_tokens": max_new_tokens,
            "warmup": warmup,
            "iters": iters,
            "block_size": BLOCK_SIZE,
            "attention_backend": sweep_res.get("attention_backend"),
            "vllm": vllm_res["config"],
            "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant",
            "same_container": {
                "llm_infer_and_hf": True,
                "vllm": False,
            },
            "repro_command": f"modal run scripts/modal_esme_batch_curve.py --command {command}",
        },
    }
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-batch-curve-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-curve] wrote {out_path}")
