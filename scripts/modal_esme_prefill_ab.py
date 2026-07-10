"""Same-GPU A/B benchmark for serial versus packed batched Esme prefill.

Both paths use one loaded bf16 runtime in one A100 container. Engine/KV-pool construction
is outside the timed region; the first synchronized ``step()`` measures TTFT, CUDA events
measure model prefill, and the complete request drain measures end-to-end latency. Paired
iterations alternate which path runs first to reduce clock and temperature bias.

The benchmark records timing even when a path diverges. Greedy outputs are classified
against the fp32 full-recompute oracle as exact, tie-tolerant, or diverged; there is no
fallback that hides the candidate's real cost.

    modal run scripts/modal_esme_prefill_ab.py --command prefill-smoke
    modal run scripts/modal_esme_prefill_ab.py --command prefill-ab
"""

from __future__ import annotations

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
from scripts.modal_flash_image import FLASH_IMAGE, REPO_ROOT

BLOCK_SIZE = 128
RAGGED_LENGTHS = (16, 37, 79, 128, 257, 389, 512)
CAPTURE_SIZES = (1, 8, 64)

app = modal.App("llm-infer-esme-prefill-ab")
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=3 * 60 * 60,
)
def benchmark_prefill_ab(
    batch_sizes: list[int],
    uniform_lengths: list[int],
    output_lengths: list[int],
    include_ragged: bool,
    warmup: int,
    iters: int,
) -> str:
    """Run paired serial/packed-prefill measurements on one loaded model and GPU."""
    import math
    import statistics

    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_paged import (
        HEADLINE_PROMPTS,
        EsmeBenchRequest,
        build_requests,
        reference_outputs,
        requests_at_context_length,
    )
    from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.profiling import TimingProfiler
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    if not batch_sizes or any(size < 1 for size in batch_sizes):
        raise ValueError(f"batch_sizes must contain positive integers; got {batch_sizes}")
    if not output_lengths or any(length < 1 for length in output_lengths):
        raise ValueError(f"output_lengths must contain positive integers; got {output_lengths}")
    if any(length < 1 for length in uniform_lengths):
        raise ValueError(f"uniform_lengths must contain positive integers; got {uniform_lengths}")
    if not uniform_lengths and not include_ragged:
        raise ValueError("at least one uniform or ragged workload is required")
    if warmup < 0 or iters < 1:
        raise ValueError(f"warmup must be >= 0 and iters >= 1; got {warmup=}, {iters=}")

    esme_bundles.reload()
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    engine_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    eos = frozenset()
    graph_capture_s = enable_decode_graphs_if_cuda(engine_runtime.model, CAPTURE_SIZES)

    def uniform_requests(batch_size: int, context_length: int) -> list[EsmeBenchRequest]:
        base = build_requests(engine_runtime.tokenizer, batch_size, HEADLINE_PROMPTS)
        return requests_at_context_length(base, context_length)

    def ragged_requests(batch_size: int) -> list[EsmeBenchRequest]:
        base = build_requests(engine_runtime.tokenizer, batch_size, HEADLINE_PROMPTS)
        requests: list[EsmeBenchRequest] = []
        for index, request in enumerate(base):
            context_length = RAGGED_LENGTHS[index % len(RAGGED_LENGTHS)]
            resized = requests_at_context_length([request], context_length)[0]
            requests.append(
                EsmeBenchRequest(
                    request_id=resized.request_id,
                    prompt=f"{request.prompt} [ragged context: {context_length} tokens]",
                    prompt_ids=resized.prompt_ids,
                )
            )
        return requests

    def agreement_dict(agreement) -> dict[str, object]:  # noqa: ANN001 - remote-only type
        return {
            "status": (
                "exact"
                if agreement.exact == agreement.total
                else "tie_tolerant"
                if agreement.all_ties_or_exact
                else "diverged"
            ),
            "exact": agreement.exact,
            "tie": agreement.tie,
            "nontie": agreement.nontie,
            "total": agreement.total,
            "ties_sample": agreement.ties_sample,
            "divergences_sample": agreement.divergences_sample,
        }

    def run_once(
        requests: list[EsmeBenchRequest], max_new_tokens: int, *, batched_prefill: bool
    ) -> dict[str, object]:
        needed_blocks = sum(
            math.ceil((len(request.prompt_ids) + max_new_tokens) / BLOCK_SIZE)
            for request in requests
        )
        profiler = TimingProfiler("cuda")
        engine = InferenceEngine(
            engine_runtime.model,
            block_size=BLOCK_SIZE,
            num_blocks=needed_blocks + max(4, len(requests)),
            device="cuda",
            capabilities=engine_runtime.capabilities,
            profiler=profiler,
            batched_prefill=batched_prefill,
        )
        for request in requests:
            engine.add_request(
                Request(
                    request.request_id,
                    list(request.prompt_ids),
                    max_new_tokens,
                    eos,
                )
            )

        # Engine construction and request registration are deliberately outside the timing.
        torch.cuda.synchronize()
        start = time.perf_counter()
        first_result = engine.step()
        torch.cuda.synchronize()
        first_tokens = {
            request_id: [int(token) for token in tokens]
            for request_id, tokens in first_result.tokens.items()
        }
        ttft_s = time.perf_counter() - start

        outputs = dict(first_result.finished_outputs)
        while engine.scheduler.has_work():
            result = engine.step()
            outputs.update(result.finished_outputs)
        torch.cuda.synchronize()
        end_to_end_s = time.perf_counter() - start

        phase_profile = profiler.summary().as_dict()
        phases = phase_profile["phases"]
        prefill_ms = phases["prefill"]["total_ms"]
        sampling_ms = phases["sampling"]["total_ms"]
        return {
            "prefill_device_seconds": float(prefill_ms) / 1000.0,
            "ttft_wall_seconds": ttft_s,
            "end_to_end_wall_seconds": end_to_end_s,
            "sampling_device_seconds": float(sampling_ms) / 1000.0,
            "first_tokens": first_tokens,
            "outputs": outputs,
        }

    workload_specs: list[tuple[str, int | None]] = [
        ("uniform", context_length) for context_length in uniform_lengths
    ]
    if include_ragged:
        workload_specs.append(("ragged", None))

    rows: list[dict[str, object]] = []
    for batch_size in batch_sizes:
        for shape, context_length in workload_specs:
            requests = (
                uniform_requests(batch_size, context_length)
                if context_length is not None
                else ragged_requests(batch_size)
            )
            prompt_lengths = [len(request.prompt_ids) for request in requests]
            for max_new_tokens in output_lengths:
                modes = {"baseline": False, "candidate": True}

                for pair_index in range(warmup):
                    order = (
                        ("baseline", "candidate")
                        if pair_index % 2 == 0
                        else ("candidate", "baseline")
                    )
                    for mode in order:
                        run_once(
                            requests,
                            max_new_tokens,
                            batched_prefill=modes[mode],
                        )

                raw: dict[str, list[dict[str, object]]] = {"baseline": [], "candidate": []}
                outputs_by_mode: dict[str, dict[str, list[int]]] = {}
                first_tokens_by_mode: dict[str, dict[str, list[int]]] = {}
                pair_orders: list[list[str]] = []
                for pair_index in range(iters):
                    order = (
                        ("baseline", "candidate")
                        if pair_index % 2 == 0
                        else ("candidate", "baseline")
                    )
                    pair_orders.append(list(order))
                    for order_position, mode in enumerate(order):
                        measured = run_once(
                            requests,
                            max_new_tokens,
                            batched_prefill=modes[mode],
                        )
                        outputs_by_mode[mode] = measured.pop("outputs")
                        first_tokens_by_mode[mode] = measured.pop("first_tokens")
                        raw[mode].append(
                            {
                                "iteration": pair_index,
                                "order_position": order_position,
                                **measured,
                            }
                        )

                medians: dict[str, dict[str, float]] = {}
                for mode, records in raw.items():
                    medians[mode] = {
                        metric: statistics.median(float(record[metric]) for record in records)
                        for metric in (
                            "prefill_device_seconds",
                            "ttft_wall_seconds",
                            "end_to_end_wall_seconds",
                        )
                    }

                relative_deltas = {
                    metric: medians["candidate"][metric] / medians["baseline"][metric] - 1.0
                    for metric in medians["baseline"]
                }
                speedups = {
                    metric: medians["baseline"][metric] / medians["candidate"][metric]
                    for metric in medians["baseline"]
                }

                reference = reference_outputs(
                    oracle_runtime.model,
                    requests,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos,
                )
                agreement = {
                    mode: agreement_dict(
                        tie_tolerant_agreement(
                            oracle_runtime.model,
                            requests,
                            outputs_by_mode[mode],
                            reference,
                            eos,
                        )
                    )
                    for mode in modes
                }
                row = {
                    "batch_size": batch_size,
                    "shape": shape,
                    "context_length": context_length,
                    "prompt_lengths": prompt_lengths,
                    "max_new_tokens": max_new_tokens,
                    "pair_orders": pair_orders,
                    "raw_iterations": raw,
                    "medians": medians,
                    "candidate_relative_delta": relative_deltas,
                    "candidate_speedup": speedups,
                    "candidate_matches_baseline_exact": (
                        outputs_by_mode["candidate"] == outputs_by_mode["baseline"]
                    ),
                    "agreement": agreement,
                    "first_tokens": first_tokens_by_mode,
                    "outputs": outputs_by_mode,
                    "reference_outputs": reference,
                }
                rows.append(row)
                print(
                    f"[prefill-ab] b={batch_size} shape={shape} "
                    f"context={context_length or 'ragged'} out={max_new_tokens}: "
                    f"prefill {speedups['prefill_device_seconds']:.2f}x, "
                    f"TTFT {speedups['ttft_wall_seconds']:.2f}x, "
                    f"E2E {speedups['end_to_end_wall_seconds']:.2f}x, "
                    f"candidate={agreement['candidate']['status']}"
                )

    return json.dumps(
        {
            "rows": rows,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
            "attention_backend": type(engine_runtime.model.backend).__name__,
            "decode_graphs": {"capture_sizes": list(CAPTURE_SIZES), "capture_s": graph_capture_s},
        }
    )


@app.local_entrypoint()
def main(command: str = "prefill-ab", bundle_path: str = "") -> None:
    """Stage Esme, run the requested A/B protocol, and write its raw JSON record."""
    if command == "prefill-smoke":
        batch_sizes = [8]
        uniform_lengths: list[int] = []
        output_lengths = [4]
        include_ragged = True
        warmup, iters = 1, 2
    elif command == "prefill-ab":
        batch_sizes = [1, 8, 64]
        uniform_lengths = [16, 128, 512]
        output_lengths = [1, 64]
        include_ragged = True
        warmup, iters = 2, 10
    else:
        raise ValueError(
            f"command must be 'prefill-smoke' or 'prefill-ab', got {command!r}"
        )

    local_bundle = local_bundle_path(bundle_path)
    stage_bundle(esme_bundles, local_bundle, label="esme-prefill-ab")
    result = json.loads(
        benchmark_prefill_ab.remote(
            batch_sizes,
            uniform_lengths,
            output_lengths,
            include_ragged,
            warmup,
            iters,
        )
    )
    record = {
        **result,
        "config": {
            "command": command,
            "model": "Esme-214M-Chat",
            "dtype": "bfloat16",
            "batch_sizes": batch_sizes,
            "uniform_context_lengths": uniform_lengths,
            "ragged_context_cycle": list(RAGGED_LENGTHS) if include_ragged else None,
            "max_new_tokens": output_lengths,
            "warmup_pairs": warmup,
            "measured_pairs": iters,
            "block_size": BLOCK_SIZE,
            "baseline": "InferenceEngine(batched_prefill=False)",
            "candidate": "InferenceEngine(batched_prefill=True)",
            "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant",
            "ignore_eos": True,
            "timing": {
                "engine_startup": "excluded",
                "prefill": "CUDA-event time for the model prefill span",
                "ttft": "synchronized first engine step, including planning and sampling",
                "end_to_end": "first step through complete request drain",
                "pairing": "baseline/candidate order alternates each pair",
            },
            "same_container_and_gpu": True,
            "repro_command": (
                f"modal run scripts/modal_esme_prefill_ab.py --command {command}"
            ),
        },
    }
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-prefill-ab] wrote {out_path}")
