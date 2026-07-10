"""Same-GPU A/B benchmark for serial versus packed batched Esme prefill.

Both paths use one loaded bf16 runtime in one A100 container. Engine/KV-pool construction
is outside the timed region; the first synchronized ``step()`` measures TTFT, CUDA events
measure model prefill, and the complete request drain measures end-to-end latency. Paired
iterations alternate which path runs first to reduce clock and temperature bias.

The benchmark records timing even when a path diverges. Greedy outputs are classified
against the fp32 full-recompute oracle as exact, tie-tolerant, or diverged; there is no
fallback that hides the candidate's real cost.

Each completed row is streamed to ``bench-results/esme-<command>-rows.jsonl`` as it finishes,
so an interrupted run keeps every row it already measured. Re-run with ``--resume`` to skip the
completed cells and append the rest to the same log.

    modal run scripts/modal_esme_prefill_ab.py --command prefill-smoke
    modal run scripts/modal_esme_prefill_ab.py --command prefill-ab
    modal run scripts/modal_esme_prefill_ab.py --command prefill-ab --resume

The ``prefill-divergence`` command reruns a single offending cell (batch 8, ragged, 64 output
tokens) outside the timed path with extra recording — per-attempt first-divergence step/tokens,
serial-vs-packed KV closeness per layer, and bf16 vs fp32-oracle logit margins — so a row the
0.1-logit rule labeled "diverged" can be characterized before batched prefill is enabled by
default. It writes its own ``bench-results/esme-prefill-divergence-<timestamp>.json`` and does
not touch the rows-log / resume machinery.

    modal run scripts/modal_esme_prefill_ab.py --command prefill-divergence
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Iterator
from dataclasses import dataclass
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

# Mixed-load burst cells: growing ragged bursts plus one worst-case uniform-512 burst
# (~32k prompt tokens packed into a single prefill call).
MIXED_LOAD_CELLS: list[dict[str, object]] = [
    {"burst_size": 8, "burst_shape": "ragged", "burst_context": None},
    {"burst_size": 32, "burst_shape": "ragged", "burst_context": None},
    {"burst_size": 64, "burst_shape": "ragged", "burst_context": None},
    {"burst_size": 64, "burst_shape": "uniform", "burst_context": 512},
]

app = modal.App("llm-infer-esme-prefill-ab")
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# A workload cell's identity fields, per command; --resume keys on the matching tuple.
PREFILL_AB_KEY_FIELDS = ("batch_size", "shape", "context_length", "max_new_tokens")
MIXED_LOAD_KEY_FIELDS = ("burst_size", "burst_shape")


def row_key(row: dict[str, object], key_fields: tuple[str, ...]) -> dict[str, object]:
    """Return the JSON-serializable identity of a workload cell (row or row event)."""
    return {field: row[field] for field in key_fields}


def key_tuple(key: dict[str, object], key_fields: tuple[str, ...]) -> tuple[object, ...]:
    """Hashable form of a row key for set membership."""
    return tuple(key[field] for field in key_fields)


def parse_event_lines(text: str) -> list[dict[str, object]]:
    """Parse a rows JSONL log into event dicts, rejecting malformed lines by line number."""
    events: list[dict[str, object]] = []
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"malformed JSON in rows log at line {line_number}: {error}"
            ) from error
        if not isinstance(event, dict) or "kind" not in event:
            raise ValueError(f"rows log line {line_number} is not an event object with a 'kind'")
        events.append(event)
    return events


def completed_row_keys(
    events: list[dict[str, object]], key_fields: tuple[str, ...]
) -> list[dict[str, object]]:
    """Collect the keys of every completed row event, in log order."""
    return [row_key(event, key_fields) for event in events if event.get("kind") == "row"]


def assemble_final_record(
    events: list[dict[str, object]], config: dict[str, object]
) -> dict[str, object]:
    """Rebuild the combined pretty-JSON record from a full rows log (old plus new)."""
    rows = [
        {key: value for key, value in event.items() if key != "kind"}
        for event in events
        if event.get("kind") == "row"
    ]
    metas = [event for event in events if event.get("kind") == "meta"]
    if not metas:
        raise ValueError("rows log has no meta event to source GPU and version metadata from")
    # Newest meta wins: a resumed run appends a fresh meta reflecting the container it ran in.
    meta = metas[-1]
    return {
        "rows": rows,
        "gpu": meta["gpu"],
        "versions": meta["versions"],
        "attention_backend": meta["attention_backend"],
        "decode_graphs": meta["decode_graphs"],
        "config": config,
    }


def fingerprint_outputs(
    outputs: dict[str, list[int]], reference: dict[str, list[int]]
) -> dict[str, dict[str, int | None]]:
    """First-divergence fingerprint of one run's tokens vs the fp32 reference, per request.

    Pure CPU list diff, O(tokens). For each request in ``reference``: the first index where the
    run's tokens differ from the reference (``None`` when identical over the overlap), plus the
    two token ids at that index. This lets a whole A/B row carry a per-iteration divergence
    signal instead of only the last measured iteration's surviving outputs.
    """
    fingerprint: dict[str, dict[str, int | None]] = {}
    for request_id, golden in reference.items():
        fast = outputs.get(request_id, [])
        step = next(
            (i for i, (f, g) in enumerate(zip(fast, golden, strict=False)) if f != g),
            None,
        )
        if step is None:
            fingerprint[request_id] = {
                "first_diff_step": None,
                "fast_token": None,
                "golden_token": None,
            }
        else:
            fingerprint[request_id] = {
                "first_diff_step": step,
                "fast_token": fast[step],
                "golden_token": golden[step],
            }
    return fingerprint


def stability_summary(
    records: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    """Per request and mode, whether the first-divergence step and tokens repeat across attempts.

    Groups the flat divergence records by ``(request_id, mode)`` and reports the distinct
    first-divergence steps and token pairs seen across attempts. ``stable`` is true when every
    attempt agreed (one distinct step and one distinct token pair) — the run-to-run question the
    diagnostic exists to answer.
    """

    def sort_key(value: object) -> tuple[bool, object]:
        return (value is None, value if value is not None else 0)

    grouped: dict[str, dict[str, list[dict[str, object]]]] = {}
    for record in records:
        request_id = str(record["request_id"])
        mode = str(record["mode"])
        grouped.setdefault(request_id, {}).setdefault(mode, []).append(record)

    summary: dict[str, dict[str, dict[str, object]]] = {}
    for request_id, by_mode in grouped.items():
        summary[request_id] = {}
        for mode, mode_records in by_mode.items():
            steps = sorted({record["first_diff_step"] for record in mode_records}, key=sort_key)
            pairs = sorted(
                {
                    (record["first_diff_tokens"]["fast"], record["first_diff_tokens"]["golden"])
                    for record in mode_records
                },
                key=lambda pair: (sort_key(pair[0]), sort_key(pair[1])),
            )
            summary[request_id][mode] = {
                "stable": len(steps) == 1 and len(pairs) == 1,
                "attempts": len(mode_records),
                "first_diff_steps": list(steps),
                "first_diff_tokens": [{"fast": fast, "golden": golden} for fast, golden in pairs],
            }
    return summary


def agreement_dict(agreement) -> dict[str, object]:  # noqa: ANN001 - EsmeAgreement is remote-only
    """Flatten an ``EsmeAgreement`` into the JSON status/counts a row records."""
    from llm_infer.benchmarks.reference_policy import reference_status

    return {
        "status": reference_status(agreement),
        "exact": agreement.exact,
        "tie": agreement.tie,
        "nontie": agreement.nontie,
        "total": agreement.total,
        "ties_sample": agreement.ties_sample,
        "divergences_sample": agreement.divergences_sample,
        "review_required": agreement.review_required,
        "failed": agreement.failed,
        "numerical_evidence": agreement.numerical_evidence,
    }


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated q-th percentile (q in 0..100) of a non-empty sample."""
    if not values:
        raise ValueError("percentile of an empty sample is undefined")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"percentile q must be in [0, 100]; got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass(frozen=True)
class ItlSplit:
    """One steady decoder's inter-token gaps (ms) split around the burst-admission step.

    ``before_ms`` are the steady-state gaps before the burst, ``spanning_ms`` the single gap
    that straddles the burst step (the decode-tail stall the burst injects, ``None`` if the
    decoder produced no token across the burst boundary), ``after_ms`` the recovery gaps.
    """

    before_ms: list[float]
    spanning_ms: float | None
    after_ms: list[float]

    def as_dict(self) -> dict[str, object]:
        return {
            "before_ms": self.before_ms,
            "spanning_ms": self.spanning_ms,
            "after_ms": self.after_ms,
        }


def split_itls_by_burst_step(
    token_steps: list[int], token_times_s: list[float], burst_step: int
) -> ItlSplit:
    """Split one request's inter-token latencies around the burst-admission step.

    ``token_steps[k]`` is the engine step index that produced this request's k-th recorded
    token and ``token_times_s[k]`` the synchronized wall-clock at that step boundary. Each
    inter-token gap is attributed by the steps its two endpoints fall in: a gap whose later
    token landed before ``burst_step`` is steady-state, the single gap that straddles
    ``burst_step`` (earlier token before it, later token at or after it) is the burst stall,
    and gaps whose earlier token already sits at or past ``burst_step`` are recovery.
    """
    if len(token_steps) != len(token_times_s):
        raise ValueError(
            f"token_steps and token_times_s differ in length: "
            f"{len(token_steps)} vs {len(token_times_s)}"
        )
    before: list[float] = []
    after: list[float] = []
    spanning: float | None = None
    for k in range(1, len(token_times_s)):
        gap_ms = (token_times_s[k] - token_times_s[k - 1]) * 1000.0
        if token_steps[k] < burst_step:
            before.append(gap_ms)
        elif token_steps[k - 1] >= burst_step:
            after.append(gap_ms)
        else:
            # Earlier token before the burst, later token at or past it: the straddling gap.
            spanning = gap_ms
    return ItlSplit(before_ms=before, spanning_ms=spanning, after_ms=after)


def summarize_run(splits: list[ItlSplit], burst_ttfts_ms: list[float]) -> dict[str, float]:
    """Reduce one run's per-decoder ITL splits and burst TTFTs to the cell's decision scalars.

    ``itl_spanning_burst_ms`` is the median stall across steady decoders, ``itl_before_p50_ms``
    the pooled steady-state baseline, and ``stall_ratio`` their quotient — how many normal
    inter-token gaps the burst prefill costs a decoder mid-stream.
    """
    spanning_values = [split.spanning_ms for split in splits if split.spanning_ms is not None]
    before_pool = [gap for split in splits for gap in split.before_ms]
    after_pool = [gap for split in splits for gap in split.after_ms]
    if not spanning_values:
        raise ValueError("no steady decoder produced a token across the burst boundary")
    if not before_pool:
        raise ValueError("no pre-burst inter-token gaps to form a steady-state baseline")
    if not burst_ttfts_ms:
        raise ValueError("no burst requests produced a first token")
    itl_before_p50 = percentile(before_pool, 50)
    itl_spanning = statistics.median(spanning_values)
    summary = {
        "itl_spanning_burst_ms": itl_spanning,
        "itl_before_p50_ms": itl_before_p50,
        "itl_before_p95_ms": percentile(before_pool, 95),
        "itl_before_p99_ms": percentile(before_pool, 99),
        "stall_ratio": itl_spanning / itl_before_p50,
        "burst_ttft_p50_ms": percentile(burst_ttfts_ms, 50),
        "burst_ttft_p95_ms": percentile(burst_ttfts_ms, 95),
    }
    # After-burst recovery only exists once steady decoders keep running past the burst; the
    # first-pass matrix always has it, but stay loud rather than silently fabricate a value.
    if after_pool:
        summary["itl_after_p50_ms"] = percentile(after_pool, 50)
        summary["itl_after_p95_ms"] = percentile(after_pool, 95)
    return summary


# Per-run scalars carried through the cell aggregation; every measured run reports each one.
_MIXED_LOAD_RUN_METRICS = (
    "itl_spanning_burst_ms",
    "itl_before_p50_ms",
    "itl_before_p95_ms",
    "itl_before_p99_ms",
    "stall_ratio",
    "burst_ttft_p50_ms",
    "burst_ttft_p95_ms",
)


def aggregate_mixed_load(
    runs_by_mode: dict[str, list[dict[str, float]]],
) -> dict[str, object]:
    """Median each mode's per-run scalars across pairs, then compare candidate to baseline.

    ``candidate_vs_baseline`` reports the two decision ratios the roadmap turns on: whether
    batched prefill lengthens or shortens the burst-spanning decode stall, and its effect on
    burst TTFT. Both are candidate-over-baseline, so above 1.0 means batched prefill is worse.
    """
    if set(runs_by_mode) != {"baseline", "candidate"}:
        raise ValueError(f"expected baseline and candidate runs; got {sorted(runs_by_mode)}")
    medians: dict[str, dict[str, float]] = {}
    for mode, runs in runs_by_mode.items():
        if not runs:
            raise ValueError(f"mode {mode!r} has no measured runs to aggregate")
        medians[mode] = {
            metric: statistics.median(run[metric] for run in runs)
            for metric in _MIXED_LOAD_RUN_METRICS
        }
    comparison = {
        "spanning_itl_ratio": (
            medians["candidate"]["itl_spanning_burst_ms"]
            / medians["baseline"]["itl_spanning_burst_ms"]
        ),
        "burst_ttft_p50_ratio": (
            medians["candidate"]["burst_ttft_p50_ms"] / medians["baseline"]["burst_ttft_p50_ms"]
        ),
    }
    return {"medians": medians, "candidate_vs_baseline": comparison}


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
    completed: list[dict[str, object]],
) -> Iterator[dict[str, object]]:
    """Stream paired serial/packed-prefill measurements from one loaded model and GPU.

    Yields a single ``meta`` event, then one ``row`` event per completed workload cell.
    Cells whose key is in ``completed`` are skipped so a resumed run only measures the rest.
    """
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
    completed_set = {key_tuple(key, PREFILL_AB_KEY_FIELDS) for key in completed}

    yield {
        "kind": "meta",
        "gpu": gpu_snapshot(),
        "versions": library_versions(),
        "attention_backend": type(engine_runtime.model.backend).__name__,
        "decode_graphs": {"capture_sizes": list(CAPTURE_SIZES), "capture_s": graph_capture_s},
    }

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

    for batch_size in batch_sizes:
        for shape, context_length in workload_specs:
            requests = (
                uniform_requests(batch_size, context_length)
                if context_length is not None
                else ragged_requests(batch_size)
            )
            prompt_lengths = [len(request.prompt_ids) for request in requests]
            for max_new_tokens in output_lengths:
                current_key = {
                    "batch_size": batch_size,
                    "shape": shape,
                    "context_length": context_length,
                    "max_new_tokens": max_new_tokens,
                }
                if key_tuple(current_key, PREFILL_AB_KEY_FIELDS) in completed_set:
                    print(
                        f"[prefill-ab] skip b={batch_size} shape={shape} "
                        f"context={context_length or 'ragged'} out={max_new_tokens}"
                    )
                    continue
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
                # Every measured iteration's outputs, not just the last surviving one, so the row
                # can fingerprint run-to-run divergence stability. Each run_once returns a fresh
                # outputs dict, so appending the reference is safe.
                outputs_per_iteration: dict[str, list[dict[str, list[int]]]] = {
                    "baseline": [],
                    "candidate": [],
                }
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
                        outputs_per_iteration[mode].append(outputs_by_mode[mode])
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
                # Fingerprint every measured iteration (CPU list diff, outside the timed spans),
                # so the row shows whether the divergence step/token is stable across the paired
                # runs rather than only reporting the last iteration's surviving outputs.
                divergence_fingerprints = {
                    mode: [
                        fingerprint_outputs(iteration_outputs, reference)
                        for iteration_outputs in outputs_per_iteration[mode]
                    ]
                    for mode in modes
                }
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
                direct_parity = outputs_by_mode["candidate"] == outputs_by_mode["baseline"]
                row = {
                    "policy_version": 2,
                    "batch_size": batch_size,
                    "shape": shape,
                    "context_length": context_length,
                    "prompt_lengths": prompt_lengths,
                    "max_new_tokens": max_new_tokens,
                    "pair_orders": pair_orders,
                    "raw_iterations": raw,
                    "medians": medians,
                    "raw_candidate_relative_delta": relative_deltas,
                    "raw_candidate_speedup": speedups,
                    "candidate_relative_delta": relative_deltas if direct_parity else None,
                    "candidate_speedup": speedups if direct_parity else None,
                    "candidate_matches_baseline_exact": direct_parity,
                    "parity_status": "exact" if direct_parity else "review_required",
                    "relative_claim_eligible": direct_parity,
                    "agreement": agreement,
                    "first_tokens": first_tokens_by_mode,
                    "outputs": outputs_by_mode,
                    "reference_outputs": reference,
                    "divergence_fingerprints": divergence_fingerprints,
                }
                print(
                    f"[prefill-ab] b={batch_size} shape={shape} "
                    f"context={context_length or 'ragged'} out={max_new_tokens}: "
                    f"prefill {speedups['prefill_device_seconds']:.2f}x, "
                    f"TTFT {speedups['ttft_wall_seconds']:.2f}x, "
                    f"E2E {speedups['end_to_end_wall_seconds']:.2f}x, "
                    f"candidate={agreement['candidate']['status']}, "
                    f"relative={'eligible' if direct_parity else 'withheld; raw timing only'}"
                )
                yield {"kind": "row", **row}


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=3 * 60 * 60,
)
def benchmark_mixed_load(
    cells: list[dict[str, object]],
    steady_decoders: int,
    steady_max_new_tokens: int,
    burst_max_new_tokens: int,
    warmup_tokens: int,
    warmup: int,
    iters: int,
    completed: list[dict[str, object]],
) -> Iterator[dict[str, object]]:
    """Stream serial/packed-prefill decode-stall measurements under a mid-stream prompt burst.

    N steady ragged decoders are stepped until each has produced ``warmup_tokens`` tokens; a
    burst of new prompts is then admitted in one step (reserve mode, no preemption — the whole
    burst prefills together, the worst case), and stepping continues until every request drains.
    Each ``engine.step()`` is bracketed by ``torch.cuda.synchronize()`` so per-step wall time is
    real host time, and ``decode_window_size=1`` keeps one decode token per step so a steady
    decoder's inter-token latency is exactly the wall time of the step that produced its token.
    The gap that straddles the burst step is the decode-tail stall the roadmap asks about.

    Yields one ``meta`` event, then one ``row`` event per workload cell not already in
    ``completed``. Timing is reported for every mode regardless of the fp32-oracle verdict.
    """
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
    if not cells:
        raise ValueError("mixed-load needs at least one workload cell")
    if steady_decoders < 1:
        raise ValueError(f"steady_decoders must be >= 1; got {steady_decoders}")
    if steady_max_new_tokens < 1 or burst_max_new_tokens < 1:
        raise ValueError(
            f"max_new_tokens must be positive; got steady={steady_max_new_tokens}, "
            f"burst={burst_max_new_tokens}"
        )
    if warmup_tokens < 2:
        raise ValueError(
            f"warmup_tokens must be >= 2 to form a pre-burst baseline; got {warmup_tokens}"
        )
    if warmup_tokens >= steady_max_new_tokens:
        raise ValueError(
            f"warmup_tokens ({warmup_tokens}) must leave the steady decoders mid-stream "
            f"before steady_max_new_tokens ({steady_max_new_tokens})"
        )
    if warmup < 0 or iters < 1:
        raise ValueError(f"warmup must be >= 0 and iters >= 1; got {warmup=}, {iters=}")
    for cell in cells:
        missing = [
            field for field in ("burst_size", "burst_shape", "burst_context") if field not in cell
        ]
        if missing:
            raise ValueError(f"cell {cell} is missing required fields {missing}")
        if cell["burst_shape"] not in ("ragged", "uniform"):
            raise ValueError(
                f"burst_shape must be 'ragged' or 'uniform'; got {cell['burst_shape']!r}"
            )
        if cell["burst_shape"] == "uniform" and cell["burst_context"] is None:
            raise ValueError(f"uniform burst cell needs a burst_context; got {cell}")

    esme_bundles.reload()
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    engine_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    eos = frozenset()
    graph_capture_s = enable_decode_graphs_if_cuda(engine_runtime.model, CAPTURE_SIZES)
    completed_set = {key_tuple(key, MIXED_LOAD_KEY_FIELDS) for key in completed}

    yield {
        "kind": "meta",
        "gpu": gpu_snapshot(),
        "versions": library_versions(),
        "attention_backend": type(engine_runtime.model.backend).__name__,
        "decode_graphs": {"capture_sizes": list(CAPTURE_SIZES), "capture_s": graph_capture_s},
    }

    def build_pool(
        count: int, shape: str, id_prefix: str, uniform_context: int | None
    ) -> list[EsmeBenchRequest]:
        """Build a distinct-id request pool: ragged cycles RAGGED_LENGTHS, uniform is one length."""
        base = build_requests(engine_runtime.tokenizer, count, HEADLINE_PROMPTS)
        pool: list[EsmeBenchRequest] = []
        for index, request in enumerate(base):
            context_length = (
                uniform_context
                if shape == "uniform"
                else RAGGED_LENGTHS[index % len(RAGGED_LENGTHS)]
            )
            resized = requests_at_context_length([request], context_length)[0]
            pool.append(
                EsmeBenchRequest(
                    request_id=f"{id_prefix}-{index:03d}",
                    prompt=f"{request.prompt} [{shape} context: {context_length} tokens]",
                    prompt_ids=resized.prompt_ids,
                )
            )
        return pool

    def blocks_needed(prompt_len: int, max_new: int) -> int:
        return math.ceil((prompt_len + max_new) / BLOCK_SIZE)

    def run_mixed_once(
        steady: list[EsmeBenchRequest],
        burst: list[EsmeBenchRequest],
        *,
        batched_prefill: bool,
    ) -> dict[str, object]:
        steady_ids = [request.request_id for request in steady]
        burst_ids = [request.request_id for request in burst]
        reserve = sum(
            blocks_needed(len(request.prompt_ids), steady_max_new_tokens) for request in steady
        ) + sum(blocks_needed(len(request.prompt_ids), burst_max_new_tokens) for request in burst)
        # Generous pool: the whole reserve fits with headroom, so reserve-mode admission pulls
        # the entire burst into the running set in one step — the single packed prefill we want
        # to attribute the decode-tail stall to.
        num_blocks = reserve + max(8, len(steady) + len(burst))
        profiler = TimingProfiler("cuda")
        engine = InferenceEngine(
            engine_runtime.model,
            block_size=BLOCK_SIZE,
            num_blocks=num_blocks,
            device="cuda",
            capabilities=engine_runtime.capabilities,
            profiler=profiler,
            batched_prefill=batched_prefill,
            # One decode token per step so each steady decoder's ITL is a single step's wall
            # time; the deferred window would batch several tokens behind one host sync.
            decode_window_size=1,
        )
        for request in steady:
            engine.add_request(
                Request(request.request_id, list(request.prompt_ids), steady_max_new_tokens, eos)
            )

        token_events: dict[str, list[tuple[int, float]]] = {
            request_id: [] for request_id in (*steady_ids, *burst_ids)
        }
        step_records: list[dict[str, object]] = []
        outputs: dict[str, list[int]] = {}
        burst_added = False
        burst_step: int | None = None
        burst_add_wall: float | None = None
        step_index = 0

        torch.cuda.synchronize()
        boundary = time.perf_counter()
        while engine.scheduler.has_work():
            if not burst_added and all(
                len(token_events[request_id]) >= warmup_tokens for request_id in steady_ids
            ):
                # Admit the whole burst just before this step so it prefills in one packed call
                # while the steady decoders are mid-stream. The boundary is not reset here: the
                # burst step's wall time then carries the admission and packed prefill cost.
                torch.cuda.synchronize()
                burst_add_wall = time.perf_counter()
                for request in burst:
                    engine.add_request(
                        Request(
                            request.request_id,
                            list(request.prompt_ids),
                            burst_max_new_tokens,
                            eos,
                        )
                    )
                burst_added = True
                burst_step = step_index

            result = engine.step()
            torch.cuda.synchronize()
            now = time.perf_counter()
            step_wall_ms = (now - boundary) * 1000.0
            boundary = now

            prefilled_tokens = sum(end - start for start, end in result.prefill_chunks.values())
            decode_ids = [rid for rid in result.tokens if rid not in result.prefill_chunks]
            allocator = engine.cache.allocator
            step_records.append(
                {
                    "step_index": step_index,
                    "wall_ms": step_wall_ms,
                    "prefilled_requests": len(result.prefill_chunks),
                    "prefilled_tokens": prefilled_tokens,
                    "decode_tokens": len(decode_ids),
                    "kv_utilization": allocator.num_used / allocator.num_blocks,
                    "running": len(engine.scheduler.running),
                    "waiting": len(engine.scheduler.waiting),
                }
            )
            for request_id, tokens in result.tokens.items():
                for _ in tokens:
                    token_events[request_id].append((step_index, now))
            outputs.update(result.finished_outputs)
            step_index += 1

        if burst_step is None or burst_add_wall is None:
            raise RuntimeError(
                "steady decoders drained before reaching the warmup token count; "
                "raise steady_max_new_tokens or lower warmup_tokens"
            )

        splits = {
            request_id: split_itls_by_burst_step(
                [step for step, _ in token_events[request_id]],
                [when for _, when in token_events[request_id]],
                burst_step,
            )
            for request_id in steady_ids
        }
        burst_ttfts_ms = {
            request_id: (token_events[request_id][0][1] - burst_add_wall) * 1000.0
            for request_id in burst_ids
            if token_events[request_id]
        }
        total_wall_ms = sum(float(record["wall_ms"]) for record in step_records)
        phases = profiler.summary().as_dict()["phases"]
        device_ms = (
            phases["prefill"]["total_ms"]
            + phases["decode"]["total_ms"]
            + phases["sampling"]["total_ms"]
        )
        return {
            "burst_step": burst_step,
            "burst_prompt_tokens_total": sum(len(request.prompt_ids) for request in burst),
            "splits": splits,
            "burst_ttfts_ms": burst_ttfts_ms,
            "step_records": step_records,
            "total_wall_ms": total_wall_ms,
            # Coarse: total host wall minus the top-level device phases (prefill, decode,
            # sampling). It lumps scheduler admission, Python classification, and per-step
            # cuda syncs together — a residual, not an isolated scheduler timer.
            "scheduler_sync_residual_ms": total_wall_ms - device_ms,
            "phase_device_ms": {
                "prefill": phases["prefill"]["total_ms"],
                "decode": phases["decode"]["total_ms"],
                "sampling": phases["sampling"]["total_ms"],
            },
            "outputs": outputs,
        }

    modes = {"baseline": False, "candidate": True}
    steady = build_pool(steady_decoders, "ragged", "steady", None)
    steady_ids = [request.request_id for request in steady]

    for cell in cells:
        burst_size = int(cell["burst_size"])
        burst_shape = str(cell["burst_shape"])
        burst_context = cell["burst_context"]
        current_key = {"burst_size": burst_size, "burst_shape": burst_shape}
        if key_tuple(current_key, MIXED_LOAD_KEY_FIELDS) in completed_set:
            print(f"[mixed-load] skip burst={burst_size} shape={burst_shape}")
            continue

        burst = build_pool(
            burst_size,
            burst_shape,
            "burst",
            None if burst_context is None else int(burst_context),
        )
        # The fp32 oracle reference for steady and burst is prompt-deterministic, so compute it
        # once per cell and gate every mode's outputs against it after the paired runs.
        reference = {
            **reference_outputs(
                oracle_runtime.model,
                steady,
                max_new_tokens=steady_max_new_tokens,
                eos_token_ids=eos,
            ),
            **reference_outputs(
                oracle_runtime.model,
                burst,
                max_new_tokens=burst_max_new_tokens,
                eos_token_ids=eos,
            ),
        }
        all_requests = [*steady, *burst]

        for pair_index in range(warmup):
            order = (
                ("baseline", "candidate")
                if pair_index % 2 == 0
                else ("candidate", "baseline")
            )
            for mode in order:
                run_mixed_once(steady, burst, batched_prefill=modes[mode])

        runs_by_mode: dict[str, list[dict[str, float]]] = {"baseline": [], "candidate": []}
        raw_by_mode: dict[str, list[dict[str, object]]] = {"baseline": [], "candidate": []}
        outputs_by_mode: dict[str, dict[str, list[int]]] = {}
        pair_orders: list[list[str]] = []
        for pair_index in range(iters):
            order = (
                ("baseline", "candidate")
                if pair_index % 2 == 0
                else ("candidate", "baseline")
            )
            pair_orders.append(list(order))
            for order_position, mode in enumerate(order):
                measured = run_mixed_once(steady, burst, batched_prefill=modes[mode])
                outputs_by_mode[mode] = measured["outputs"]
                splits = measured["splits"]
                summary = summarize_run(
                    [splits[request_id] for request_id in steady_ids],
                    list(measured["burst_ttfts_ms"].values()),
                )
                runs_by_mode[mode].append(summary)
                raw_by_mode[mode].append(
                    {
                        "iteration": pair_index,
                        "order_position": order_position,
                        "burst_step": measured["burst_step"],
                        "burst_prompt_tokens_total": measured["burst_prompt_tokens_total"],
                        "run_summary": summary,
                        "itl_splits": {
                            request_id: splits[request_id].as_dict()
                            for request_id in steady_ids
                        },
                        "burst_ttfts_ms": measured["burst_ttfts_ms"],
                        "step_records": measured["step_records"],
                        "total_wall_ms": measured["total_wall_ms"],
                        "scheduler_sync_residual_ms": measured["scheduler_sync_residual_ms"],
                        "phase_device_ms": measured["phase_device_ms"],
                    }
                )

        aggregate = aggregate_mixed_load(runs_by_mode)
        agreement = {
            mode: agreement_dict(
                tie_tolerant_agreement(
                    oracle_runtime.model, all_requests, outputs_by_mode[mode], reference, eos
                )
            )
            for mode in modes
        }
        direct_parity = outputs_by_mode["candidate"] == outputs_by_mode["baseline"]
        public_aggregate = aggregate if direct_parity else {
            "baseline": aggregate["baseline"],
            "candidate": aggregate["candidate"],
            "candidate_vs_baseline": None,
        }
        row = {
            "policy_version": 2,
            "burst_size": burst_size,
            "burst_shape": burst_shape,
            "burst_context": burst_context,
            "steady_decoders": steady_decoders,
            "steady_shape": "ragged",
            "steady_max_new_tokens": steady_max_new_tokens,
            "burst_max_new_tokens": burst_max_new_tokens,
            "warmup_tokens": warmup_tokens,
            "steady_prompt_lengths": [len(request.prompt_ids) for request in steady],
            "burst_prompt_lengths": [len(request.prompt_ids) for request in burst],
            "burst_prompt_tokens_total": sum(len(request.prompt_ids) for request in burst),
            "pair_orders": pair_orders,
            "raw_iterations": raw_by_mode,
            "raw_aggregate": aggregate,
            "aggregate": public_aggregate,
            "agreement": agreement,
            "reference_outputs": reference,
            "outputs": outputs_by_mode,
            "candidate_matches_baseline_exact": direct_parity,
            "parity_status": "exact" if direct_parity else "review_required",
            "relative_claim_eligible": direct_parity,
        }
        comparison = aggregate["candidate_vs_baseline"]
        print(
            f"[mixed-load] burst={burst_size} shape={burst_shape} "
            f"tokens={row['burst_prompt_tokens_total']}: "
            f"spanning-ITL {comparison['spanning_itl_ratio']:.2f}x, "
            f"burst-TTFT {comparison['burst_ttft_p50_ratio']:.2f}x, "
            f"candidate={agreement['candidate']['status']}, "
            f"relative={'eligible' if direct_parity else 'withheld; raw timing only'}"
        )
        yield {"kind": "row", **row}


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def diagnose_prefill_divergence(
    batch_size: int,
    shape: str,
    context_length: int | None,
    max_new_tokens: int,
    repeats: int,
) -> dict[str, object]:
    """Rerun one offending A/B cell outside the timed path, recording why serial vs packed diverge.

    For each attempt and mode this drives the engine directly (not the timed ``run_once``),
    snapshots prompt-length KV right after prefill — before decode recycles any block table —
    compares serial-vs-packed KV per layer per request, replays the bf16 logit margin at the
    first divergence step with the model's own prefill/decode methods, and reclassifies against
    the fp32 oracle. Diagnostic only: no serial or fp32 fallback is added to the production
    engine and the 0.1-logit tie rule is untouched.
    """
    import math

    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_paged import (
        HEADLINE_PROMPTS,
        EsmeBenchRequest,
        build_requests,
        reference_outputs,
        requests_at_context_length,
    )
    from llm_infer.benchmarks.esme_three_way import BF16_AGREEMENT_TOLERANCE
    from llm_infer.kv_cache.block_table import BlockTable
    from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving import InferenceEngine, Request
    from llm_infer.validation.tie_tolerance import compare_under_tie_tolerance

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1; got {repeats}")
    if shape not in ("uniform", "ragged"):
        raise ValueError(f"shape must be 'uniform' or 'ragged'; got {shape!r}")
    if shape == "uniform" and context_length is None:
        raise ValueError("uniform shape needs a context_length")

    esme_bundles.reload()
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    engine_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    eos = frozenset()
    # Match the decode-graph configuration the timed A/B run used so a "diverged" row reproduces.
    graph_capture_s = enable_decode_graphs_if_cuda(engine_runtime.model, CAPTURE_SIZES)
    model = engine_runtime.model
    num_layers = model.num_layers
    modes = {"baseline": False, "candidate": True}

    if shape == "ragged":
        base = build_requests(engine_runtime.tokenizer, batch_size, HEADLINE_PROMPTS)
        requests: list[EsmeBenchRequest] = []
        for index, request in enumerate(base):
            length = RAGGED_LENGTHS[index % len(RAGGED_LENGTHS)]
            resized = requests_at_context_length([request], length)[0]
            requests.append(
                EsmeBenchRequest(
                    request_id=resized.request_id,
                    prompt=f"{request.prompt} [ragged context: {length} tokens]",
                    prompt_ids=resized.prompt_ids,
                )
            )
    else:
        base = build_requests(engine_runtime.tokenizer, batch_size, HEADLINE_PROMPTS)
        requests = requests_at_context_length(base, context_length)

    reference = reference_outputs(
        oracle_runtime.model, requests, max_new_tokens=max_new_tokens, eos_token_ids=eos
    )

    def first_diff(fast: list[int], golden: list[int]) -> int | None:
        return next(
            (i for i, (f, g) in enumerate(zip(fast, golden, strict=False)) if f != g), None
        )

    def margin_fields(logits: torch.Tensor, fast_token: int, golden_token: int) -> dict[str, float]:
        values = logits.float()
        top2 = torch.topk(values, 2).values
        max_logit = float(top2[0].item())
        return {
            "fast_below": max_logit - float(values[fast_token].item()),
            "golden_below": max_logit - float(values[golden_token].item()),
            "top2_gap": float((top2[0] - top2[1]).item()),
            "fast_logit": float(values[fast_token].item()),
            "golden_logit": float(values[golden_token].item()),
        }

    def drive_engine(batched_prefill: bool) -> tuple[dict[str, list], dict[str, list[int]]]:
        """One engine run; return prompt-length KV snapshot and per-request generated tokens."""
        needed_blocks = sum(
            math.ceil((len(r.prompt_ids) + max_new_tokens) / BLOCK_SIZE) for r in requests
        )
        engine = InferenceEngine(
            model,
            block_size=BLOCK_SIZE,
            num_blocks=needed_blocks + max(4, len(requests)),
            device="cuda",
            capabilities=engine_runtime.capabilities,
            batched_prefill=batched_prefill,
        )
        for r in requests:
            engine.add_request(Request(r.request_id, list(r.prompt_ids), max_new_tokens, eos))

        first = engine.step()  # pure prefill: block tables now hold prompt-length KV
        # Snapshot before the decode loop: with ignore_eos every request runs to the token cap
        # and _release_finished_in recycles its block table at finish, so this is the only point
        # the prefill KV is still addressable. engine._requests is engine-internal; this is a
        # diagnostic driver reading it directly, not the production serving path.
        kv_snapshot: dict[str, list] = {}
        for r in requests:
            table = engine._requests[r.request_id].block_table
            if table is None:
                raise RuntimeError(f"request {r.request_id} has no block table after prefill")
            kv_snapshot[r.request_id] = [
                engine.cache.read(table, layer, len(r.prompt_ids)) for layer in range(num_layers)
            ]

        step_tokens: dict[str, list[int]] = {r.request_id: [] for r in requests}
        for request_id, tokens in first.tokens.items():
            step_tokens[request_id].extend(int(token) for token in tokens)
        while engine.scheduler.has_work():
            result = engine.step()
            for request_id, tokens in result.tokens.items():
                step_tokens[request_id].extend(int(token) for token in tokens)
        return kv_snapshot, step_tokens

    def kv_diffs(
        serial_kv: dict[str, list], packed_kv: dict[str, list], request_id: str
    ) -> tuple[list[float], list[float]]:
        max_by_layer: list[float] = []
        mean_by_layer: list[float] = []
        for layer in range(num_layers):
            serial_k, serial_v = serial_kv[request_id][layer]
            packed_k, packed_v = packed_kv[request_id][layer]
            k_abs = (serial_k.float() - packed_k.float()).abs()
            v_abs = (serial_v.float() - packed_v.float()).abs()
            max_by_layer.append(float(torch.maximum(k_abs.max(), v_abs.max()).item()))
            total = float((k_abs.sum() + v_abs.sum()).item())
            mean_by_layer.append(total / float(k_abs.numel() + v_abs.numel()))
        return max_by_layer, mean_by_layer

    def new_replay_cache(num_blocks: int) -> PagedKVCache:
        return PagedKVCache(
            num_layers=num_layers,
            num_blocks=num_blocks,
            block_size=BLOCK_SIZE,
            num_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
            dtype=model.dtype,
            device="cuda",
        )

    PackedReplayState = tuple[PagedKVCache, dict[str, BlockTable], dict[str, torch.Tensor]]

    def packed_replay_state() -> PackedReplayState:
        """Full-batch packed prefill replay: the same packed group shape as the real run.

        The serial-vs-packed numeric difference under diagnosis comes from packing ALL the
        cell's prompts into one group (one packed kernel launch, batch-dependent reduction
        shapes). A packed group of one would collapse to serial-prefill numerics and report a
        margin from KV the divergent run never saw, so the candidate replay must prefill the
        whole batch in the real run's request order. Safe to share across a single attempt's
        diverged requests: each decode replay advances only its own request's table.
        """
        needed = sum(
            math.ceil((len(r.prompt_ids) + max_new_tokens) / BLOCK_SIZE) for r in requests
        ) + max(4, len(requests))
        cache = new_replay_cache(needed)
        tables = [cache.new_request() for _ in requests]
        logits = model.prefill_many([list(r.prompt_ids) for r in requests], cache, tables)
        return (
            cache,
            {r.request_id: table for r, table in zip(requests, tables, strict=True)},
            {r.request_id: logits[i] for i, r in enumerate(requests)},
        )

    def engine_margin(
        batched_prefill: bool,
        request_id: str,
        prompt_ids: list[int],
        step_index: int,
        fast_tokens: list[int],
        golden_token: int,
        packed_state: PackedReplayState | None,
    ) -> dict[str, float]:
        """bf16 logits at ``step_index`` from the model's own path, replaying the fast tokens.

        Candidate mode reads its prefill logits and KV from the full-batch packed replay in
        ``packed_state``; baseline prefill is per-request and batch-independent, so it replays
        alone. Decode steps replay through eager single-request ``decode_one`` while the real
        run used graphed batched decode, so the margin is a close approximation of the batched
        decode logits, not a bit-exact reproduction (stated in ``config.engine_margin``).
        """
        if batched_prefill:
            if packed_state is None:
                raise ValueError("candidate engine_margin needs the packed replay state")
            cache, tables, prefill_logits = packed_state
            table = tables[request_id]
            logits = prefill_logits[request_id]
        else:
            cache = new_replay_cache(
                math.ceil((len(prompt_ids) + step_index + 1) / BLOCK_SIZE) + 2
            )
            table = cache.new_request()
            logits = model.prefill(prompt_ids, cache, table)
        for t in range(1, step_index + 1):
            logits = model.decode_one(cache, table, fast_tokens[t - 1])
        return margin_fields(logits, fast_tokens[step_index], golden_token)

    def oracle_margin(
        prompt_ids: list[int], golden: list[int], step_index: int, fast_token: int
    ) -> dict[str, float]:
        logits = oracle_runtime.model.logits(prompt_ids + golden[:step_index])[-1]
        return margin_fields(logits, fast_token, golden[step_index])

    records: list[dict[str, object]] = []
    for attempt in range(repeats):
        serial_kv, serial_tokens = drive_engine(False)
        packed_kv, packed_tokens = drive_engine(True)
        tokens_by_mode = {"baseline": serial_tokens, "candidate": packed_tokens}
        # One full-batch packed prefill replay per attempt, built only if a candidate-mode
        # divergence needs an engine margin; shared across this attempt's diverged requests.
        packed_state: PackedReplayState | None = None
        for r in requests:
            request_id = r.request_id
            golden = reference[request_id]
            prompt_ids = list(r.prompt_ids)
            # KV closeness is a serial-vs-packed property; the same value is attached to both
            # mode records for this request/attempt so the flat schema stays uniform.
            kv_max, kv_mean = kv_diffs(serial_kv, packed_kv, request_id)
            for mode in modes:
                fast_tokens = tokens_by_mode[mode][request_id]
                step_index = first_diff(fast_tokens, golden)
                record: dict[str, object] = {
                    "request_id": request_id,
                    "mode": mode,
                    "attempt": attempt,
                    "first_diff_step": step_index,
                    "first_diff_tokens": {
                        "fast": fast_tokens[step_index] if step_index is not None else None,
                        "golden": golden[step_index] if step_index is not None else None,
                    },
                    "kv_max_abs_diff_by_layer": kv_max,
                    "kv_mean_abs_diff_by_layer": kv_mean,
                    "decoded_fast": engine_runtime.tokenizer.decode(fast_tokens),
                    "decoded_golden": engine_runtime.tokenizer.decode(golden),
                    "oracle_margin": None,
                    "engine_margin": None,
                    "oracle_verdict": None,
                }
                if step_index is not None:
                    fast_token = fast_tokens[step_index]
                    record["oracle_margin"] = oracle_margin(
                        prompt_ids, golden, step_index, fast_token
                    )
                    if modes[mode] and packed_state is None:
                        packed_state = packed_replay_state()
                    record["engine_margin"] = engine_margin(
                        modes[mode],
                        request_id,
                        prompt_ids,
                        step_index,
                        fast_tokens,
                        golden[step_index],
                        packed_state,
                    )
                    # Reuse the actual 0.1-logit rule for the verdict rather than re-deriving it.
                    result = compare_under_tie_tolerance(
                        oracle_runtime.model,
                        prompt_ids,
                        fast_tokens,
                        golden,
                        tolerance=BF16_AGREEMENT_TOLERANCE,
                    )
                    record["oracle_verdict"] = (
                        "tie" if result.ok and result.divergence is not None else "nontie"
                    )
                records.append(record)

    return {
        "records": records,
        "summary": {"stability": stability_summary(records)},
        "meta": {
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
            "attention_backend": type(model.backend).__name__,
            "decode_graphs": {"capture_sizes": list(CAPTURE_SIZES), "capture_s": graph_capture_s},
        },
        "config": {
            "command": "prefill-divergence",
            "model": "Esme-214M-Chat",
            "dtype": "bfloat16",
            "batch_size": batch_size,
            "shape": shape,
            "context_length": context_length,
            "ragged_context_cycle": list(RAGGED_LENGTHS) if shape == "ragged" else None,
            "max_new_tokens": max_new_tokens,
            "repeats": repeats,
            "block_size": BLOCK_SIZE,
            "baseline": "InferenceEngine(batched_prefill=False)",
            "candidate": "InferenceEngine(batched_prefill=True)",
            "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant",
            "tie_tolerance": BF16_AGREEMENT_TOLERANCE,
            "ignore_eos": True,
            "kv_snapshot": "prompt-length K/V read after prefill, before the decode loop",
            "engine_margin": (
                "candidate prefill replayed as the full packed batch in request order; decode "
                "steps replayed via eager single-request decode_one while the real run used "
                "graphed batched decode — a close approximation, not bit-exact"
            ),
            "repro_command": (
                "modal run scripts/modal_esme_prefill_ab.py --command prefill-divergence"
            ),
        },
    }


def _run_prefill_divergence(bundle_path: str, context_length: int) -> None:
    """Rerun one batch-8 64-token cell with extra recording; write the divergence JSON.

    ``context_length`` 0 selects the ragged cell; a positive value selects that uniform cell.
    Single cheap cell, so no rows-log / resume machinery: it writes one timestamped file whose
    name cannot collide with the A/B rows log.
    """
    if context_length < 0:
        raise ValueError(f"context-length must be 0 (ragged) or positive; got {context_length}")
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    local_bundle = local_bundle_path(bundle_path)
    stage_bundle(esme_bundles, local_bundle, label="esme-prefill-divergence")
    record = diagnose_prefill_divergence.remote(
        batch_size=8,
        shape="ragged" if context_length == 0 else "uniform",
        context_length=None if context_length == 0 else context_length,
        max_new_tokens=64,
        repeats=5,
    )
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-prefill-divergence-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-prefill-divergence] wrote {out_path}")


@app.local_entrypoint()
def main(
    command: str = "prefill-ab",
    bundle_path: str = "",
    resume: bool = False,
    context_length: int = 0,
) -> None:
    """Stage Esme, stream a benchmark protocol to a rows log, and write the combined JSON record."""
    if command == "prefill-divergence":
        _run_prefill_divergence(bundle_path, context_length)
        return
    if context_length != 0:
        raise ValueError("--context-length only applies to --command prefill-divergence")

    prefill_commands = {"prefill-smoke", "prefill-ab"}
    mixed_commands = {"mixed-load-smoke", "mixed-load"}
    if command in prefill_commands:
        key_fields = PREFILL_AB_KEY_FIELDS
    elif command in mixed_commands:
        key_fields = MIXED_LOAD_KEY_FIELDS
    else:
        raise ValueError(
            "command must be 'prefill-smoke', 'prefill-ab', 'prefill-divergence', "
            f"'mixed-load-smoke', or 'mixed-load'; got {command!r}"
        )

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
    elif command == "mixed-load-smoke":
        # One cheap cell, short outputs, one measured pair — a Modal-side wiring check.
        cells = [{"burst_size": 8, "burst_shape": "ragged", "burst_context": None}]
        steady_decoders, steady_max_new_tokens, burst_max_new_tokens = 8, 32, 8
        warmup_tokens = 4
        warmup, iters = 1, 1
    else:  # mixed-load
        cells = MIXED_LOAD_CELLS
        steady_decoders, steady_max_new_tokens, burst_max_new_tokens = 8, 256, 16
        warmup_tokens = 8
        warmup, iters = 1, 5

    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    # Deterministic (no timestamp) so --resume finds the log of a prior interrupted run.
    rows_path = out_dir / f"esme-{command}-rows.jsonl"

    completed: list[dict[str, object]] = []
    if resume:
        if not rows_path.is_file():
            raise FileNotFoundError(
                f"--resume set but no rows log at {rows_path}; drop --resume to start a fresh run"
            )
        completed = completed_row_keys(
            parse_event_lines(rows_path.read_text(encoding="utf-8")), key_fields
        )
        print(f"[esme-{command}] resuming {rows_path} with {len(completed)} completed rows")
    elif rows_path.is_file():
        raise FileExistsError(
            f"{rows_path} already holds partial results; pass --resume to continue it "
            "or move the file to start fresh"
        )

    local_bundle = local_bundle_path(bundle_path)
    stage_bundle(esme_bundles, local_bundle, label=f"esme-{command}")

    if command in prefill_commands:
        events = benchmark_prefill_ab.remote_gen(
            batch_sizes,
            uniform_lengths,
            output_lengths,
            include_ragged,
            warmup,
            iters,
            completed,
        )
        config = {
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
            "repro_command": f"modal run scripts/modal_esme_prefill_ab.py --command {command}",
        }
    else:
        events = benchmark_mixed_load.remote_gen(
            cells,
            steady_decoders,
            steady_max_new_tokens,
            burst_max_new_tokens,
            warmup_tokens,
            warmup,
            iters,
            completed,
        )
        config = {
            "command": command,
            "model": "Esme-214M-Chat",
            "dtype": "bfloat16",
            "steady_decoders": steady_decoders,
            "steady_shape": "ragged",
            "ragged_context_cycle": list(RAGGED_LENGTHS),
            "steady_max_new_tokens": steady_max_new_tokens,
            "burst_max_new_tokens": burst_max_new_tokens,
            "warmup_tokens": warmup_tokens,
            "cells": cells,
            "warmup_pairs": warmup,
            "measured_pairs": iters,
            "block_size": BLOCK_SIZE,
            "baseline": "InferenceEngine(batched_prefill=False)",
            "candidate": "InferenceEngine(batched_prefill=True)",
            "reference": (
                "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant over "
                "steady+burst requests"
            ),
            "ignore_eos": True,
            "preemption": False,
            "decode_window_size": 1,
            "burst_admission": (
                "whole burst admitted in one reserve-mode step (generously sized pool), "
                "preemption off — the worst-case single packed prefill"
            ),
            "timing": {
                "engine_startup": "excluded",
                "per_step": (
                    "torch.cuda.synchronize() brackets every engine.step(); per-step wall is "
                    "real host time, unlike the throughput harness"
                ),
                "itl": (
                    "one decode token per step (decode_window_size=1), so a steady decoder's "
                    "inter-token latency is the wall time of the step that produced its token"
                ),
                "spanning_itl": "the inter-token gap that straddles the burst-admission step",
                "burst_ttft": "burst add_request wall to its first sampled token",
                "scheduler_sync_residual": (
                    "coarse: total wall minus prefill+decode+sampling device ms; lumps "
                    "admission, Python classification, and per-step syncs together"
                ),
                "pairing": "baseline/candidate order alternates each pair",
            },
            "same_container_and_gpu": True,
            "repro_command": f"modal run scripts/modal_esme_prefill_ab.py --command {command}",
        }

    # Append on resume so earlier rows survive; each event is flushed the moment it arrives.
    with rows_path.open("a" if resume else "w", encoding="utf-8") as rows_log:
        for event in events:
            rows_log.write(json.dumps(event) + "\n")
            rows_log.flush()

    # Rebuild from the full log (old plus new) so a resumed run emits every row it ever measured.
    record = assemble_final_record(
        parse_event_lines(rows_path.read_text(encoding="utf-8")), config
    )
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-{command}] wrote rows log {rows_path}")
    print(f"[esme-{command}] wrote {out_path}")
