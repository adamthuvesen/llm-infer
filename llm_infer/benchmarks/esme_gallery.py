"""Technique-gallery experiments: one targeted, oracle-gated measurement per technique.

Each experiment isolates one serving technique on a workload built to exercise it, and
returns the numbers the public benchmark story cites:

* :func:`run_prefix_cache_on_off` — sibling requests with one shared prompt, prefix caching
  on vs off. The delta is the prefill work the shared prompt blocks save.
* :func:`run_chunked_prefill_latency` — long prompts arrive while short requests decode;
  chunked vs whole-prompt prefill. The metric is the worst inter-token stall of the
  in-flight decodes, plus the total-wall cost chunking pays for that protection.
* :func:`run_preemption_starved_pool` — more requests than the KV pool can hold at once,
  recompute preemption on vs the reserve scheduler. The evidence is exact completions under
  real evictions, not throughput.
* :func:`run_speculative_batch1` — one request over repetition-heavy text, prompt-lookup
  speculation on vs off. Its honest niche: batch-1 latency when drafts actually match.

Per the repo rule, every experiment gates its outputs against the fp32
``PretrainBundleModel.logits()`` oracle with the audited tie-tolerant rule before any
timing is worth reporting. The functions are pure (no Modal import) so the same experiments
run on CPU with the tiny synthetic bundle in tests; the A100 runs live in
``scripts/modal_esme_technique_gallery.py``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from llm_infer.benchmarks.esme_paged import EsmeBenchRequest, _time, reference_outputs
from llm_infer.benchmarks.esme_three_way import EsmeAgreement, tie_tolerant_agreement
from llm_infer.model.runtime import ModelRuntime
from llm_infer.serving import InferenceEngine, Request
from llm_infer.serving.speculative import SpeculativeDecodingConfig
from llm_infer.tracing import TraceRecorder


@dataclass(frozen=True)
class GalleryTiming:
    """One gallery configuration's gated result: wall-clock plus its agreement profile."""

    label: str
    median_seconds: float
    total_output_tokens: int
    agreement: EsmeAgreement

    @property
    def tokens_per_second(self) -> float | None:
        if not self.agreement.all_ties_or_exact or self.median_seconds <= 0:
            return None
        return self.total_output_tokens / self.median_seconds

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "median_seconds": self.median_seconds,
            "total_output_tokens": self.total_output_tokens,
            "tokens_per_second": self.tokens_per_second,
            "matches_reference": self.agreement.all_ties_or_exact,
            "agreement": _agreement_dict(self.agreement),
        }


def _agreement_dict(agreement: EsmeAgreement) -> dict[str, object]:
    return {
        "exact": agreement.exact,
        "tie": agreement.tie,
        "nontie": agreement.nontie,
        "total": agreement.total,
        "ties_sample": agreement.ties_sample,
        "divergences_sample": agreement.divergences_sample,
    }


def _bench_requests(prompts: Sequence[Sequence[int]], prefix: str) -> list[EsmeBenchRequest]:
    return [
        EsmeBenchRequest(f"{prefix}-{index:03d}", "", tuple(int(t) for t in prompt))
        for index, prompt in enumerate(prompts)
    ]


def _gate(
    oracle_runtime: ModelRuntime,
    requests: list[EsmeBenchRequest],
    outputs: dict[str, list[int]],
    *,
    max_new_tokens: int,
) -> tuple[EsmeAgreement, int]:
    """Tie-tolerant agreement vs the fp32 oracle, plus the scored token total."""
    from llm_infer.benchmarks.report import total_output_tokens

    reference = reference_outputs(
        oracle_runtime.model,
        requests,
        max_new_tokens=max_new_tokens,
        eos_token_ids=oracle_runtime.eos_token_ids,
    )
    agreement = tie_tolerant_agreement(
        oracle_runtime.model, requests, outputs, reference, oracle_runtime.eos_token_ids
    )
    return agreement, total_output_tokens(outputs, oracle_runtime.eos_token_ids)


def run_prefix_cache_on_off(
    oracle_runtime: ModelRuntime,
    engine_runtime: ModelRuntime,
    *,
    prompt_ids: Sequence[int],
    num_siblings: int,
    max_new_tokens: int,
    block_size: int,
    num_blocks: int,
    device: str,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    """Sibling requests sharing one prompt: prefix caching on (one prefill) vs off (N)."""
    requests = _bench_requests([prompt_ids] * num_siblings, "prefix")
    eos = engine_runtime.eos_token_ids
    sync = device == "cuda"

    def run_once(group: str | None) -> dict[str, list[int]]:
        engine = InferenceEngine(
            engine_runtime.model,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
            capabilities=engine_runtime.capabilities,
        )
        for req in requests:
            engine.add_request(
                Request(
                    req.request_id,
                    list(req.prompt_ids),
                    max_new_tokens,
                    eos,
                    prefix_group_id=group,
                )
            )
        return engine.run()

    timings: list[GalleryTiming] = []
    for label, group in (("prefix caching on", "gallery-shared"), ("prefix caching off", None)):
        median_s, outputs = _time(
            lambda g=group: run_once(g), warmup=warmup, iters=iters, sync=sync
        )
        agreement, tokens = _gate(
            oracle_runtime, requests, outputs, max_new_tokens=max_new_tokens
        )
        timings.append(GalleryTiming(label, median_s, tokens, agreement))

    on, off = timings
    return {
        "experiment": "prefix-caching",
        "workload": {
            "num_siblings": num_siblings,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": max_new_tokens,
            "prefilled_prompt_tokens_on": len(prompt_ids),
            "prefilled_prompt_tokens_off": len(prompt_ids) * num_siblings,
        },
        "rows": [t.as_dict() for t in timings],
        "wall_speedup_on_vs_off": (
            off.median_seconds / on.median_seconds if on.median_seconds > 0 else None
        ),
    }


def run_chunked_prefill_latency(
    oracle_runtime: ModelRuntime,
    engine_runtime: ModelRuntime,
    *,
    active_prompts: Sequence[Sequence[int]],
    active_max_new_tokens: int,
    long_prompts: Sequence[Sequence[int]],
    long_max_new_tokens: int,
    arrival_after_steps: int,
    prefill_chunk_size: int,
    block_size: int,
    num_blocks: int,
    device: str,
) -> dict[str, object]:
    """Long prompts arrive mid-decode: worst in-flight stall with vs without chunking.

    Runs one pass per config with ``decode_window_size=1`` so each engine step is a host
    boundary and per-step token timestamps are real (the deferred window would quantize
    them). The stall metric is the maximum inter-token gap any already-decoding request
    sees after the long prompts arrive, normalized by the pre-arrival median step time.
    """
    active_requests = _bench_requests(active_prompts, "active")
    long_requests = _bench_requests(long_prompts, "long")
    eos = engine_runtime.eos_token_ids
    sync = device == "cuda"

    def run_once(chunk: int | None) -> dict[str, object]:
        engine = InferenceEngine(
            engine_runtime.model,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
            capabilities=engine_runtime.capabilities,
            prefill_chunk_size=chunk,
            decode_window_size=1,
        )
        for req in active_requests:
            engine.add_request(
                Request(req.request_id, list(req.prompt_ids), active_max_new_tokens, eos)
            )

        token_times: dict[str, list[float]] = {}
        outputs: dict[str, list[int]] = {}
        prefill_tokens_per_step: list[int] = []

        def step() -> None:
            result = engine.step()
            if sync:
                torch.cuda.synchronize()
            now = time.perf_counter()
            prefill_tokens_per_step.append(
                sum(end - start for start, end in result.prefill_chunks.values())
            )
            for request_id, tokens in result.tokens.items():
                token_times.setdefault(request_id, []).extend([now] * len(tokens))
            outputs.update(result.finished_outputs)

        for _ in range(arrival_after_steps):
            if not engine.scheduler.has_work():
                raise ValueError(
                    "active requests finished before the long prompts arrived; raise "
                    "active_max_new_tokens or lower arrival_after_steps"
                )
            step()
        arrival_s = time.perf_counter()
        for req in long_requests:
            engine.add_request(
                Request(req.request_id, list(req.prompt_ids), long_max_new_tokens, eos)
            )
        while engine.scheduler.has_work():
            step()

        pre_arrival_gaps: list[float] = []
        post_arrival_gaps: list[float] = []
        for req in active_requests:
            times = token_times.get(req.request_id, [])
            for left, right in zip(times, times[1:], strict=False):
                gap = right - left
                (post_arrival_gaps if right > arrival_s else pre_arrival_gaps).append(gap)
        if not pre_arrival_gaps or not post_arrival_gaps:
            raise ValueError("active requests must decode both before and after the arrival")
        long_finish = max(
            (max(token_times[req.request_id]) for req in long_requests), default=arrival_s
        )
        return {
            "prefill_chunk_size": chunk,
            "max_prefill_tokens_in_one_step": max(prefill_tokens_per_step),
            "pre_arrival_step_p50_s": _median(pre_arrival_gaps),
            "post_arrival_max_gap_s": max(post_arrival_gaps),
            "long_prompts_done_after_s": long_finish - arrival_s,
            "total_wall_s": max(t for times in token_times.values() for t in times)
            - min(t for times in token_times.values() for t in times),
            "outputs": outputs,
        }

    all_requests = active_requests + long_requests
    configs = [("whole-prompt prefill", None), ("chunked prefill", prefill_chunk_size)]
    rows: list[dict[str, object]] = []
    for label, chunk in configs:
        run = run_once(chunk)
        outputs = run.pop("outputs")
        max_new_by_id = {
            **{r.request_id: active_max_new_tokens for r in active_requests},
            **{r.request_id: long_max_new_tokens for r in long_requests},
        }
        agreement = _gate_mixed(oracle_runtime, all_requests, outputs, max_new_by_id)
        stall = run["post_arrival_max_gap_s"] / run["pre_arrival_step_p50_s"]
        rows.append(
            {
                "label": label,
                **run,
                "stall_vs_pre_arrival_step": stall,
                "matches_reference": agreement.all_ties_or_exact,
                "agreement": _agreement_dict(agreement),
            }
        )
    return {
        "experiment": "chunked-prefill",
        "workload": {
            "active_requests": len(active_requests),
            "active_prompt_tokens": [len(p) for p in active_prompts],
            "active_max_new_tokens": active_max_new_tokens,
            "long_requests": len(long_requests),
            "long_prompt_tokens": [len(p) for p in long_prompts],
            "long_max_new_tokens": long_max_new_tokens,
            "arrival_after_steps": arrival_after_steps,
            "prefill_chunk_size": prefill_chunk_size,
            "decode_window_size": 1,
        },
        "rows": rows,
    }


def run_preemption_starved_pool(
    oracle_runtime: ModelRuntime,
    engine_runtime: ModelRuntime,
    *,
    prompts: Sequence[Sequence[int]],
    max_new_tokens: int,
    block_size: int,
    num_blocks: int,
    device: str,
) -> dict[str, object]:
    """A pool too small for the offered load: recompute preemption vs reserve admission.

    The claim under test is robustness, not speed: with real evictions the completions must
    still be token-exact against the oracle. The reserve scheduler on the same pool is the
    control — it queues instead of over-committing, so it never preempts.
    """
    requests = _bench_requests(prompts, "preempt")
    eos = engine_runtime.eos_token_ids
    sync = device == "cuda"

    def run_once(preemption: bool) -> tuple[dict[str, list[int]], int, float]:
        engine = InferenceEngine(
            engine_runtime.model,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
            capabilities=engine_runtime.capabilities,
            preemption=preemption,
        )
        for req in requests:
            engine.add_request(Request(req.request_id, list(req.prompt_ids), max_new_tokens, eos))
        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.run()
        if sync:
            torch.cuda.synchronize()
        return outputs, engine.preemption_count, time.perf_counter() - start

    rows: list[dict[str, object]] = []
    preemption_count_on = 0
    for label, preemption in (("preemption on", True), ("reserve scheduler", False)):
        outputs, preemptions, wall_s = run_once(preemption)
        agreement, tokens = _gate(oracle_runtime, requests, outputs, max_new_tokens=max_new_tokens)
        if preemption:
            preemption_count_on = preemptions
        rows.append(
            {
                "label": label,
                "preemptions": preemptions,
                "wall_s": wall_s,
                "completed_requests": len(outputs),
                "total_output_tokens": tokens,
                "matches_reference": agreement.all_ties_or_exact,
                "agreement": _agreement_dict(agreement),
            }
        )
    if preemption_count_on == 0:
        raise ValueError(
            "the starved pool produced zero preemptions; shrink num_blocks or grow the "
            "workload — a preemption-free run proves nothing"
        )
    return {
        "experiment": "preemption",
        "workload": {
            "num_requests": len(requests),
            "prompt_tokens": [len(p) for p in prompts],
            "max_new_tokens": max_new_tokens,
            "block_size": block_size,
            "num_blocks": num_blocks,
        },
        "rows": rows,
    }


def run_speculative_batch1(
    oracle_runtime: ModelRuntime,
    engine_runtime: ModelRuntime,
    *,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    speculative: SpeculativeDecodingConfig,
    block_size: int,
    num_blocks: int,
    device: str,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    """One repetition-heavy request: prompt-lookup speculation vs the default engine."""
    requests = _bench_requests([prompt_ids], "spec")
    eos = engine_runtime.eos_token_ids
    sync = device == "cuda"

    def run_once(config: SpeculativeDecodingConfig | None) -> dict[str, list[int]]:
        engine = InferenceEngine(
            engine_runtime.model,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
            capabilities=engine_runtime.capabilities,
            speculative=config,
        )
        engine.add_request(
            Request(requests[0].request_id, list(prompt_ids), max_new_tokens, eos)
        )
        return engine.run()

    timings: list[GalleryTiming] = []
    for label, config in (("speculative on", speculative), ("speculative off", None)):
        median_s, outputs = _time(
            lambda c=config: run_once(c), warmup=warmup, iters=iters, sync=sync
        )
        agreement, tokens = _gate(
            oracle_runtime, requests, outputs, max_new_tokens=max_new_tokens
        )
        timings.append(GalleryTiming(label, median_s, tokens, agreement))

    # Acceptance profile from one traced pass: how many tokens each verify step emitted.
    trace = TraceRecorder()
    engine = InferenceEngine(
        engine_runtime.model,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        capabilities=engine_runtime.capabilities,
        speculative=speculative,
        trace=trace,
    )
    engine.add_request(Request(requests[0].request_id, list(prompt_ids), max_new_tokens, eos))
    engine.run()
    speculative_steps = [
        event for event in trace.events
        if event.event == "decode_step" and event.token_source == "speculative"
    ]
    emitted = [len(event.token_ids or []) for event in speculative_steps]

    on, off = timings
    return {
        "experiment": "speculative-decoding",
        "workload": {
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": max_new_tokens,
            "max_draft_tokens": speculative.max_draft_tokens,
            "max_ngram_size": speculative.max_ngram_size,
        },
        "rows": [t.as_dict() for t in timings],
        "latency_speedup_on_vs_off": (
            off.median_seconds / on.median_seconds if on.median_seconds > 0 else None
        ),
        "verify_steps": len(speculative_steps),
        "mean_tokens_per_verify_step": (
            sum(emitted) / len(emitted) if emitted else None
        ),
    }


def _gate_mixed(
    oracle_runtime: ModelRuntime,
    requests: list[EsmeBenchRequest],
    outputs: dict[str, list[int]],
    max_new_by_id: dict[str, int],
) -> EsmeAgreement:
    """Agreement for a workload whose requests carry different token budgets."""
    from llm_infer.model.decode import greedy_decode

    by_key: dict[tuple[tuple[int, ...], int], list[int]] = {}
    reference: dict[str, list[int]] = {}
    for req in requests:
        key = (req.prompt_ids, max_new_by_id[req.request_id])
        if key not in by_key:
            by_key[key] = greedy_decode(
                oracle_runtime.model,
                list(req.prompt_ids),
                max_new_tokens=max_new_by_id[req.request_id],
                eos_token_ids=set(oracle_runtime.eos_token_ids),
            )
        reference[req.request_id] = list(by_key[key])
    return tie_tolerant_agreement(
        oracle_runtime.model, requests, outputs, reference, oracle_runtime.eos_token_ids
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
