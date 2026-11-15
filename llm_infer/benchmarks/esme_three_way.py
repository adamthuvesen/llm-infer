"""vLLM decode closures and the tie-tolerant fp32 agreement rule for Esme benchmarks.

vLLM runs on the ``Qwen3ForCausalLM`` checkpoint that ``scripts/convert_esme_to_hf.py`` emits
from the bundle, with prefix caching off. vLLM is imported only inside the functions, so this
module loads on CPU without it.

Agreement uses policy v2 against the fp32 reference. A bf16 flip whose fp32 gap is under the 0.1
bf16 tolerance is a genuine tie, not a bug.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol, cast

from llm_infer.benchmarks.esme_paged import EsmeBenchRequest
from llm_infer.benchmarks.reference_policy import BF16_AGREEMENT_TOLERANCE, EsmeAgreement
from llm_infer.benchmarks.report import normalize_at_eos
from llm_infer.validation.tie_tolerance import LogitsOracle

DecodeOnce = Callable[[], dict[str, list[int]]]


class VllmCompletion(Protocol):
    token_ids: Sequence[int]


class VllmRequestOutput(Protocol):
    outputs: Sequence[VllmCompletion]


class VllmEngine(Protocol):
    def generate(
        self, prompts: list[dict[str, list[int]]], params: object, *, use_tqdm: bool
    ) -> Sequence[VllmRequestOutput]: ...


def build_vllm_llm(
    hf_checkpoint: Path | str,
    *,
    max_model_len: int,
    gpu_memory_utilization: float = 0.6,
) -> tuple[VllmEngine, dict[str, object]]:
    """Build ONE vLLM engine (prefix caching off) plus its pinned config.

    A process gets exactly one engine: vLLM's memory reservation
    (``gpu_memory_utilization``) is never returned while the process lives, so a second
    ``LLM()`` in the same process fails engine-core startup on free-memory checks.
    Multi-workload runs reuse this engine via :func:`vllm_decode_closure`.
    """
    vllm = importlib.import_module("vllm")

    llm = cast(
        VllmEngine,
        vllm.LLM(
            model=str(hf_checkpoint),
            dtype="bfloat16",
            enable_prefix_caching=False,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            gpu_memory_utilization=gpu_memory_utilization,
        ),
    )
    config: dict[str, object] = {
        "version": vllm.__version__,
        "dtype": "bfloat16",
        "enable_prefix_caching": False,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "tensor_parallel_size": 1,
        "served_checkpoint": "converted Qwen3 (from Esme bundle)",
    }
    return llm, config


def vllm_decode_closure(
    llm: VllmEngine,
    requests: list[EsmeBenchRequest],
    *,
    max_new_tokens: int,
    eos_token_ids: frozenset[int],
    ignore_eos: bool = False,
) -> DecodeOnce:
    """The greedy offline-generate closure for one workload on an already-built engine."""
    params = importlib.import_module("vllm").SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        n=1,
        stop_token_ids=[] if ignore_eos else sorted(eos_token_ids),
        ignore_eos=ignore_eos,
    )
    prompts = [{"prompt_token_ids": list(req.prompt_ids)} for req in requests]
    ids_by_index = [req.request_id for req in requests]

    def decode_once() -> dict[str, list[int]]:
        results = llm.generate(prompts, params, use_tqdm=False)
        return {
            ids_by_index[i]: [int(token) for token in results[i].outputs[0].token_ids]
            for i in range(len(results))
        }

    return decode_once


def tie_tolerant_agreement(
    oracle_model: LogitsOracle,
    requests: list[EsmeBenchRequest],
    outputs: dict[str, list[int]],
    reference: dict[str, list[int]],
    eos_token_ids: frozenset[int],
    *,
    tolerance: float = BF16_AGREEMENT_TOLERANCE,
) -> EsmeAgreement:
    """Classify a system's tokens vs the fp32 oracle for policy-v2 review.

    For each request, exact match counts as exact; otherwise the first divergence is recomputed on
    the fp32 oracle via ``compare_under_tie_tolerance``. Within ``tolerance`` it is automatically
    accepted numerical behavior; beyond it needs review unless the mismatch is structural.
    Missing requests fail individually. Unexpected request IDs invalidate the complete batch.
    """
    from llm_infer.validation.tie_tolerance import compare_under_tie_tolerance

    prompts_by_id = {req.request_id: list(req.prompt_ids) for req in requests}
    exact = 0
    ties: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    missing = sorted(set(reference) - set(outputs))
    extra = sorted(set(outputs) - set(reference))
    if extra:
        return EsmeAgreement(
            exact=0,
            tie=0,
            nontie=len(reference),
            total=len(reference),
            ties_sample=[],
            divergences_sample=[
                {
                    "detail": "unexpected outputs invalidate the complete reference batch",
                    "unexpected_request_ids": extra,
                    "missing_request_ids": missing,
                }
            ],
            failed=len(reference),
        )
    failures.extend(
        {"request": request_id, "detail": f"missing outputs for {request_id!r}"}
        for request_id in missing
    )
    for request_id in reference:
        if request_id not in outputs:
            continue
        raw = outputs[request_id]
        fast = normalize_at_eos(raw, eos_token_ids)
        gold = normalize_at_eos(reference[request_id], eos_token_ids)
        if fast == gold:
            exact += 1
            continue
        result = compare_under_tie_tolerance(
            oracle_model, prompts_by_id[request_id], fast, gold, tolerance=tolerance
        )
        if result.ok and result.divergence is not None:
            d = result.divergence
            ties.append(
                {
                    "request": request_id,
                    "step": d.step,
                    "fast_token": d.fast_token,
                    "golden_token": d.golden_token,
                    "fp32_margin": abs(d.fast_logit - d.golden_logit),
                    "reference_gap": d.reference_gap,
                    "fast_logit": d.fast_logit,
                    "golden_logit": d.golden_logit,
                }
            )
        elif not result.ok:
            numerical = compare_under_tie_tolerance(
                oracle_model,
                prompts_by_id[request_id],
                fast,
                gold,
                tolerance=float("inf"),
            )
            if numerical.divergence is None:
                failures.append({"request": request_id, "detail": result.failure})
                continue
            d = numerical.divergence
            reviews.append(
                {
                    "request": request_id,
                    "step": d.step,
                    "fast_token": d.fast_token,
                    "golden_token": d.golden_token,
                    "fp32_margin": abs(d.fast_logit - d.golden_logit),
                    "reference_gap": d.reference_gap,
                    "fast_logit": d.fast_logit,
                    "golden_logit": d.golden_logit,
                    "automatic_boundary": tolerance,
                    "detail": result.failure,
                }
            )
    divergences = [*reviews, *failures]
    return EsmeAgreement(
        exact=exact,
        tie=len(ties),
        nontie=len(divergences),
        total=len(reference),
        ties_sample=ties[:3],
        divergences_sample=divergences[:3],
        review_required=len(reviews),
        failed=len(failures),
        numerical_evidence=[*ties, *reviews],
    )
