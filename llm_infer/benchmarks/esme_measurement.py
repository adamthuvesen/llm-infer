"""Reference qualification for measured Esme benchmark rows."""

from __future__ import annotations

import statistics

from llm_infer.benchmarks.esme_paged import EsmeBenchRequest
from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
from llm_infer.benchmarks.reference_policy import SystemEvidence
from llm_infer.benchmarks.report import total_output_tokens
from llm_infer.model.interface import CausalLMBackend


def qualified_system_row(
    model: CausalLMBackend,
    requests: list[EsmeBenchRequest],
    outputs: dict[str, list[int]],
    reference: dict[str, list[int]],
    eos_token_ids: frozenset[int],
    per_iter_seconds: list[float],
) -> dict[str, object]:
    """Serialize complete agreement and measured throughput under the reference policy."""
    agreement = tie_tolerant_agreement(model, requests, outputs, reference, eos_token_ids)
    total_tokens = total_output_tokens(outputs, eos_token_ids)
    median_seconds = statistics.median(per_iter_seconds)
    return {
        "agreement": agreement.as_dict(),
        "median_seconds": median_seconds,
        "per_iter_seconds": list(per_iter_seconds),
        "total_output_tokens": total_tokens,
        **SystemEvidence.from_timing(
            agreement=agreement, median_seconds=median_seconds, total_tokens=total_tokens
        ).as_dict(),
    }
