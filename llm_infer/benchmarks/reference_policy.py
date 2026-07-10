"""Policy-v2 evidence contract for Esme benchmark records.

Raw timing is always kept. Public tok/s is a separate claim: a row reports
``tokens_per_second`` only when its reference status qualifies (``headline_eligible``).
``review_required`` marks a divergence above the automatic tie boundary that is waiting on a
durable review — it withholds the public number, it does not mean the row is broken.
"""

from __future__ import annotations

from typing import Literal

from llm_infer.benchmarks.esme_three_way import EsmeAgreement
from llm_infer.benchmarks.report import normalize_at_eos

POLICY_VERSION = 2
ReferenceStatus = Literal["exact", "accepted_numerical", "review_required", "failed"]

QUALIFIED_REFERENCE_STATUSES = frozenset({"exact", "accepted_numerical"})


def reference_status(agreement: EsmeAgreement) -> ReferenceStatus:
    """Classify one system's fp32 agreement profile.

    Every request must be classified: exact, tie, or non-tie, with each non-tie counted as
    either ``review_required`` or ``failed``. An agreement whose counts do not add up is a
    construction bug, not a benchmark result, so it raises instead of guessing.
    """
    if agreement.total < 1:
        raise ValueError("agreement covers no requests")
    if agreement.exact + agreement.tie + agreement.nontie != agreement.total:
        raise ValueError(
            "inconsistent agreement: exact + tie + nontie must equal total "
            f"(exact={agreement.exact}, tie={agreement.tie}, "
            f"nontie={agreement.nontie}, total={agreement.total})"
        )
    if agreement.nontie != agreement.review_required + agreement.failed:
        raise ValueError(
            "inconsistent agreement: nontie must equal review_required + failed "
            f"(nontie={agreement.nontie}, review_required={agreement.review_required}, "
            f"failed={agreement.failed})"
        )
    if agreement.failed:
        return "failed"
    if agreement.review_required:
        return "review_required"
    if agreement.tie:
        return "accepted_numerical"
    return "exact"


def normalized_outputs_match(
    baseline_outputs: dict[str, list[int]],
    candidate_outputs: dict[str, list[int]],
    eos_token_ids: frozenset[int],
) -> bool:
    """Exact direct parity after applying the shared EOS normalization."""
    if set(baseline_outputs) != set(candidate_outputs):
        return False
    return all(
        normalize_at_eos(baseline_outputs[request_id], eos_token_ids)
        == normalize_at_eos(candidate_outputs[request_id], eos_token_ids)
        for request_id in baseline_outputs
    )


def build_reference_only_record(agreement: EsmeAgreement) -> dict[str, object]:
    """Policy fields for a correctness-only row that makes no tok/s claim."""
    return {
        "policy_version": POLICY_VERSION,
        "reference_status": reference_status(agreement),
        "parity_status": "not_applicable",
        "headline_eligible": False,
    }


def build_system_evidence_record(
    *,
    agreement: EsmeAgreement,
    median_seconds: float,
    total_tokens: int,
) -> dict[str, object]:
    """Policy fields for one independently reference-checked system row.

    ``raw_tokens_per_second`` is always retained for diagnosis. ``tokens_per_second`` is the
    public field and stays ``None`` unless the row is headline eligible.
    """
    status = reference_status(agreement)
    raw_tps = total_tokens / median_seconds if median_seconds > 0 else None
    headline_eligible = (
        status in QUALIFIED_REFERENCE_STATUSES and raw_tps is not None and raw_tps > 0
    )
    return {
        "policy_version": POLICY_VERSION,
        "reference_status": status,
        "parity_status": "not_applicable",
        "headline_eligible": headline_eligible,
        "raw_tokens_per_second": raw_tps,
        "tokens_per_second": raw_tps if headline_eligible else None,
    }


__all__ = [
    "POLICY_VERSION",
    "QUALIFIED_REFERENCE_STATUSES",
    "ReferenceStatus",
    "build_reference_only_record",
    "build_system_evidence_record",
    "normalized_outputs_match",
    "reference_status",
]
