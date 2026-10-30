"""Policy-v2 evidence contract for Esme benchmark records.

Raw timing is always kept. Public tok/s is a separate claim: a row reports
``tokens_per_second`` only when its reference status qualifies (``headline_eligible``).
``review_required`` marks a divergence above the automatic tie boundary that is waiting on a
durable review — it withholds the public number, it does not mean the row is broken.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, TypedDict

from llm_infer.benchmarks.report import normalize_at_eos

POLICY_VERSION = 2
ReferenceStatus = Literal["exact", "accepted_numerical", "review_required", "failed"]
ParityStatus = Literal["exact", "accepted_numerical", "review_required", "failed", "not_applicable"]

QUALIFIED_REFERENCE_STATUSES = frozenset({"exact", "accepted_numerical"})

# docs/findings.md entry 20 audits this automatic bf16 acceptance boundary.
BF16_AGREEMENT_TOLERANCE = 0.1


@dataclass(frozen=True)
class EsmeAgreement:
    """One system's agreement with the fp32 reference, independent of model execution."""

    exact: int
    tie: int
    nontie: int
    total: int
    ties_sample: list[dict[str, object]]
    divergences_sample: list[dict[str, object]]
    review_required: int = 0
    failed: int = 0
    numerical_evidence: list[dict[str, object]] = field(default_factory=list)

    @property
    def all_ties_or_exact(self) -> bool:
        return self.nontie == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "exact": self.exact,
            "tie": self.tie,
            "nontie": self.nontie,
            "total": self.total,
            "ties_sample": self.ties_sample,
            "divergences_sample": self.divergences_sample,
            "review_required": self.review_required,
            "failed": self.failed,
            "numerical_evidence": self.numerical_evidence,
        }

    @classmethod
    def from_record(cls, value: object) -> EsmeAgreement:
        row = record_object(value, "agreement")

        def count(name: str) -> int:
            value = row.get(name)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"agreement.{name}={value!r}, expected a non-negative integer count; "
                    "reclassify measured outputs before building evidence"
                )
            return value

        def samples(name: str) -> list[dict[str, object]]:
            value = row.get(name, [])
            if not isinstance(value, list):
                raise ValueError(f"agreement.{name} must be a list")
            return [record_object(item, f"agreement.{name}") for item in value]

        agreement = cls(
            exact=count("exact"),
            tie=count("tie"),
            nontie=count("nontie"),
            total=count("total"),
            ties_sample=samples("ties_sample"),
            divergences_sample=samples("divergences_sample"),
            review_required=count("review_required"),
            failed=count("failed"),
            numerical_evidence=samples("numerical_evidence"),
        )
        reference_status(agreement)
        return agreement


class ReferenceEvidence(TypedDict):
    policy_version: int
    reference_status: ReferenceStatus
    parity_status: ParityStatus
    headline_eligible: bool


class SystemEvidence(ReferenceEvidence):
    raw_tokens_per_second: float | None
    tokens_per_second: float | None


def record_object(value: object, context: str) -> dict[str, object]:
    """Validate a JSON object before reading persisted evidence fields."""
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object with string keys")
    return {key: item for key, item in value.items()}


def benchmark_rows(value: object) -> list[dict[str, object]]:
    record = record_object(value, "benchmark record")
    rows = record.get("rows")
    if not isinstance(rows, list):
        raise ValueError("benchmark record must contain a rows list")
    return [record_object(row, "benchmark row") for row in rows]


def headline_tokens_per_second(row: Mapping[str, object]) -> float:
    """Read a public speed claim through the same policy used to construct it."""
    agreement = EsmeAgreement.from_record(row.get("agreement"))
    expected_status = reference_status(agreement)
    raw = row.get("raw_tokens_per_second")
    public = row.get("tokens_per_second")
    problems: list[str] = []
    if row.get("policy_version") != POLICY_VERSION:
        problems.append(f"policy_version={row.get('policy_version')!r}, expected {POLICY_VERSION}")
    if row.get("reference_status") != expected_status:
        problems.append(
            f"reference_status={row.get('reference_status')!r}, "
            f"expected {expected_status!r} from agreement counts"
        )
    if expected_status not in QUALIFIED_REFERENCE_STATUSES:
        problems.append(
            f"agreement reference_status={expected_status!r}, expected 'exact' or "
            "'accepted_numerical' before publishing speed; resolve the reference failure or review"
        )
    if row.get("headline_eligible") is not True:
        problems.append(f"headline_eligible={row.get('headline_eligible')!r}, expected True")
    if (
        not isinstance(raw, (float, int))
        or isinstance(raw, bool)
        or not math.isfinite(raw)
        or raw <= 0
    ):
        problems.append(f"raw_tokens_per_second={raw!r}, expected a finite positive measurement")
    if not isinstance(public, (float, int)) or isinstance(public, bool) or public != raw:
        problems.append(
            f"tokens_per_second={public!r}, expected raw_tokens_per_second={raw!r} "
            "only after reference qualification"
        )
    if problems:
        raise ValueError(
            "row is not a reference-gated, headline-eligible policy-v2 speed claim: "
            + "; ".join(problems)
            + ". Rebuild policy fields with build_system_evidence_record() in "
            "llm_infer/benchmarks/reference_policy.py. Keep tokens_per_second=None "
            "until the reference evidence qualifies."
        )
    assert isinstance(public, (float, int))
    return float(public)


def reference_status(agreement: EsmeAgreement) -> ReferenceStatus:
    """Classify one system's fp32 agreement profile.

    Every request must be classified: exact, tie, or non-tie, with each non-tie counted as
    either ``review_required`` or ``failed``. An agreement whose counts do not add up is a
    construction bug, not a benchmark result, so it raises instead of guessing.
    """
    if agreement.total < 1:
        raise ValueError(
            f"agreement covers no requests: total={agreement.total}, expected total >= 1; "
            "classify measured outputs before building evidence"
        )
    for field_name, count in (
        ("exact", agreement.exact),
        ("tie", agreement.tie),
        ("nontie", agreement.nontie),
        ("review_required", agreement.review_required),
        ("failed", agreement.failed),
    ):
        if count < 0:
            raise ValueError(
                f"agreement.{field_name}={count}, expected a non-negative count; "
                "reclassify measured outputs before building evidence"
            )
    if agreement.exact + agreement.tie + agreement.nontie != agreement.total:
        raise ValueError(
            "inconsistent agreement: exact + tie + nontie must equal total "
            f"(exact={agreement.exact}, tie={agreement.tie}, "
            f"nontie={agreement.nontie}, total={agreement.total}). "
            "Classify every measured request once before calling build_system_evidence_record()."
        )
    if agreement.nontie != agreement.review_required + agreement.failed:
        raise ValueError(
            "inconsistent agreement: nontie must equal review_required + failed "
            f"(nontie={agreement.nontie}, review_required={agreement.review_required}, "
            f"failed={agreement.failed}). Classify every non-tie as review_required or failed "
            "before calling build_system_evidence_record()."
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


def build_reference_only_record(agreement: EsmeAgreement) -> ReferenceEvidence:
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
) -> SystemEvidence:
    """Policy fields for one independently reference-checked system row.

    ``raw_tokens_per_second`` is always retained for diagnosis. ``tokens_per_second`` is the
    public field and stays ``None`` unless the row is headline eligible.
    """
    status = reference_status(agreement)
    if not math.isfinite(median_seconds) or median_seconds < 0 or total_tokens < 0:
        raise ValueError("timing and token count must be finite and non-negative")
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
    "EsmeAgreement",
    "BF16_AGREEMENT_TOLERANCE",
    "headline_tokens_per_second",
    "benchmark_rows",
    "build_reference_only_record",
    "build_system_evidence_record",
    "normalized_outputs_match",
    "reference_status",
]
