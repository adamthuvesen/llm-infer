"""Policy-v2 reference gates for Esme benchmark evidence."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

from llm_infer.benchmarks.esme_paged import EsmeBenchRequest
from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
from llm_infer.benchmarks.reference_policy import (
    EsmeAgreement,
    ReferenceEvidence,
    SystemEvidence,
    build_reference_only_record,
    build_system_evidence_record,
    normalized_outputs_match,
    reference_status,
)


def _agreement(
    *,
    exact: int = 1,
    tie: int = 0,
    review_required: int = 0,
    failed: int = 0,
) -> EsmeAgreement:
    review_rows = [
        {
            "request": "r0",
            "step": 0,
            "fast_token": 2,
            "golden_token": 1,
            "fp32_margin": 0.2,
            "automatic_boundary": 0.1,
        }
        for _ in range(review_required)
    ]
    return EsmeAgreement(
        exact=exact,
        tie=tie,
        nontie=review_required + failed,
        total=exact + tie + review_required + failed,
        ties_sample=[],
        divergences_sample=review_rows,
        review_required=review_required,
        failed=failed,
        numerical_evidence=review_rows,
    )


@pytest.mark.parametrize(
    ("agreement", "expected"),
    [
        (_agreement(), "exact"),
        (_agreement(tie=1), "accepted_numerical"),
        (_agreement(exact=0, review_required=1), "review_required"),
        (_agreement(exact=0, failed=1), "failed"),
        (_agreement(exact=7, tie=0, review_required=1, failed=1), "failed"),
    ],
)
def test_reference_status_classification(agreement: EsmeAgreement, expected: str) -> None:
    assert reference_status(agreement) == expected


def test_reference_status_rejects_an_empty_agreement() -> None:
    with pytest.raises(ValueError, match="no requests"):
        EsmeAgreement(exact=0, tie=0, nontie=0, total=0, ties_sample=[], divergences_sample=[])


def test_reference_status_rejects_counts_that_do_not_add_up() -> None:
    with pytest.raises(ValueError, match="must equal total") as error:
        EsmeAgreement(exact=1, tie=0, nontie=0, total=2, ties_sample=[], divergences_sample=[])
    assert "exact=1, tie=0, nontie=0, total=2" in str(error.value)
    assert "Classify every measured request once" in str(error.value)
    assert "build_system_evidence_record()" in str(error.value)


def test_reference_status_rejects_an_unclassified_nontie() -> None:
    with pytest.raises(ValueError, match="review_required \\+ failed") as error:
        EsmeAgreement(exact=0, tie=0, nontie=1, total=1, ties_sample=[], divergences_sample=[])
    assert "nontie=1, review_required=0, failed=0" in str(error.value)
    assert "Classify every non-tie as review_required or failed" in str(error.value)


def test_normalized_outputs_match_truncates_at_eos() -> None:
    eos = frozenset({99})
    assert normalized_outputs_match({"r0": [1, 99, 7]}, {"r0": [1, 99]}, eos)
    assert not normalized_outputs_match({"r0": [1, 99]}, {"r0": [2, 99]}, eos)
    assert not normalized_outputs_match({"r0": [1, 99]}, {"r0": [1, 99], "extra": [2]}, eos)


def test_system_record_keeps_raw_tps_while_review_is_pending() -> None:
    record = build_system_evidence_record(
        agreement=_agreement(exact=0, review_required=1),
        median_seconds=0.5,
        total_tokens=10,
    )

    assert record == {
        "policy_version": 2,
        "reference_status": "review_required",
        "parity_status": "not_applicable",
        "headline_eligible": False,
        "raw_tokens_per_second": 20.0,
        "tokens_per_second": None,
    }


def test_system_record_reports_public_tps_only_when_qualified() -> None:
    record = build_system_evidence_record(
        agreement=_agreement(tie=1), median_seconds=0.5, total_tokens=10
    )

    assert record["reference_status"] == "accepted_numerical"
    assert record["headline_eligible"] is True
    assert record["tokens_per_second"] == 20.0


def test_system_record_does_not_promote_a_zero_token_measurement() -> None:
    record = build_system_evidence_record(
        agreement=_agreement(),
        median_seconds=0.5,
        total_tokens=0,
    )

    assert record["reference_status"] == "exact"
    assert record["raw_tokens_per_second"] == 0.0
    assert record["headline_eligible"] is False
    assert record["tokens_per_second"] is None


def test_reference_only_record_never_claims_speed() -> None:
    record = build_reference_only_record(_agreement())

    assert record == {
        "policy_version": 2,
        "reference_status": "exact",
        "parity_status": "not_applicable",
        "headline_eligible": False,
    }


class _FixedOracle:
    def __init__(self, logits: list[float]) -> None:
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def logits(self, token_ids: list[int]) -> torch.Tensor:
        return self._logits


@pytest.mark.parametrize(
    ("margin", "expected"),
    [(0.05, "accepted_numerical"), (0.2, "review_required")],
)
def test_esme_classifier_uses_point_one_as_automatic_boundary(margin: float, expected: str) -> None:
    agreement = tie_tolerant_agreement(
        _FixedOracle([1.0, 1.0 - margin]),
        [EsmeBenchRequest("r0", "prompt", (3,))],
        {"r0": [1]},
        {"r0": [0]},
        frozenset(),
    )

    assert reference_status(agreement) == expected
    if expected == "review_required":
        assert agreement.review_required == 1
        assert agreement.failed == 0


def test_evidence_derives_claims_and_rejects_independent_policy_fields() -> None:
    evidence = SystemEvidence.from_timing(
        agreement=_agreement(exact=0, failed=1), median_seconds=0.5, total_tokens=10
    )
    assert evidence.reference_status == "failed"
    assert evidence.raw_tokens_per_second == 20.0
    assert evidence.headline_eligible is False
    assert evidence.tokens_per_second is None
    with pytest.raises(TypeError, match="headline_eligible"):
        SystemEvidence(agreement=_agreement(), raw_tokens_per_second=20.0, headline_eligible=True)
    with pytest.raises(FrozenInstanceError):
        evidence.agreement.exact = 1
    with pytest.raises(FrozenInstanceError):
        evidence.raw_tokens_per_second = 100.0
    with pytest.raises(FrozenInstanceError):
        evidence.headline_eligible = True


def test_agreement_diagnostics_cannot_change_stored_evidence() -> None:
    nested = {"tokens": [1, 2]}
    samples = [{"detail": nested}]
    agreement = EsmeAgreement(0, 1, 0, 1, samples, [], numerical_evidence=samples)
    evidence = SystemEvidence(agreement, 20.0)
    before = agreement.as_dict()
    nested["tokens"].append(3)
    samples.clear()
    agreement.ties_sample[0]["detail"] = "changed"
    serialized = agreement.as_dict()
    serialized["numerical_evidence"].clear()
    assert agreement.as_dict() == before
    assert evidence.reference_status == "accepted_numerical"
    assert evidence.tokens_per_second == 20.0
    assert ReferenceEvidence(agreement).as_dict()["headline_eligible"] is False


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), -1.0, True])
def test_system_evidence_rejects_invalid_measured_rates(rate: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        SystemEvidence(_agreement(), rate)
