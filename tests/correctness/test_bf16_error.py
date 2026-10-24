import pytest
import torch

from llm_infer.validation.bf16_error import concat, position_errors, summarize


def _rows(*tops: tuple[float, float]) -> torch.Tensor:
    rows = torch.full((len(tops), 16), -5.0)
    for row, (first, second) in zip(rows, tops, strict=True):
        row[3], row[7] = first, second
    return rows


def test_identical_logits_have_no_error_and_no_flip() -> None:
    reference = _rows((2.0, 1.0), (0.5, 0.4))
    errors = position_errors(reference, reference.clone())
    assert errors.fp32_margin.tolist() == pytest.approx([1.0, 0.1])
    assert errors.margin_error.abs().max() == 0
    assert errors.competitor_shrink.abs().max() == 0
    assert not errors.flipped.any()
    assert errors.bf16_spacing.tolist() == [2**-6, 2**-8]


def test_competitor_gain_larger_than_the_margin_flips_the_row() -> None:
    reference = _rows((2.0, 1.95))
    candidate = reference.clone()
    candidate[0, 7] += 0.1
    errors = position_errors(reference, candidate)
    assert errors.flipped.tolist() == [True]
    assert errors.flip_gap.tolist() == pytest.approx([0.05])
    assert errors.margin_error.tolist() == pytest.approx([0.1])
    assert errors.competitor_shrink.tolist() == pytest.approx([0.1])


def test_summary_bins_flips_by_fp32_margin_and_counts_review_rows() -> None:
    reference = _rows((2.0, 1.95), (2.0, 1.8), (3.0, 1.0))
    candidate = reference.clone()
    candidate[0, 7] += 0.1  # gap 0.05: accepted tie
    candidate[1, 7] += 0.3  # gap 0.2: beyond a 0.1 tolerance
    summary = summarize(concat([position_errors(reference, candidate)]), tolerance=0.1)
    assert summary["positions"] == 3
    assert summary["flips"] == 2
    assert summary["flips_beyond_tolerance"] == 1
    assert summary["max_flip_gap"] == pytest.approx(0.2)
    assert summary["competitor_shrink_above_tolerance"] == 1
    flips_by_bin = {row["fp32_margin_low"]: row["flips"] for row in summary["by_fp32_margin"]}
    assert flips_by_bin[0.03] == 1 and flips_by_bin[0.1] == 1 and flips_by_bin[1.0] == 0


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="matching"):
        position_errors(torch.zeros(2, 4), torch.zeros(2, 5))


def test_exact_reference_tie_is_not_a_flip() -> None:
    reference = _rows((2.0, 2.0))
    candidate = reference.clone()
    candidate[0, 7] += 0.01  # the candidate picks the other tied token
    errors = position_errors(reference, candidate)
    assert errors.flipped.tolist() == [False]
    assert errors.flip_gap.tolist() == [0.0]
