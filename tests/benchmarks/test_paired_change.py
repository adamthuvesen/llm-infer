import pytest

from llm_infer.benchmarks.paired_change import paired_change


def test_consistent_speedup_has_an_interval_below_zero() -> None:
    baseline = [10.0, 10.2, 9.9, 10.1, 10.0, 10.3, 9.8]
    candidate = [value * 0.9 for value in baseline]
    change = paired_change(baseline, candidate)
    assert change.median_pct == pytest.approx(-10.0, abs=0.5)
    assert change.ci_high_pct < 0


def test_drift_shared_by_both_arms_cancels_in_the_pairing() -> None:
    # Both arms slow down together over the run; a paired comparison sees no change.
    baseline = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    change = paired_change(baseline, list(baseline))
    assert change.median_pct == 0
    assert change.ci_low_pct == 0 and change.ci_high_pct == 0


def test_mixed_signs_give_an_interval_that_spans_zero() -> None:
    baseline = [10.0] * 9
    candidate = [9.5, 10.6, 9.7, 10.4, 9.8, 10.3, 9.6, 10.5, 10.0]
    change = paired_change(baseline, candidate)
    assert change.ci_low_pct < 0 < change.ci_high_pct


def test_unpaired_rounds_are_rejected() -> None:
    with pytest.raises(ValueError, match="paired rounds"):
        paired_change([1.0, 2.0], [1.0])
