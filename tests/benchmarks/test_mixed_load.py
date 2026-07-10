"""Pure ITL-splitting and aggregation helpers for the mixed-load burst harness."""

from __future__ import annotations

import pytest

from scripts.modal_esme_prefill_ab import (
    ItlSplit,
    aggregate_mixed_load,
    percentile,
    split_itls_by_burst_step,
    summarize_run,
)


def test_percentile_interpolates_between_ranks() -> None:
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 0) == 10.0
    assert percentile(values, 100) == 40.0
    assert percentile(values, 50) == 25.0


def test_percentile_single_value_and_bounds() -> None:
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError, match="empty sample"):
        percentile([], 50)
    with pytest.raises(ValueError, match=r"in \[0, 100\]"):
        percentile([1.0], 150)


def test_split_attributes_the_straddling_gap_to_spanning() -> None:
    # One token per step at steps 5..9; the burst lands at step 8. Times are 1ms apart except
    # the burst step, whose gap (step 7 -> step 8) is inflated by the packed prefill.
    token_steps = [5, 6, 7, 8, 9]
    token_times_s = [0.000, 0.001, 0.002, 0.050, 0.051]

    split = split_itls_by_burst_step(token_steps, token_times_s, burst_step=8)

    assert split.before_ms == pytest.approx([1.0, 1.0])
    assert split.spanning_ms == pytest.approx(48.0)
    assert split.after_ms == pytest.approx([1.0])


def test_split_straddles_even_when_no_token_lands_on_burst_step() -> None:
    # A decoder that produced no token exactly on the burst step: the gap 11 -> 13 still
    # contains step 12 and must be the spanning gap.
    split = split_itls_by_burst_step(
        token_steps=[10, 11, 13, 14],
        token_times_s=[0.0, 0.001, 0.040, 0.041],
        burst_step=12,
    )

    assert split.before_ms == pytest.approx([1.0])
    assert split.spanning_ms == pytest.approx(39.0)
    assert split.after_ms == pytest.approx([1.0])


def test_split_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        split_itls_by_burst_step([1, 2], [0.0], burst_step=2)


def test_summarize_run_reports_stall_ratio_and_ttft_percentiles() -> None:
    # Two decoders, each with a clean 2ms baseline and a 20ms spanning stall.
    splits = [
        ItlSplit(before_ms=[2.0, 2.0], spanning_ms=20.0, after_ms=[2.0]),
        ItlSplit(before_ms=[2.0, 2.0], spanning_ms=20.0, after_ms=[2.0]),
    ]

    summary = summarize_run(splits, burst_ttfts_ms=[30.0, 50.0])

    assert summary["itl_spanning_burst_ms"] == pytest.approx(20.0)
    assert summary["itl_before_p50_ms"] == pytest.approx(2.0)
    assert summary["stall_ratio"] == pytest.approx(10.0)
    assert summary["burst_ttft_p50_ms"] == pytest.approx(40.0)
    assert summary["itl_after_p50_ms"] == pytest.approx(2.0)


def test_summarize_run_loud_when_no_gap_straddles_the_burst() -> None:
    splits = [ItlSplit(before_ms=[2.0], spanning_ms=None, after_ms=[2.0])]

    with pytest.raises(ValueError, match="across the burst boundary"):
        summarize_run(splits, burst_ttfts_ms=[10.0])


def test_aggregate_medians_and_candidate_ratios() -> None:
    def run(spanning: float, before: float, ttft: float) -> dict[str, float]:
        return {
            "itl_spanning_burst_ms": spanning,
            "itl_before_p50_ms": before,
            "itl_before_p95_ms": before,
            "itl_before_p99_ms": before,
            "stall_ratio": spanning / before,
            "burst_ttft_p50_ms": ttft,
            "burst_ttft_p95_ms": ttft,
        }

    runs_by_mode = {
        "baseline": [run(10.0, 2.0, 40.0), run(12.0, 2.0, 44.0)],
        "candidate": [run(20.0, 2.0, 60.0), run(22.0, 2.0, 66.0)],
    }

    aggregate = aggregate_mixed_load(runs_by_mode)

    assert aggregate["medians"]["baseline"]["itl_spanning_burst_ms"] == pytest.approx(11.0)
    assert aggregate["medians"]["candidate"]["itl_spanning_burst_ms"] == pytest.approx(21.0)
    # candidate spanning ITL is worse: 21 / 11.
    assert aggregate["candidate_vs_baseline"]["spanning_itl_ratio"] == pytest.approx(21.0 / 11.0)
    assert aggregate["candidate_vs_baseline"]["burst_ttft_p50_ratio"] == pytest.approx(63.0 / 42.0)


def test_aggregate_rejects_missing_mode() -> None:
    with pytest.raises(ValueError, match="baseline and candidate"):
        aggregate_mixed_load({"baseline": [{}]})
