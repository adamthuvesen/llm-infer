"""Pure-CPU parts of the prefill-divergence diagnostic: fingerprinting and stability rollup."""

from __future__ import annotations

from scripts.modal_esme_prefill_ab import fingerprint_outputs, stability_summary


def test_fingerprint_marks_identical_run_with_null_step() -> None:
    reference = {"a": [1, 2, 3], "b": [4, 5]}
    outputs = {"a": [1, 2, 3], "b": [4, 5]}

    fingerprint = fingerprint_outputs(outputs, reference)

    assert fingerprint == {
        "a": {"first_diff_step": None, "fast_token": None, "golden_token": None},
        "b": {"first_diff_step": None, "fast_token": None, "golden_token": None},
    }


def test_fingerprint_records_first_divergence_index_and_tokens() -> None:
    reference = {"a": [1, 2, 3, 4]}
    outputs = {"a": [1, 2, 9, 4]}

    fingerprint = fingerprint_outputs(outputs, reference)

    assert fingerprint["a"] == {"first_diff_step": 2, "fast_token": 9, "golden_token": 3}


def test_fingerprint_keys_on_reference_and_ignores_extra_outputs() -> None:
    reference = {"a": [1, 2]}
    outputs = {"a": [1, 2], "unexpected": [7, 7]}

    fingerprint = fingerprint_outputs(outputs, reference)

    assert set(fingerprint) == {"a"}


def _record(
    request_id: str, mode: str, attempt: int, step: int | None, fast: int | None, golden: int | None
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "mode": mode,
        "attempt": attempt,
        "first_diff_step": step,
        "first_diff_tokens": {"fast": fast, "golden": golden},
    }


def test_stability_summary_flags_repeated_divergence_as_stable() -> None:
    records = [_record("a", "candidate", attempt, 5, 9, 3) for attempt in range(3)]

    summary = stability_summary(records)

    assert summary["a"]["candidate"] == {
        "stable": True,
        "attempts": 3,
        "first_diff_steps": [5],
        "first_diff_tokens": [{"fast": 9, "golden": 3}],
    }


def test_stability_summary_flags_varying_step_as_unstable() -> None:
    records = [
        _record("a", "candidate", 0, 5, 9, 3),
        _record("a", "candidate", 1, 7, 8, 3),
    ]

    summary = stability_summary(records)

    assert summary["a"]["candidate"]["stable"] is False
    assert summary["a"]["candidate"]["first_diff_steps"] == [5, 7]


def test_stability_summary_groups_by_mode_and_sorts_none_last() -> None:
    records = [
        _record("a", "baseline", 0, None, None, None),
        _record("a", "baseline", 1, 4, 1, 2),
        _record("a", "candidate", 0, None, None, None),
        _record("a", "candidate", 1, None, None, None),
    ]

    summary = stability_summary(records)

    # baseline saw a divergence on one attempt and none on the other: unstable, None sorts last.
    assert summary["a"]["baseline"]["stable"] is False
    assert summary["a"]["baseline"]["first_diff_steps"] == [4, None]
    # candidate never diverged across attempts: stable at "no divergence".
    assert summary["a"]["candidate"]["stable"] is True
    assert summary["a"]["candidate"]["first_diff_steps"] == [None]
