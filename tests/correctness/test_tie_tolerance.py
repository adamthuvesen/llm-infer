"""Tie-tolerance gate hardening (audit 2026-06-21): a prefix-equal length mismatch fails.

The bar is token-for-token vs the golden, except at a genuine numerical tie. A fast output
that is a strict *prefix* of the golden (or vice versa) has no differing token in the
overlap, so the old gate returned ``ok=True`` — silently passing a truncated/over-run
output as equivalent. These unit tests pin the hardened behavior. The fp32 reference is only
consulted when a token actually diverges, so a sentinel model that raises if touched proves
the prefix path never reaches it.
"""

from __future__ import annotations

from tests.correctness.tie_tolerance import compare_under_tie_tolerance


class _ReferenceMustNotBeUsed:
    """Stand-in fp32 reference: any consult means the no-divergence path took a wrong turn."""

    def logits(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("reference consulted though no token diverged")


def test_exact_match_passes() -> None:
    result = compare_under_tie_tolerance(_ReferenceMustNotBeUsed(), [1, 2], [5, 6, 7], [5, 6, 7])
    assert result.ok and result.divergence is None and result.failure is None


def test_fast_strict_prefix_of_golden_fails() -> None:
    result = compare_under_tie_tolerance(_ReferenceMustNotBeUsed(), [1, 2], [5, 6], [5, 6, 7])
    assert not result.ok
    assert result.divergence is None
    assert "lengths differ" in result.failure


def test_golden_strict_prefix_of_fast_fails() -> None:
    result = compare_under_tie_tolerance(_ReferenceMustNotBeUsed(), [1, 2], [5, 6, 7], [5, 6])
    assert not result.ok
    assert "lengths differ" in result.failure
