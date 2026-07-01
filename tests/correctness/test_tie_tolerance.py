"""Tie-tolerance invariant: a prefix-equal length mismatch fails.

The rule is token-for-token vs the golden, except at a genuine numerical tie. A fast output
that is a strict *prefix* of the golden (or vice versa) has no differing token in the
overlap; the check must still fail it rather than pass a truncated/over-run output as
equivalent. These unit tests pin that behavior. The fp32 reference is only consulted when a
token actually diverges, so a sentinel model that raises if touched proves the prefix path
never reaches it.
"""

from __future__ import annotations

import torch

from llm_infer.validation.tie_tolerance import compare_under_tie_tolerance


class _ReferenceMustNotBeUsed:
    """Stand-in fp32 reference: any consult means the no-divergence path took a wrong turn."""

    def logits(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("reference consulted though no token diverged")


class _FixedReference:
    """A reference whose next-token logits are a fixed row, for classifying one divergence."""

    def __init__(self, row: list[float]) -> None:
        self._row = row

    def logits(self, _ids: list[int]) -> torch.Tensor:
        return torch.tensor([self._row], dtype=torch.float32)  # (1, vocab); caller takes [-1]


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


def test_small_top2_gap_does_not_excuse_a_far_off_fast_token() -> None:
    """The hardened rule: a near-tie between the top two does NOT bless a third, far-down token.

    Regression for a check that accepted any divergence whenever the reference top-2 gap was
    within tolerance, without checking the token the fast kernel actually chose. Here tokens 0
    and 1 are tied (gap 1e-4), but the fast kernel picked token 5, a full 5 logits below the
    winner — a real bug, not a tie.
    """
    # vocab row: token0=10.0 (max), token1=9.9999 (tied), token5=5.0 (far below).
    ref = _FixedReference([10.0, 9.9999, 0.0, 0.0, 0.0, 5.0])
    result = compare_under_tie_tolerance(ref, [1], fast_tokens=[5], golden_tokens=[0])
    assert not result.ok
    assert result.divergence is None
    assert "below the reference max" in result.failure


def test_genuine_runner_up_tie_is_accepted() -> None:
    """The fast kernel choosing the genuinely near-tied runner-up is still an accepted tie."""
    ref = _FixedReference([10.0, 9.9999, 0.0, 0.0, 0.0, 5.0])
    result = compare_under_tie_tolerance(ref, [1], fast_tokens=[1], golden_tokens=[0])
    assert result.ok
    assert result.divergence is not None
    assert result.divergence.fast_token == 1 and result.divergence.golden_token == 0
