"""Per-position logit error of a candidate path against the fp32 reference, on one context.

The 0.1-logit bf16 rule accepts a greedy divergence when the token the fast path chose sits
within 0.1 of the fp32 maximum. That rule is safe only if bf16 noise rarely moves one logit
by more than 0.1 relative to another. This module measures that movement directly: both
logit rows come from the same token context, so every difference is numerical error, not a
different continuation.

For each position, with ``a`` the fp32 top token and ``err = candidate - reference``:

* ``fp32_margin``: ``f[a] - f[b]`` for the fp32 runner-up ``b``.
* ``margin_error``: ``|err[a] - err[b]|``, the error in that top-two margin.
* ``competitor_shrink``: the largest ``err[c] - err[a]`` over the fp32 top competitors ``c``.
  The candidate flips to ``c`` when this exceeds ``f[a] - f[c]``, so this is the quantity a
  tie tolerance must bound.
* ``flipped`` and ``flip_gap``: whether the candidate's argmax differs from ``a``, and the fp32
  gap ``f[a] - f[candidate argmax]`` when it does. A flip with a gap above the tolerance is a
  ``review_required`` row under policy v2.
* ``max_abs_error``: ``max |err|`` over the vocabulary.
* ``bf16_spacing``: the gap between adjacent bf16 values at the fp32 top logit. Logits stored
  in bf16 cannot resolve a margin finer than this, so it sets a floor under any tie tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TOP_COMPETITORS = 8
MARGIN_BIN_EDGES = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, float("inf"))
QUANTILES = (0.5, 0.9, 0.99, 0.999)


@dataclass(frozen=True)
class PositionErrors:
    """One entry per measured position; every tensor is 1-D, float32 or bool, on the CPU."""

    fp32_margin: torch.Tensor
    margin_error: torch.Tensor
    competitor_shrink: torch.Tensor
    flipped: torch.Tensor
    flip_gap: torch.Tensor
    max_abs_error: torch.Tensor
    bf16_spacing: torch.Tensor


def position_errors(reference: torch.Tensor, candidate: torch.Tensor) -> PositionErrors:
    """Compare ``(positions, vocab)`` candidate logits to fp32 reference logits row by row."""
    if reference.shape != candidate.shape or reference.dim() != 2:
        raise ValueError(
            f"expected matching (positions, vocab) logits; got {tuple(reference.shape)} "
            f"and {tuple(candidate.shape)}"
        )
    reference = reference.float()
    error = candidate.float() - reference
    top_values, top_ids = reference.topk(TOP_COMPETITORS + 1, dim=-1)
    top_error = error.gather(-1, top_ids)
    winner_error = top_error[:, :1]
    candidate_top = candidate.float().argmax(dim=-1)
    flipped = candidate_top != top_ids[:, 0]
    flip_gap = top_values[:, 0] - reference.gather(-1, candidate_top[:, None]).squeeze(-1)
    return PositionErrors(
        fp32_margin=(top_values[:, 0] - top_values[:, 1]).cpu(),
        margin_error=(winner_error[:, 0] - top_error[:, 1]).abs().cpu(),
        competitor_shrink=(top_error[:, 1:] - winner_error).amax(dim=-1).cpu(),
        flipped=flipped.cpu(),
        flip_gap=torch.where(flipped, flip_gap, torch.zeros_like(flip_gap)).cpu(),
        max_abs_error=error.abs().amax(dim=-1).cpu(),
        bf16_spacing=_bf16_spacing(top_values[:, 0]).cpu(),
    )


def _bf16_spacing(values: torch.Tensor) -> torch.Tensor:
    # bf16 keeps 8 significant bits, so values in [2^e, 2^(e+1)) are 2^(e-7) apart.
    exponent = torch.floor(torch.log2(values.abs().clamp_min(torch.finfo(torch.float32).tiny)))
    return torch.exp2(exponent - 7)


def concat(parts: list[PositionErrors]) -> PositionErrors:
    return PositionErrors(
        *(
            torch.cat([getattr(part, name) for part in parts])
            for name in PositionErrors.__dataclass_fields__
        )
    )


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    levels = torch.tensor(QUANTILES, dtype=torch.float64)
    found = torch.quantile(values.double(), levels).tolist()
    summary = {f"p{level * 100:g}": value for level, value in zip(QUANTILES, found, strict=True)}
    summary["max"] = float(values.max())
    return summary


def summarize(errors: PositionErrors, *, tolerance: float) -> dict[str, object]:
    """Distribution summary, flip counts by fp32 margin, and exceedance of ``tolerance``."""
    positions = int(errors.fp32_margin.numel())
    if positions == 0:
        raise ValueError("no positions to summarize")
    bins = []
    for low, high in zip(MARGIN_BIN_EDGES, MARGIN_BIN_EDGES[1:], strict=False):
        in_bin = (errors.fp32_margin >= low) & (errors.fp32_margin < high)
        bins.append(
            {
                "fp32_margin_low": low,
                "fp32_margin_high": high,
                "positions": int(in_bin.sum()),
                "flips": int((in_bin & errors.flipped).sum()),
            }
        )
    flips = int(errors.flipped.sum())
    return {
        "positions": positions,
        "tolerance": tolerance,
        "margin_error": _quantiles(errors.margin_error),
        "competitor_shrink": _quantiles(errors.competitor_shrink),
        "max_abs_error": _quantiles(errors.max_abs_error),
        "bf16_spacing_at_top_logit": _quantiles(errors.bf16_spacing),
        "competitor_shrink_above_tolerance": int((errors.competitor_shrink > tolerance).sum()),
        "flips": flips,
        "flips_beyond_tolerance": int((errors.flip_gap > tolerance).sum()),
        "max_flip_gap": float(errors.flip_gap.max()) if flips else 0.0,
        "by_fp32_margin": bins,
    }
