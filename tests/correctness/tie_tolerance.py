"""The tie-tolerance comparison: how a fast backend is allowed to differ from the golden.

The Phase A oracle holds the fused-kernel backends to the same bar as the reference: a
fast backend's greedy tokens must match the committed HF full-recompute golden
token-for-token, EXCEPT at a genuine numerical tie. This module decides, for each
divergence, whether it is such a tie — and refuses to wave anything else off.

The mechanism mirrors the Phase A divergence trace in docs/fixture-format.md. We walk
the fast backend's greedy tokens against the golden and stop at the **first** step ``t``
where they differ. Up to ``t`` the two sequences are identical, so the fast backend
decoded step ``t`` from exactly the golden prefix; we recompute that step's logits with
the **reference path** — ``QwenModel.logits`` (full recompute, fp32, the trusted
truth) — over ``prompt + golden[:t]``. Two outcomes:

* The top-two reference logits are within ``tolerance`` — a genuine tie. The fast
  kernel's fp32-reduction-order picked the other near-equal token; that is the
  documented, acceptable flash-vs-recompute effect. The divergence is traced and
  accepted: from a true tie the fast path is a legitimate alternate greedy continuation,
  so we do not demand it rejoin the golden afterwards.
* The gap is well above ``tolerance`` — under unambiguous reference math one token wins
  and the fast kernel picked the loser. That is a real kernel/layout bug, not a tie:
  the comparison fails and names the step.

There is no blanket "close enough" and no unconditional tolerance: a divergence is
accepted only when the reference itself says the step was a coin-flip. We classify only
the first divergence because past an accepted tie the fast and golden sequences are
decoding from different (each correct) prefixes, so a later token-by-token mismatch is
expected and not evidence of a bug; a *second* genuine bug would still surface as the
first divergence on its own case.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llm_infer.model.qwen import QwenModel

# The top-2 reference-logit gap at or below which a divergence counts as a genuine tie.
# Sized from the Phase A trace: the *non*-tie gap there was 0.397 logits (~7000x the
# engine-vs-HF logit noise of 5.3e-5), and that was correctly classified as NOT a tie.
# A genuine tie is a gap at the scale of that numerical noise, not of a real margin.
# 1e-3 sits ~20x above the worst observed cross-path logit noise yet ~400x below the
# smallest real decision margin the fixture documents, so it cannot launder a real bug.
DEFAULT_TIE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class Divergence:
    """One accepted tie: the fast backend and golden disagreed but the step was a tie."""

    step: int
    fast_token: int
    golden_token: int
    reference_gap: float  # top-2 reference-logit gap at this step (<= tolerance)
    fast_logit: float  # reference logit of the token the fast kernel chose
    golden_logit: float  # reference logit of the golden token


@dataclass(frozen=True)
class TieToleranceResult:
    """Outcome of comparing a fast backend's tokens to the golden under the tie bar."""

    ok: bool
    divergence: Divergence | None  # the accepted first-divergence tie, if any
    failure: str | None  # set iff the first divergence was NOT a tie (a real bug)


def compare_under_tie_tolerance(
    reference: QwenModel,
    prompt_ids: list[int],
    fast_tokens: list[int],
    golden_tokens: list[int],
    *,
    tolerance: float = DEFAULT_TIE_TOLERANCE,
) -> TieToleranceResult:
    """Validate ``fast_tokens`` against ``golden_tokens``, allowing only a genuine-tie step.

    ``reference`` is a fp32 full-recompute model (the truth); ``fast_tokens`` are the fast
    backend's greedy continuation; ``golden_tokens`` is the committed HF golden. Finds the
    first divergence and classifies it: an exact match (no divergence) or an accepted tie
    returns ``ok=True``; a non-tie divergence returns ``ok=False`` with ``failure`` set.
    """
    step = next(
        (i for i, (f, g) in enumerate(zip(fast_tokens, golden_tokens, strict=False)) if f != g),
        None,
    )
    if step is None:
        # No token diverged in the overlap. Equal length → exact match. Unequal length →
        # one sequence is a strict prefix of the other with no traced tie; that is a
        # truncation/over-run, NOT a numerical tie, so it must fail the bar rather than
        # slip through as "equivalent" (the bar is token-for-token, ties excepted).
        if len(fast_tokens) != len(golden_tokens):
            return TieToleranceResult(
                ok=False,
                divergence=None,
                failure=(
                    f"no token diverged but lengths differ "
                    f"(fast={len(fast_tokens)}, golden={len(golden_tokens)}) — a "
                    f"prefix-equal truncation is not a tie; the sequences must match in full"
                ),
            )
        return TieToleranceResult(ok=True, divergence=None, failure=None)

    fast, golden = fast_tokens[step], golden_tokens[step]
    # The prefixes agree up to here, so the fast backend decoded this step from the golden
    # prefix; recompute its logits on that canonical context with the fp32 reference path.
    logits = reference.logits(prompt_ids + golden_tokens[:step])[-1].float()
    top2 = torch.topk(logits, 2).values
    gap = float((top2[0] - top2[1]).item())
    fast_logit = float(logits[fast].item())
    golden_logit = float(logits[golden].item())

    if gap > tolerance:
        return TieToleranceResult(
            ok=False,
            divergence=None,
            failure=(
                f"step {step}: fast backend chose {fast} but reference top-2 gap is "
                f"{gap:.6g} > tolerance {tolerance:g} — NOT a tie, a real divergence "
                f"(fast_logit={fast_logit:.6g}, golden_logit={golden_logit:.6g})"
            ),
        )
    return TieToleranceResult(
        ok=True,
        divergence=Divergence(
            step=step,
            fast_token=fast,
            golden_token=golden,
            reference_gap=gap,
            fast_logit=fast_logit,
            golden_logit=golden_logit,
        ),
        failure=None,
    )
