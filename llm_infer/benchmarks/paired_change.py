"""Paired relative change between two arms measured in alternating rounds, with a bootstrap CI.

Same-container A/Bs here alternate which arm runs first, so round ``i`` of the baseline and round
``i`` of the candidate saw nearly the same machine state. Pairing them by round removes most of
the drift that makes two independent medians disagree. Each pair gives one relative change,
``(candidate_i - baseline_i) / median(baseline)``. The statistic is the median of those changes,
and its 95% interval is a percentile bootstrap over the pairs.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class PairedChange:
    """Relative change of candidate against baseline, in percent; negative means faster."""

    median_pct: float
    ci_low_pct: float
    ci_high_pct: float
    pairs: int

    @property
    def ci_half_width_pct(self) -> float:
        return (self.ci_high_pct - self.ci_low_pct) / 2


def paired_change(
    baseline: list[float],
    candidate: list[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> PairedChange:
    """Median per-round relative change of ``candidate`` against ``baseline``, with a 95% CI."""
    if len(baseline) != len(candidate) or len(baseline) < 2:
        raise ValueError(
            f"need two or more paired rounds; got {len(baseline)} and {len(candidate)}"
        )
    scale = statistics.median(baseline)
    changes = [
        100.0 * (after - before) / scale for before, after in zip(baseline, candidate, strict=True)
    ]
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(changes, k=len(changes))) for _ in range(resamples)
    )
    return PairedChange(
        median_pct=statistics.median(changes),
        ci_low_pct=medians[int(0.025 * resamples)],
        ci_high_pct=medians[int(0.975 * resamples) - 1],
        pairs=len(changes),
    )
