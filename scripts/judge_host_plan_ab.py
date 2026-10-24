#!/usr/bin/env python3
"""Apply entry 19's registered confirmatory rule to two host-plan-ab records.

Usage: uv run scripts/judge_host_plan_ab.py RECORD_A.json RECORD_B.json

Rows are judged by this script, not by eye, so the verdict follows the rule registered in
docs/findings.md entry 19 before the run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from llm_infer.benchmarks.paired_change import PairedChange, paired_change

REGRESSION_MEDIAN_PCT = 2.0
INCONCLUSIVE_HALF_WIDTH_PCT = 3.0
CLAIMED_CONFIG = "grouped"


def _changes(record: dict) -> dict[tuple[str, int], tuple[PairedChange, str]]:
    return {
        (config, row["batch_size"]): (
            paired_change(row["device"]["walls_s"], row["host"]["walls_s"]),
            row["parity_status"],
        )
        for config, rows in record["configs"].items()
        for row in rows
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: uv run scripts/judge_host_plan_ab.py RECORD_A.json RECORD_B.json",
            file=sys.stderr,
        )
        return 2
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:]]
    changes = [_changes(record) for record in records]
    if changes[0].keys() != changes[1].keys():
        print("error: the two records cover different rows", file=sys.stderr)
        return 2

    gate_passes = True
    claims: dict[str, str] = {}
    for key in changes[0]:
        config, batch = key
        cells = []
        for container in changes:
            change, parity = container[key]
            regresses = change.ci_low_pct > 0 and change.median_pct > REGRESSION_MEDIAN_PCT
            status = (
                "regresses"
                if regresses
                else "inconclusive"
                if change.ci_half_width_pct > INCONCLUSIVE_HALF_WIDTH_PCT
                else "ok"
            )
            gate_passes &= parity == "exact" and not regresses
            cells.append(
                f"{change.median_pct:+.1f}% [{change.ci_low_pct:+.1f}, {change.ci_high_pct:+.1f}] "
                f"{status}, parity {parity}"
            )
        if config == CLAIMED_CONFIG:
            first, second = (container[key][0] for container in changes)
            if first.ci_high_pct < 0 and second.ci_high_pct < 0:
                smaller = min(first.median_pct, second.median_pct, key=abs)
                claims[f"{config} b{batch}"] = f"qualified gain {smaller:+.1f}%"
            elif (first.median_pct < 0) != (second.median_pct < 0):
                claims[f"{config} b{batch}"] = "not confirmed (containers disagree on sign)"
            else:
                claims[f"{config} b{batch}"] = "not confirmed (interval reaches 0)"
        print(f"{config:9} b{batch:<3} | " + " | ".join(cells))
    print()
    for row, claim in claims.items():
        print(f"{row}: {claim}")
    print(f"\nmerge gate: {'PASS' if gate_passes else 'FAIL'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
