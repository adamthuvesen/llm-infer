#!/usr/bin/env python3
"""Apply entry 19's registered confirmatory rule to two host-plan-ab records.

Usage: uv run scripts/judge_host_plan_ab.py RECORD_A.json RECORD_B.json

Rows are judged by this script, not by eye, so the verdict follows the rule registered in
docs/findings.md entry 19 before the run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from llm_infer.benchmarks.paired_change import PairedChange, paired_change

REGRESSION_MEDIAN_PCT = 2.0
INCONCLUSIVE_HALF_WIDTH_PCT = 3.0
CLAIMED_CONFIG = "grouped"
REGISTERED_ROUNDS = 11
REGISTERED_ROWS = frozenset(
    (config, batch) for config in ("piecewise", "grouped", "compiled") for batch in (1, 8, 64)
)


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
    raw = [Path(path).read_bytes() for path in sys.argv[1:]]
    records = [json.loads(content) for content in raw]
    container_ids = [record.get("container_id") for record in records]
    same_container = None not in container_ids and container_ids[0] == container_ids[1]
    if hashlib.sha256(raw[0]).digest() == hashlib.sha256(raw[1]).digest() or same_container:
        print("error: the rule needs records from two different containers", file=sys.stderr)
        return 2
    if any(record["rounds"] != REGISTERED_ROUNDS for record in records):
        print(f"error: the registered run has {REGISTERED_ROUNDS} rounds per row", file=sys.stderr)
        return 2
    changes = [_changes(record) for record in records]
    if any(container.keys() != REGISTERED_ROWS for container in changes):
        print("error: a record does not cover exactly the registered rows", file=sys.stderr)
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
            # An inconclusive row is only labeled: under the rule it cannot fail the gate unless
            # it also regresses, and it cannot support a claim (the claim below needs CI < 0).
            gate_passes &= parity == "exact" and not regresses
            cells.append(
                f"{change.median_pct:+.1f}% [{change.ci_low_pct:+.1f}, {change.ci_high_pct:+.1f}] "
                f"{status}, parity {parity}"
            )
        if config == CLAIMED_CONFIG:
            first, second = (container[key][0] for container in changes)
            both_exact = all(container[key][1] == "exact" for container in changes)
            if not both_exact:
                claims[f"{config} b{batch}"] = "not confirmed (parity not exact)"
            elif first.ci_high_pct < 0 and second.ci_high_pct < 0:
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
    return 0 if gate_passes else 1


if __name__ == "__main__":
    sys.exit(main())
