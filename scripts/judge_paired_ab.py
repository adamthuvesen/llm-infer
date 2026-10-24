#!/usr/bin/env python3
"""Apply a registered paired-A/B rule to two containers' records of one experiment.

Usage: uv run scripts/judge_paired_ab.py RECORD_A.json RECORD_B.json

Rows are judged by this script, not by eye, so the verdict follows the rule registered in
docs/findings.md before the run. Each experiment's rule is keyed by the record's ``probe``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from llm_infer.benchmarks.paired_change import PairedChange, paired_change

REGRESSION_MEDIAN_PCT = 2.0
INCONCLUSIVE_HALF_WIDTH_PCT = 3.0


@dataclass(frozen=True)
class Rule:
    """One experiment's registered rule: which arms, rows, and rounds, and what may be claimed."""

    baseline: str
    candidate: str
    rows: frozenset[tuple[str, int]]
    claimed_configs: frozenset[str]
    rounds: int = 11


RULES = {
    # docs/findings.md entry 19.
    "decode-host-plan-ab": Rule(
        baseline="device",
        candidate="host",
        rows=frozenset(
            (config, batch)
            for config in ("piecewise", "grouped", "compiled")
            for batch in (1, 8, 64)
        ),
        claimed_configs=frozenset({"grouped"}),
    ),
    # docs/findings.md entry 21.
    "decode-full-step-ab": Rule(
        baseline="grouped4",
        candidate="full_step",
        rows=frozenset(("grouped", batch) for batch in (1, 8, 64)),
        claimed_configs=frozenset({"grouped"}),
    ),
    # docs/findings.md entry 22. Each prompt is one pair, so "rounds" counts prompts.
    "decode-speculative-ab": Rule(
        baseline="baseline",
        candidate="speculative",
        rows=frozenset({("open", 1), ("grounded", 1)}),
        claimed_configs=frozenset({"open", "grounded"}),
        rounds=64,
    ),
}


def _changes(record: dict, rule: Rule) -> dict[tuple[str, int], tuple[PairedChange, str]]:
    return {
        (config, row["batch_size"]): (
            paired_change(row[rule.baseline]["walls_s"], row[rule.candidate]["walls_s"]),
            row["parity_status"],
        )
        for config, rows in record["configs"].items()
        for row in rows
    }


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: uv run scripts/judge_paired_ab.py RECORD_A.json RECORD_B.json",
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
    probes = {record.get("probe") for record in records}
    if len(probes) != 1 or (probe := probes.pop()) not in RULES:
        print(f"error: both records must come from one of {sorted(RULES)}", file=sys.stderr)
        return 2
    rule = RULES[probe]
    if any(record["rounds"] != rule.rounds for record in records):
        print(f"error: the registered run has {rule.rounds} rounds per row", file=sys.stderr)
        return 2
    changes = [_changes(record, rule) for record in records]
    if any(container.keys() != rule.rows for container in changes):
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
        if config in rule.claimed_configs:
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
