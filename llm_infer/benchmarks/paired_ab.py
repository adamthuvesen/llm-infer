"""Registered paired-A/B rules and verdicts, independent of GPU runtimes."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from llm_infer.benchmarks.paired_change import PairedChange, paired_change
from llm_infer.benchmarks.reference_policy import EsmeAgreement, record_object

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
    # The arms resolve bf16 ties differently by design, so direct parity cannot be exact. The
    # gate then needs zero non-tie divergences from fp32 in both arms, and a gain is raw timing.
    raw_timing: bool = False


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
    # docs/findings.md entry 23.
    "decode-fusion-ab": Rule(
        baseline="full_step",
        candidate="full_step_fused",
        rows=frozenset(("grouped", batch) for batch in (1, 8, 64)),
        claimed_configs=frozenset({"grouped"}),
        raw_timing=True,
    ),
}


@dataclass(frozen=True)
class ArmEvidence:
    walls_s: list[float]
    agreement: EsmeAgreement


@dataclass(frozen=True)
class PairedRow:
    baseline: ArmEvidence
    candidate: ArmEvidence
    parity_status: str


@dataclass(frozen=True)
class PairedRecord:
    probe: str
    container_id: str | None
    rows: dict[tuple[str, int], PairedRow]


@dataclass(frozen=True)
class Judgement:
    lines: tuple[str, ...]
    gate_passes: bool

    @property
    def exit_code(self) -> int:
        return 0 if self.gate_passes else 1


def read_paired_record(value: object) -> PairedRecord:
    record = record_object(value, "paired record")
    probe = record.get("probe")
    if not isinstance(probe, str) or probe not in RULES:
        raise ValueError(f"probe={probe!r}, expected one of {sorted(RULES)}")
    rule = RULES[probe]
    if record.get("rounds") != rule.rounds:
        raise ValueError(
            f"rounds={record.get('rounds')!r}, expected {rule.rounds} "
            f"for registered probe {probe!r}; "
            "supply the complete measured record for this experiment"
        )
    container_id = record.get("container_id")
    if container_id is not None and not isinstance(container_id, str):
        raise ValueError("container_id must be a string")
    for field, expected in (("baseline_arm", rule.baseline), ("candidate_arm", rule.candidate)):
        if field in record and record[field] != expected:
            raise ValueError(f"{field} must be {expected}")
    configs = record_object(record.get("configs"), "configs")
    rows: dict[tuple[str, int], PairedRow] = {}
    for config, values in configs.items():
        if not isinstance(values, list):
            raise ValueError(f"configs.{config} must be a list")
        for value in values:
            row = record_object(value, f"configs.{config} row")
            batch = row.get("batch_size")
            if type(batch) is not int or batch < 1:
                raise ValueError("batch_size must be a positive integer")
            key = (config, batch)
            if key in rows:
                raise ValueError(f"duplicate registered row {key}")
            arms: list[ArmEvidence] = []
            for name in (rule.baseline, rule.candidate):
                arm = record_object(row.get(name), f"{key} {name}")
                walls = arm.get("walls_s")
                if not isinstance(walls, list) or len(walls) != rule.rounds:
                    raise ValueError(f"{key} {name}: expected {rule.rounds} measured walls_s")
                seconds: list[float] = []
                for wall in walls:
                    if (
                        not isinstance(wall, (float, int))
                        or isinstance(wall, bool)
                        or not math.isfinite(wall)
                        or wall <= 0
                    ):
                        raise ValueError(f"{key} {name}: walls_s must be finite and positive")
                    seconds.append(float(wall))
                try:
                    agreement = EsmeAgreement.from_record(arm.get("agreement"))
                except ValueError as error:
                    raise ValueError(f"{key} {name}: {error}") from error
                arms.append(ArmEvidence(seconds, agreement))
            parity = row.get("parity_status")
            if parity not in ("exact", "accepted_numerical", "review_required", "failed"):
                raise ValueError(f"{key}: invalid parity_status {parity!r}")
            assert isinstance(parity, str)
            rows[key] = PairedRow(arms[0], arms[1], parity)
    if rows.keys() != rule.rows:
        raise ValueError("a record does not cover exactly the registered rows")
    return PairedRecord(probe, container_id, rows)


def _parity(row: PairedRow, rule: Rule) -> str:
    if not rule.raw_timing:
        return row.parity_status
    if any(arm.agreement.nontie != 0 for arm in (row.baseline, row.candidate)):
        return "nontie divergence"
    return "exact" if row.parity_status == "exact" else "nontie 0"


def judge_records(record_a: PairedRecord, record_b: PairedRecord) -> Judgement:
    if record_a.probe != record_b.probe:
        raise ValueError("both records must come from the same registered probe")
    if record_a.container_id is not None and record_a.container_id == record_b.container_id:
        raise ValueError("the rule needs records from two different containers")
    rule = RULES[record_a.probe]
    changes: list[dict[tuple[str, int], tuple[PairedChange, str]]] = [
        {
            key: (paired_change(row.baseline.walls_s, row.candidate.walls_s), _parity(row, rule))
            for key, row in record.rows.items()
        }
        for record in (record_a, record_b)
    ]
    gate_passes = True
    claims: dict[str, str] = {}
    lines: list[str] = []
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
            gate_passes &= parity in ({"exact", "nontie 0"} if rule.raw_timing else {"exact"})
            gate_passes &= not regresses
            cells.append(
                f"{change.median_pct:+.1f}% [{change.ci_low_pct:+.1f}, {change.ci_high_pct:+.1f}] "
                f"{status}, parity {parity}"
            )
        if config in rule.claimed_configs:
            first, second = (container[key][0] for container in changes)
            both_exact = all(container[key][1] == "exact" for container in changes)
            if any(container[key][1] == "nontie divergence" for container in changes):
                claims[f"{config} b{batch}"] = "not confirmed (non-tie divergence from fp32)"
            elif not both_exact and not rule.raw_timing:
                claims[f"{config} b{batch}"] = "not confirmed (parity not exact)"
            elif first.ci_high_pct < 0 and second.ci_high_pct < 0:
                smaller = min(first.median_pct, second.median_pct, key=abs)
                kind = "qualified gain" if both_exact else "raw gain"
                claims[f"{config} b{batch}"] = f"{kind} {smaller:+.1f}%"
            elif (first.median_pct < 0) != (second.median_pct < 0):
                claims[f"{config} b{batch}"] = "not confirmed (containers disagree on sign)"
            else:
                claims[f"{config} b{batch}"] = "not confirmed (interval reaches 0)"
        lines.append(f"{config:9} b{batch:<3} | " + " | ".join(cells))
    lines.append("")
    for row, claim in claims.items():
        lines.append(f"{row}: {claim}")
    lines.extend(("", f"merge gate: {'PASS' if gate_passes else 'FAIL'}"))
    return Judgement(tuple(lines), gate_passes)


def judge_json_records(
    first: bytes,
    second: bytes,
    *,
    sources: tuple[str, str] = ("record A", "record B"),
) -> Judgement:
    import json

    if hashlib.sha256(first).digest() == hashlib.sha256(second).digest():
        raise ValueError("the rule needs records from two different containers")
    records: list[PairedRecord] = []
    for content, source in zip((first, second), sources, strict=True):
        try:
            records.append(read_paired_record(json.loads(content)))
        except ValueError as error:
            raise ValueError(
                f"{source}: {error}. Repair the source record using the registered "
                "experiment schema in llm_infer/benchmarks/paired_ab.py; malformed input "
                "is not an experimental FAIL verdict."
            ) from error
    return judge_records(records[0], records[1])
