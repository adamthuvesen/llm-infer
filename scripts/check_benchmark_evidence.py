#!/usr/bin/env python3
"""Private benchmark-evidence checks for committed public numbers."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURVE_RECORD = ROOT / "assets" / "esme-batch-curve.json"
LADDER_RECORD = ROOT / "assets" / "esme-ablation-ladder.json"
VLLM_RECORD = ROOT / "assets" / "esme-vllm-same-host.json"
PUBLIC_BENCHMARK_DOC = ROOT / "docs" / "benchmark.md"
README = ROOT / "README.md"
PLOT_SCRIPT = ROOT / "scripts" / "plot_benchmark_curve.py"
EXPECTED_BATCHES = (8, 16, 32, 64, 128, 256)
PLOTTED_SYSTEMS = ("llm_infer", "hf_sequential")
QUALIFIED_REFERENCE_STATUSES = frozenset({"exact", "accepted_numerical"})


def _failures() -> list[str]:
    record = json.loads(CURVE_RECORD.read_text(encoding="utf-8"))
    rows = record.get("rows")
    if not isinstance(rows, list):
        return [f"{CURVE_RECORD}: expected top-level rows list"]

    errors: list[str] = []
    by_system: dict[str, list[dict]] = {system: [] for system in PLOTTED_SYSTEMS}
    for row in rows:
        system = row.get("system")
        if system in by_system:
            by_system[system].append(row)
        if row.get("policy_version") != 2:
            errors.append(
                f"{CURVE_RECORD}: {system} batch={row.get('batch_size')} does not declare "
                "policy_version 2"
            )
        if not isinstance(row.get("headline_eligible"), bool):
            errors.append(
                f"{CURVE_RECORD}: {system} batch={row.get('batch_size')} does not declare "
                "headline_eligible"
            )
        if row.get("headline_eligible") is not True and row.get("tokens_per_second") is not None:
            errors.append(
                f"{CURVE_RECORD}: {system} batch={row.get('batch_size')} reports public tok/s "
                "without headline eligibility"
            )

    for system, system_rows in by_system.items():
        batches = tuple(row.get("batch_size") for row in system_rows)
        if batches != EXPECTED_BATCHES:
            errors.append(
                f"{CURVE_RECORD}: {system} batches must be {EXPECTED_BATCHES}, got {batches}"
            )
        for row in system_rows:
            batch = row.get("batch_size")
            if row.get("reference_status") not in QUALIFIED_REFERENCE_STATUSES:
                errors.append(f"{CURVE_RECORD}: {system} batch={batch} is not reference-gated")
            if row.get("headline_eligible") is not True:
                errors.append(f"{CURVE_RECORD}: {system} batch={batch} is not headline-eligible")
            if row.get("tokens_per_second") is None:
                errors.append(f"{CURVE_RECORD}: {system} batch={batch} has no tok/s")
            agreement = row.get("agreement", {})
            if agreement.get("nontie") != 0:
                errors.append(f"{CURVE_RECORD}: {system} batch={batch} has non-tie divergence")

    errors.extend(_doc_table_errors(by_system))
    errors.extend(_headline_table_errors())
    errors.extend(_figure_regeneration_errors())
    return errors


def _gated_rows(path: Path) -> tuple[dict[tuple[str, int, int], float], list[str]]:
    """Raw tok/s keyed by (system, context, batch); every row must be gated with no non-tie."""
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
    errors: list[str] = []
    tok_s: dict[tuple[str, int, int], float] = {}
    for row in rows:
        key = (row["system"], int(row.get("context_tokens", 0)), int(row["batch_size"]))
        if row.get("headline_eligible") is not True or row.get("agreement", {}).get("nontie") != 0:
            errors.append(f"{path}: {key} is not a headline-eligible, non-tie-free row")
        tok_s[key] = float(row["raw_tokens_per_second"])
    return tok_s, errors


def _headline_table_errors() -> list[str]:
    """The ladder and vLLM tables in docs/benchmark.md and README.md must match their records."""
    ladder, errors = _gated_rows(LADDER_RECORD)
    vllm, vllm_errors = _gated_rows(VLLM_RECORD)
    errors.extend(vllm_errors)
    expected: list[str] = []
    for batch in (8, 64, 256):
        hf_sequential = ladder.get(("hf_sequential", 0, batch))
        hf_batched = ladder[("hf_batched", 0, batch)]
        engine = ladder[("+grouped_graphs", 0, batch)]
        floor = f"{hf_sequential:,.1f}" if hf_sequential is not None else "—"
        expected.append(
            f"| {batch} | {floor} | {hf_batched:,.1f} | {engine:,.1f} | "
            f"**{engine / hf_batched:.1f}x** |"
        )
    for batch in (1, 8, 64, 256):
        engine = vllm[("llm_infer_grouped", 256, batch)]
        reference = vllm[("vllm", 256, batch)]
        expected.append(
            f"| {batch} | {engine:,.1f} | {reference:,.1f} | {reference / engine:.2f}x |"
        )
    for doc in (PUBLIC_BENCHMARK_DOC, README):
        lines = set(doc.read_text(encoding="utf-8").splitlines())
        errors.extend(
            f"{doc}: missing headline row {line!r} from the committed records"
            for line in expected
            if line not in lines
        )
    return errors


def _doc_table_errors(by_system: dict[str, list[dict]]) -> list[str]:
    expected: dict[int, dict[str, float]] = {batch: {} for batch in EXPECTED_BATCHES}
    for system, rows in by_system.items():
        for row in rows:
            expected[int(row["batch_size"])][system] = round(float(row["tokens_per_second"]), 1)

    found: dict[int, tuple[float, float]] = {}
    pattern = re.compile(r"^\|\s*(\d+)\s*\|\s*([\d,]+\.\d)\s*\|\s*([\d,]+\.\d)\s*\|$")
    for line in PUBLIC_BENCHMARK_DOC.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        batch = int(match.group(1))
        if batch in EXPECTED_BATCHES:
            found[batch] = (
                float(match.group(2).replace(",", "")),
                float(match.group(3).replace(",", "")),
            )

    errors: list[str] = []
    if tuple(found) != EXPECTED_BATCHES:
        errors.append(
            f"{PUBLIC_BENCHMARK_DOC}: curve table batches must be {EXPECTED_BATCHES}, "
            f"got {tuple(found)}"
        )
        return errors

    for batch, (doc_llm, doc_hf) in found.items():
        exp = expected[batch]
        if doc_llm != exp["llm_infer"]:
            errors.append(
                f"{PUBLIC_BENCHMARK_DOC}: batch {batch} llm_infer tok/s {doc_llm} "
                f"!= committed {exp['llm_infer']}"
            )
        if doc_hf != exp["hf_sequential"]:
            errors.append(
                f"{PUBLIC_BENCHMARK_DOC}: batch {batch} hf_sequential tok/s {doc_hf} "
                f"!= committed {exp['hf_sequential']}"
            )
    return errors


def _figure_regeneration_errors() -> list[str]:
    with tempfile.TemporaryDirectory(prefix="llm-infer-evidence-") as tmp:
        output_dir = Path(tmp)
        result = subprocess.run(
            ["uv", "run", str(PLOT_SCRIPT), "--output-dir", str(output_dir)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            return [f"{PLOT_SCRIPT}: figure regeneration failed: {detail}"]
        return [
            f"{PLOT_SCRIPT}: did not write a non-empty SVG to {output_dir / name}"
            for name in ("fig-esme-batch-curve.svg", "fig-esme-ablation-ladder.svg")
            if not (output_dir / name).is_file() or (output_dir / name).stat().st_size == 0
        ]


def main() -> int:
    errors = _failures()
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("benchmark evidence: committed records, public tables, and figure regen are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
