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
PUBLIC_BENCHMARK_DOC = ROOT / "docs" / "benchmark.md"
PLOT_SCRIPT = ROOT / "scripts" / "plot_benchmark_curve.py"
EXPECTED_BATCHES = (8, 16, 32, 64, 128, 256)
PLOTTED_SYSTEMS = ("llm_infer", "hf_sequential")


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
        if row.get("matches_reference") is False and row.get("tokens_per_second") is not None:
            errors.append(
                f"{CURVE_RECORD}: {system} batch={row.get('batch_size')} reports tok/s "
                "despite failed reference agreement"
            )

    for system, system_rows in by_system.items():
        batches = tuple(row.get("batch_size") for row in system_rows)
        if batches != EXPECTED_BATCHES:
            errors.append(
                f"{CURVE_RECORD}: {system} batches must be {EXPECTED_BATCHES}, got {batches}"
            )
        for row in system_rows:
            batch = row.get("batch_size")
            if row.get("matches_reference") is not True:
                errors.append(f"{CURVE_RECORD}: {system} batch={batch} is not reference-gated")
            if row.get("tokens_per_second") is None:
                errors.append(f"{CURVE_RECORD}: {system} batch={batch} has no tok/s")
            agreement = row.get("agreement", {})
            if agreement.get("nontie") != 0:
                errors.append(f"{CURVE_RECORD}: {system} batch={batch} has non-tie divergence")

    errors.extend(_doc_table_errors(by_system))
    errors.extend(_figure_regeneration_errors())
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
        figure = output_dir / "fig-esme-batch-curve.svg"
        if not figure.is_file() or figure.stat().st_size == 0:
            return [f"{PLOT_SCRIPT}: did not write a non-empty SVG to {figure}"]
    return []


def main() -> int:
    errors = _failures()
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("benchmark evidence: committed curve, public table, and figure regen are consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
