"""Replay the evidence failure classes corrected in the benchmark history."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from llm_infer.benchmarks.paired_ab import judge_json_records, read_paired_record
from llm_infer.benchmarks.reference_policy import (
    EsmeAgreement,
    SystemEvidence,
    headline_tokens_per_second,
)
from scripts import check_benchmark_evidence as evidence_check
from scripts.judge_paired_ab import main as judge_main
from tests.support.json_result import dig_dict, dig_number

ROOT = Path(__file__).resolve().parents[2]


def _pair(stem: str) -> tuple[dict[str, object], dict[str, object]]:
    return tuple(
        json.loads((ROOT / "assets" / f"{stem}-{index}.json").read_text()) for index in (1, 2)
    )


def _judge(records: tuple[dict[str, object], dict[str, object]]):
    return judge_json_records(*(json.dumps(record).encode() for record in records))


def test_policy_and_registered_judge_import_without_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from llm_infer.benchmarks.reference_policy import EsmeAgreement; "
            "from llm_infer.benchmarks.paired_ab import judge_json_records, read_paired_record; "
            "import sys; assert 'torch' not in sys.modules",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "fields",
    [
        {"reference_status": "failed"},
        {"tokens_per_second": None},
        {"tokens_per_second": 1.0},
    ],
)
def test_public_tables_reject_raw_timing_disguised_as_a_headline(
    fields: dict[str, object], tmp_path: Path
) -> None:
    # 9fea059 fixed a reader that trusted headline_eligible and published raw timing.
    record = json.loads((ROOT / "assets/esme-batch-curve.json").read_text())
    row = record["rows"][0]
    expected_status = row["reference_status"]
    row.update(fields)
    assert row["headline_eligible"] is True
    assert row["raw_tokens_per_second"] > 0
    with pytest.raises(ValueError, match="reference-gated"):
        headline_tokens_per_second(row)
    path = tmp_path / "invalid-headline.json"
    path.write_text(json.dumps({"rows": [row]}))
    values, errors = evidence_check._gated_rows(path)
    assert values == {}
    assert errors and "reference-gated" in errors[0]
    message = errors[0]
    assert str(path) in message
    assert row["system"] in message
    for field, observed in fields.items():
        assert f"{field}={observed!r}" in message
    if "reference_status" in fields:
        assert f"expected {expected_status!r} from agreement counts" in message
    else:
        assert f"expected raw_tokens_per_second={row['raw_tokens_per_second']!r}" in message
    assert "build_system_evidence_record() in llm_infer/benchmarks/reference_policy.py" in message
    assert "Keep tokens_per_second=None until the reference evidence qualifies" in message


def test_fusion_exact_direct_parity_cannot_bypass_its_registered_fp32_gate() -> None:
    # f209bfe fixed the exact-parity branch bypassing both arms' fp32 checks.
    records = _pair("esme-fusion-ab")
    row = dig_dict(records[0], "configs", "grouped", 2)
    assert row["parity_status"] == "exact"
    agreement = dig_dict(row, "full_step", "agreement")
    agreement["exact"] = dig_number(agreement, "exact") - 1
    agreement["nontie"] = dig_number(agreement, "nontie") + 1
    agreement["review_required"] = dig_number(agreement, "review_required") + 1
    result = _judge(records)
    assert result.exit_code == 1
    assert "grouped b64: not confirmed (non-tie divergence from fp32)" in result.lines


def test_exact_relative_parity_can_share_a_pending_reference_review() -> None:
    records = _pair("esme-full-step-ab")
    for record in records:
        row = dig_dict(record, "configs", "grouped", 0)
        assert row["parity_status"] == "exact"
        for arm in ("grouped4", "full_step"):
            agreement = dig_dict(row, arm, "agreement")
            agreement["exact"] = agreement["tie"] = 0
            agreement["nontie"] = agreement["review_required"] = agreement["total"]
    result = _judge(records)
    assert result.exit_code == 0
    assert "grouped b1: qualified gain -2.9%" in result.lines


def test_malformed_judge_input_is_not_an_expected_experiment_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Before 228636f, reproduce accepted any exit 1, including a judge traceback.
    records = _pair("esme-spec-decode-ab")
    del records[0]["configs"]
    paths = [tmp_path / f"container-{index}.json" for index in (1, 2)]
    for path, record in zip(paths, records, strict=True):
        path.write_text(json.dumps(record))
    assert judge_main([str(path) for path in paths]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"error: {paths[0]}: configs must be")
    assert "Repair the source record" in captured.err
    assert "llm_infer/benchmarks/paired_ab.py" in captured.err
    assert "malformed input is not an experimental FAIL verdict" in captured.err
    assert "Traceback" not in captured.err


def test_reproduction_rejects_a_crash_with_the_expected_failure_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def crashed(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, "", "KeyError: configs")

    monkeypatch.setattr(evidence_check.subprocess, "run", crashed)
    errors = evidence_check._ab_verdict_errors()
    assert len(errors) == len(evidence_check.AB_VERDICTS)
    assert any("entry 22" in error and "KeyError: configs" in error for error in errors)


def test_timeline_metadata_must_describe_executed_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ac6acde corrected this committed config from one warmup / three iterations.
    path = tmp_path / "assets/esme-decode-timeline.json"
    path.parent.mkdir()
    record = json.loads((ROOT / "assets/esme-decode-timeline.json").read_text())
    record["config"].update(warmup=1, iters=3)
    path.write_text(json.dumps(record))
    monkeypatch.setattr(evidence_check, "ROOT", tmp_path)
    errors = evidence_check._timeline_config_errors()
    assert len(errors) == 2
    assert "warmup=1 differs from child execution count 3" in errors[0]
    assert "iters=3 differs from child execution count 1" in errors[1]
    assert all("llm_infer/benchmarks/report.py" in error for error in errors)
    assert all("keep the measured timing unchanged" in error for error in errors)


def test_paired_record_snapshots_measured_rounds_and_row_membership() -> None:
    source, _ = _pair("esme-full-step-ab")
    record = read_paired_record(source)
    key = ("grouped", 1)
    walls = record.rows[key].baseline.walls_s
    agreement = record.rows[key].baseline.agreement.as_dict()
    arm = dig_dict(source, "configs", "grouped", 0, "grouped4")
    cast(list[float], arm["walls_s"])[0] = 999.0
    dig_dict(arm, "agreement")["exact"] = 0
    assert record.rows[key].baseline.walls_s == walls
    assert record.rows[key].baseline.agreement.as_dict() == agreement
    assert isinstance(walls, tuple)
    with pytest.raises(TypeError):
        record.rows[key] = record.rows[key]  # pyright: ignore[reportIndexIssue]  # Mapping is read-only on purpose


def test_wire_evidence_accepts_measured_rate_without_new_timing_metadata() -> None:
    row = json.loads((ROOT / "assets/esme-batch-curve.json").read_text())["rows"][0]
    del row["median_seconds"]
    del row["total_output_tokens"]
    evidence = SystemEvidence.from_record(row)
    assert evidence.tokens_per_second == row["tokens_per_second"]
    assert evidence.as_dict() == {key: row[key] for key in evidence.as_dict()}


def test_wire_evidence_rejects_rate_that_conflicts_with_recorded_measurements() -> None:
    row = json.loads((ROOT / "assets/esme-batch-curve.json").read_text())["rows"][0]
    row["raw_tokens_per_second"] *= 2
    row["tokens_per_second"] = row["raw_tokens_per_second"]
    with pytest.raises(ValueError, match="from measured timing and token count"):
        headline_tokens_per_second(row)


def test_committed_headline_claims_round_trip_through_immutable_evidence() -> None:
    def claims(value: object) -> Iterator[dict[str, object]]:
        if isinstance(value, dict):
            if (
                value.get("headline_eligible") is True
                and "agreement" in value
                and "raw_tokens_per_second" in value
            ):
                yield value
            for nested in value.values():
                yield from claims(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from claims(nested)

    rows = [
        row
        for path in (ROOT / "assets").rglob("*.json")
        for row in claims(json.loads(path.read_text()))
    ]
    assert rows
    for row in rows:
        evidence = SystemEvidence.from_record(row)
        assert headline_tokens_per_second(row) == evidence.tokens_per_second
        assert evidence.as_dict() == {key: row[key] for key in evidence.as_dict()}


def test_wire_evidence_rejects_boolean_public_rate_equal_to_a_valid_number() -> None:
    agreement = EsmeAgreement(1, 0, 0, 1, [], [])
    evidence = SystemEvidence(agreement, 1.0)
    row = {**evidence.as_dict(), "agreement": agreement.as_dict(), "tokens_per_second": True}
    with pytest.raises(ValueError, match="tokens_per_second=True"):
        headline_tokens_per_second(row)
