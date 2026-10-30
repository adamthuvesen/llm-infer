"""Replay the evidence failure classes corrected in the benchmark history."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from llm_infer.benchmarks.paired_ab import judge_json_records
from llm_infer.benchmarks.reference_policy import headline_tokens_per_second
from scripts import check_benchmark_evidence as evidence_check
from scripts.judge_paired_ab import main as judge_main

ROOT = Path(__file__).resolve().parents[2]


def _pair(stem: str) -> tuple[dict, dict]:
    return tuple(
        json.loads((ROOT / "assets" / f"{stem}-{index}.json").read_text()) for index in (1, 2)
    )


def _judge(records: tuple[dict, dict]):
    return judge_json_records(*(json.dumps(record).encode() for record in records))


def test_policy_and_registered_judge_import_without_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from llm_infer.benchmarks.reference_policy import EsmeAgreement; "
            "from llm_infer.benchmarks.paired_ab import judge_json_records; "
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
    row = records[0]["configs"]["grouped"][2]
    assert row["parity_status"] == "exact"
    agreement = row["full_step"]["agreement"]
    agreement["exact"] -= 1
    agreement["nontie"] += 1
    agreement["review_required"] += 1
    result = _judge(records)
    assert result.exit_code == 1
    assert "grouped b64: not confirmed (non-tie divergence from fp32)" in result.lines


def test_exact_relative_parity_can_share_a_pending_reference_review() -> None:
    records = _pair("esme-full-step-ab")
    for record in records:
        row = record["configs"]["grouped"][0]
        assert row["parity_status"] == "exact"
        for arm in ("grouped4", "full_step"):
            agreement = row[arm]["agreement"]
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
