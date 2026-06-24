from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.generate_kv_trace_fixture import FIXTURE_PATH, build_trace_jsonl

ROOT = Path(__file__).resolve().parents[2]


def _fixture_events() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (ROOT / FIXTURE_PATH).read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_committed_kv_trace_fixture_matches_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)

    assert (ROOT / FIXTURE_PATH).read_text(encoding="utf-8") == build_trace_jsonl()


def test_kv_trace_fixture_is_schema_v2_ordered_and_shaped() -> None:
    events = _fixture_events()
    names = {event["event"] for event in events}

    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert all(event["schema_version"] == 2 for event in events)
    assert {
        "request_admitted",
        "prefill_chunk_started",
        "prefill_chunk_progress",
        "decode_step",
        "request_finished",
        "batch_size_changed",
        "tokens_per_second_sampled",
    } <= names
    assert "block_allocated" not in names
    assert "block_freed" not in names
    assert any(event.get("prefix_group_id") == "rollout" for event in events)
    assert any(event.get("waiting") == 2 for event in events)
    assert any(
        event["event"] == "prefill_chunk_progress" and event.get("completed") is False
        for event in events
    )
    assert any(
        event["event"] == "decode_step" and len(event.get("request_ids", [])) > 1
        for event in events
    )


def test_kv_trace_fixture_has_no_missing_generated_token_events() -> None:
    events = _fixture_events()
    traced_tokens: dict[str, list[int]] = {}

    for event in events:
        if event["event"] != "decode_step":
            continue
        request_ids = event.get("request_ids", [])
        token_ids = event.get("token_ids", [])
        assert event.get("token_source") in {"prefill", "decode", "speculative"}
        if len(request_ids) == 1:
            traced_tokens.setdefault(request_ids[0], []).extend(token_ids)
            continue
        for request_id, token_id in zip(request_ids, token_ids, strict=True):
            traced_tokens.setdefault(request_id, []).append(token_id)

    finished = {
        event["request_id"]: event["token_ids"]
        for event in events
        if event["event"] == "request_finished"
    }
    assert traced_tokens == finished


def test_visualizer_parser_loads_fixture_with_node() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; browser parser smoke skipped")

    result = subprocess.run(
        [node, "--test", "visualizer/trace_loader.test.mjs"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
