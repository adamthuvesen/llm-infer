"""The committed Esme KV trace is a real engine run that the visualizer can render.

``docs/assets/esme_kv_trace_schema_v3.jsonl`` is produced by ``scripts/generate_esme_kv_trace.py``
driving the actual ``InferenceEngine`` over the tiny Esme bundle with preemption on — every event
comes from the real engine/scheduler/allocator on the Esme backend, not a synthetic simulation.
These tests pin that the artifact still regenerates byte-for-byte (so it can't silently drift) and
that the browser trace loader builds a renderable model from it (so an Esme run really does show
up in the visualizer).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.generate_esme_kv_trace import FIXTURE_PATH, build_trace_jsonl

ROOT = Path(__file__).resolve().parents[2]


def _events() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (ROOT / FIXTURE_PATH).read_text(encoding="utf-8").splitlines()
        if line
    ]


def test_committed_esme_trace_matches_generator() -> None:
    """The artifact is exactly what the deterministic Esme engine run emits — no drift."""
    assert (ROOT / FIXTURE_PATH).read_text(encoding="utf-8") == build_trace_jsonl()


def test_esme_trace_is_a_real_engine_run_with_preemption() -> None:
    """Schema-v3, contiguous, and carries the full real-engine surface incl. a forced preemption."""
    events = _events()
    names = {event["event"] for event in events}

    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert all(event["schema_version"] == 3 for event in events)
    # The full engine surface a real Esme run produces — not a subset.
    assert {
        "request_admitted",
        "prefill_chunk_started",
        "prefill_chunk_progress",
        "decode_step",
        "block_allocated",
        "block_freed",
        "request_preempted",
        "request_resumed",
        "request_finished",
        "batch_size_changed",
        "tokens_per_second_sampled",
    } <= names
    preempts = [event for event in events if event["event"] == "request_preempted"]
    resumes = [event for event in events if event["event"] == "request_resumed"]
    assert preempts, "the Esme demo trace must show a real preemption"
    assert {event["request_id"] for event in preempts} <= {event["request_id"] for event in resumes}
    for event in preempts:
        assert event["preempt_reason"] == "kv_pressure"


def test_esme_trace_decode_tokens_match_finish_tokens() -> None:
    """No dropped tokens: every request's decoded ids equal the ids it finished with."""
    events = _events()
    decoded: dict[str, list[int]] = {}
    for event in events:
        if event["event"] != "decode_step":
            continue
        request_ids = event.get("request_ids", [])
        token_ids = event.get("token_ids", [])
        if len(request_ids) == 1:
            decoded.setdefault(request_ids[0], []).extend(token_ids)
            continue
        for request_id, token_id in zip(request_ids, token_ids, strict=True):
            decoded.setdefault(request_id, []).append(token_id)

    finished = {
        event["request_id"]: event["token_ids"]
        for event in events
        if event["event"] == "request_finished"
    }
    assert decoded == finished


def test_visualizer_loads_esme_trace_with_node() -> None:
    """The browser trace loader builds a renderable model from the Esme run (skips without node)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed; visualizer loader smoke skipped")

    script = (
        "import assert from 'node:assert/strict';"
        "import { readFile } from 'node:fs/promises';"
        "import { buildTraceModel, parseJsonlTrace } from './visualizer/trace_loader.js';"
        f"const text = await readFile('{FIXTURE_PATH}', 'utf8');"
        "const events = parseJsonlTrace(text);"
        "const model = buildTraceModel(events);"
        "assert.ok(model.requestList.length >= 1, 'no request lanes');"
        "assert.equal(model.hasBlockLifecycle, true);"
        "assert.ok(model.batchSignals.length > 0 && model.throughputSignals.length > 0);"
        "for (const r of model.requestList) {"
        "  assert.deepEqual(r.decodes.flatMap(d => d.tokenIds), r.finish.tokenIds);"
        "}"
        "const pid = events.filter(e => e.event === 'request_preempted')[0].request_id;"
        "const lane = model.requestList.find(r => r.requestId === pid);"
        "assert.ok(lane.preempts.length > 0 && lane.resumes.length > 0);"
    )
    result = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
