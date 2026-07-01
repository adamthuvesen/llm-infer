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


def test_kv_trace_fixture_is_schema_v3_ordered_and_shaped() -> None:
    events = _fixture_events()
    names = {event["event"] for event in events}

    assert [event["sequence"] for event in events] == list(range(1, len(events) + 1))
    assert all(event["schema_version"] == 3 for event in events)
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
    assert any(event.get("prefix_group_id") == "shared prompt" for event in events)
    assert any(
        event["event"] == "batch_size_changed" and event.get("waiting", 0) >= 1 for event in events
    )
    assert any(
        event["event"] == "prefill_chunk_progress" and event.get("completed") is False
        for event in events
    )
    assert any(
        event["event"] == "decode_step" and len(event.get("request_ids", [])) > 1
        for event in events
    )


def test_kv_trace_fixture_shows_clear_preemption_and_resume() -> None:
    """The bundled demo includes at least one real preempt + resume, shaped like the engine."""
    events = _fixture_events()
    preempts = [event for event in events if event["event"] == "request_preempted"]
    resumes = [event for event in events if event["event"] == "request_resumed"]

    assert preempts, "the demo fixture must show at least one preemption"
    assert resumes, "a preempted request must resume"

    for event in preempts:
        assert event["preempt_reason"] == "kv_pressure"
        assert event["block_count"] >= 1  # freed at least one KV block
        assert event["generated_tokens"] >= 1  # kept its progress
        assert event["pool_used"] + event["pool_free"] == 12

    # Every preempted request resumes, and a preempt is immediately preceded by the clear
    # block_freed that returned its KV to the pool (recompute eviction, not a bookkeeping fudge).
    preempted_ids = {event["request_id"] for event in preempts}
    resumed_ids = {event["request_id"] for event in resumes}
    assert preempted_ids <= resumed_ids

    by_seq = {event["sequence"]: event for event in events}
    for event in preempts:
        prior = by_seq.get(event["sequence"] - 1)
        assert prior is not None and prior["event"] == "block_freed"
        assert prior["request_id"] == event["request_id"]


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


def test_kv_trace_fixture_block_lifecycle_is_clear() -> None:
    """Every freed block was live, pool counts stay valid, and the shared prefix frees once."""
    events = _fixture_events()
    pool_size = None
    live: set[int] = set()
    allocations = 0
    frees = 0
    for event in events:
        if event["event"] not in {"block_allocated", "block_freed"}:
            continue
        block_ids = event["block_ids"]
        assert block_ids, "block lifecycle events must carry block_ids"
        assert event["block_count"] == len(block_ids)
        assert event["pool_used"] >= 0 and event["pool_free"] >= 0
        total = event["pool_used"] + event["pool_free"]
        pool_size = total if pool_size is None else pool_size
        assert total == pool_size

        if event["event"] == "block_allocated":
            allocations += 1
            for block in block_ids:
                assert block not in live, f"block {block} allocated while still live"
                live.add(block)
        else:
            frees += 1
            for block in block_ids:
                assert block in live, f"free-before-alloc of block {block}"
                live.discard(block)
        assert len(live) == event["pool_used"]

    assert allocations > 0 and frees > 0
    assert not live, f"blocks never returned to the pool: {sorted(live)}"

    # The shared prompt block is allocated once (the leader) and the sibling retains it by
    # fork — so it is never re-allocated and is reported freed exactly once, by the last owner.
    leader_first_alloc = next(
        event["block_ids"][0]
        for event in events
        if event["event"] == "block_allocated" and event.get("request_id") == "sample-a"
    )
    sibling_allocs = [
        block
        for event in events
        if event["event"] == "block_allocated" and event.get("request_id") == "sample-b"
        for block in event["block_ids"]
    ]
    assert leader_first_alloc not in sibling_allocs


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
