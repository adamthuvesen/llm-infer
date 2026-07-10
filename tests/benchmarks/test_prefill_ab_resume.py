"""Rows-log parsing and final-record assembly for the resumable prefill A/B harness."""

from __future__ import annotations

import json

import pytest

from scripts.modal_esme_prefill_ab import (
    MIXED_LOAD_KEY_FIELDS,
    PREFILL_AB_KEY_FIELDS,
    assemble_final_record,
    completed_row_keys,
    key_tuple,
    parse_event_lines,
    row_key,
)


def _meta_event(backend: str) -> dict[str, object]:
    return {
        "kind": "meta",
        "gpu": {"name": "A100"},
        "versions": {"torch": "2.4"},
        "attention_backend": backend,
        "decode_graphs": {"capture_sizes": [1, 8], "capture_s": 0.5},
    }


def _row_event(
    batch_size: int, shape: str, context_length: int | None, out: int
) -> dict[str, object]:
    return {
        "kind": "row",
        "batch_size": batch_size,
        "shape": shape,
        "context_length": context_length,
        "max_new_tokens": out,
        "candidate_speedup": {"prefill_device_seconds": 1.2},
    }


def _log(events: list[dict[str, object]]) -> str:
    return "".join(json.dumps(event) + "\n" for event in events)


def test_row_key_and_tuple_agree_including_ragged_none() -> None:
    row = _row_event(8, "ragged", None, 64)
    key = row_key(row, PREFILL_AB_KEY_FIELDS)

    assert key == {
        "batch_size": 8,
        "shape": "ragged",
        "context_length": None,
        "max_new_tokens": 64,
    }
    assert key_tuple(key, PREFILL_AB_KEY_FIELDS) == (8, "ragged", None, 64)


def test_completed_row_keys_ignores_meta_and_preserves_order() -> None:
    events = [
        _meta_event("Flash"),
        _row_event(1, "uniform", 16, 1),
        _row_event(8, "ragged", None, 64),
    ]

    assert completed_row_keys(events, PREFILL_AB_KEY_FIELDS) == [
        {"batch_size": 1, "shape": "uniform", "context_length": 16, "max_new_tokens": 1},
        {"batch_size": 8, "shape": "ragged", "context_length": None, "max_new_tokens": 64},
    ]


def test_row_key_honors_a_different_command_key_field_set() -> None:
    # The mixed-load command keys resume on (burst_size, burst_shape), not the prefill fields.
    row = {"kind": "row", "burst_size": 64, "burst_shape": "uniform", "burst_context": 512}

    key = row_key(row, MIXED_LOAD_KEY_FIELDS)

    assert key == {"burst_size": 64, "burst_shape": "uniform"}
    assert key_tuple(key, MIXED_LOAD_KEY_FIELDS) == (64, "uniform")
    assert completed_row_keys([row], MIXED_LOAD_KEY_FIELDS) == [key]


def test_parse_event_lines_skips_blank_lines() -> None:
    text = _log([_meta_event("Flash"), _row_event(1, "uniform", 16, 1)]) + "\n"

    events = parse_event_lines(text)

    assert [event["kind"] for event in events] == ["meta", "row"]


def test_parse_event_lines_rejects_malformed_line_with_number() -> None:
    text = _log([_meta_event("Flash")]) + "{not valid json}\n"

    with pytest.raises(ValueError, match="line 2"):
        parse_event_lines(text)


def test_parse_event_lines_rejects_event_without_kind() -> None:
    text = json.dumps({"batch_size": 1}) + "\n"

    with pytest.raises(ValueError, match="line 1"):
        parse_event_lines(text)


def test_assemble_final_record_combines_rows_and_uses_newest_meta() -> None:
    # A resumed log: an old meta and its rows, then a fresh meta and appended rows.
    events = [
        _meta_event("OldBackend"),
        _row_event(1, "uniform", 16, 1),
        _meta_event("NewBackend"),
        _row_event(8, "ragged", None, 64),
    ]

    record = assemble_final_record(events, config={"command": "prefill-ab"})

    assert record["attention_backend"] == "NewBackend"
    assert record["config"] == {"command": "prefill-ab"}
    assert [row["batch_size"] for row in record["rows"]] == [1, 8]
    # Rows carry the domain schema, not the streaming envelope.
    assert all("kind" not in row for row in record["rows"])


def test_assemble_final_record_requires_a_meta_event() -> None:
    with pytest.raises(ValueError, match="no meta event"):
        assemble_final_record([_row_event(1, "uniform", 16, 1)], config={})
