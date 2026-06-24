"""Runtime trace tests for the real ``InferenceEngine`` request path."""

from __future__ import annotations

import json

import torch

from llm_infer.kv_cache import BlockTable, PagedKVCache
from llm_infer.serving import InferenceEngine, Request
from llm_infer.tracing import TRACE_SCHEMA_VERSION, TraceRecorder


class TraceToyModel:
    """Small model-shaped object that writes real KV rows while producing scripted logits."""

    def __init__(self) -> None:
        self.num_layers = 1
        self.num_kv_heads = 1
        self.head_dim = 2
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.profiler = None

    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        return self.prefill_chunk(prompt_ids, cache, table, start_pos=0, chunk_size=len(prompt_ids))

    def prefill_chunk(
        self,
        prompt_ids: list[int],
        cache: PagedKVCache,
        table: BlockTable,
        *,
        start_pos: int,
        chunk_size: int,
    ) -> torch.Tensor:
        end_pos = min(len(prompt_ids), start_pos + chunk_size)
        table.reserve(end_pos - start_pos)
        rows = torch.arange(start_pos, end_pos, dtype=torch.float32).reshape(-1, 1, 1)
        key = torch.cat([rows, rows + 0.5], dim=-1)
        cache.write(table, layer=0, start_pos=start_pos, key=key, value=key + 100.0)
        table.length = end_pos
        return self._logits(10 + end_pos)

    def decode_many(
        self,
        cache: PagedKVCache,
        tables: list[BlockTable],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.as_tensor(token_ids, dtype=torch.long)
        positions = [table.length for table in tables]
        for table in tables:
            table.reserve(1)
        key = torch.stack(
            [
                torch.tensor([[float(pos), float(token)]], dtype=torch.float32)
                for pos, token in zip(positions, tokens.tolist(), strict=True)
            ]
        )
        cache.write_many(tables, layer=0, positions=positions, key=key, value=key + 200.0)
        for table, pos in zip(tables, positions, strict=True):
            table.length = pos + 1
        return torch.stack([self._logits(int(token) + 1) for token in tokens.tolist()])

    def _logits(self, token_id: int) -> torch.Tensor:
        logits = torch.full((64,), -100.0)
        logits[token_id] = 100.0
        return logits


def test_tracing_is_off_by_default() -> None:
    engine = InferenceEngine(TraceToyModel(), block_size=4, num_blocks=8)
    engine.add_request(Request("r", [1, 2], 2, frozenset({63})))

    assert engine.trace is None
    assert engine.run() == {"r": [12, 13]}


def test_trace_recorder_captures_real_engine_events() -> None:
    recorder = TraceRecorder()
    engine = InferenceEngine(
        TraceToyModel(),
        block_size=4,
        num_blocks=8,
        prefill_chunk_size=2,
        trace=recorder,
    )
    engine.add_request(Request("active", [1], 4, frozenset({63})))
    engine.add_request(Request("long", [2, 3, 4, 5, 6], 2, frozenset({63})))

    assert engine.run() == {"active": [11, 12, 13, 14], "long": [15, 16]}

    events = recorder.events
    names = [event.event for event in events]
    assert "request_admitted" in names
    assert "prefill_started" in names
    assert "prefill_progress" in names
    assert "decode_step" in names
    assert "request_finished" in names
    assert "batch_size_changed" in names
    assert "tokens_per_second_sampled" in names
    assert "block_allocated" not in names
    assert "block_freed" not in names

    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].event == "request_admitted"
    assert events[0].request_id == "active"
    assert events[0].prompt_tokens == 1
    assert events[0].reserved_blocks == 1

    long_progress = [
        event
        for event in events
        if event.event == "prefill_progress" and event.request_id == "long"
    ]
    assert [event.cached_tokens for event in long_progress] == [2, 4, 5]
    assert [event.completed for event in long_progress] == [False, False, True]

    decode_events = [event for event in events if event.event == "decode_step"]
    assert decode_events[0].request_ids == ("active",)
    assert decode_events[0].token_ids == (12,)
    assert any(event.request_ids == ("active", "long") for event in decode_events)

    finished = {event.request_id: event for event in events if event.event == "request_finished"}
    assert finished["active"].reason == "length"
    assert finished["active"].token_ids == (11, 12, 13, 14)
    assert finished["long"].generated_tokens == 2

    samples = [event for event in events if event.event == "tokens_per_second_sampled"]
    assert samples[-1].total_generated_tokens == 6
    assert samples[-1].tokens_per_second is not None
    assert samples[-1].tokens_per_second > 0

    first_json_event = json.loads(recorder.to_jsonl().splitlines()[0])
    assert first_json_event["schema_version"] == TRACE_SCHEMA_VERSION
    assert first_json_event["event"] == "request_admitted"
