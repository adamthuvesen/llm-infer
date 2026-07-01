"""Runtime trace tests for the real ``InferenceEngine`` request path."""

from __future__ import annotations

import json

import torch

from llm_infer.kv_cache import BlockTable, PagedKVCache
from llm_infer.serving import InferenceEngine, Request
from llm_infer.tracing import TRACE_SCHEMA_VERSION, TraceRecorder
from tests.support.fake_causal_lm import FakeCausalLMBase


class TraceModel(FakeCausalLMBase):
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
    engine = InferenceEngine(TraceModel(), block_size=4, num_blocks=8)
    engine.add_request(Request("r", [1, 2], 2, frozenset({63})))

    assert engine.trace is None
    assert engine.run() == {"r": [12, 13]}


def test_trace_recorder_captures_real_engine_events() -> None:
    recorder = TraceRecorder()
    engine = InferenceEngine(
        TraceModel(),
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
    assert "prefill_chunk_started" in names
    assert "prefill_chunk_progress" in names
    assert "decode_step" in names
    assert "request_finished" in names
    assert "batch_size_changed" in names
    assert "tokens_per_second_sampled" in names
    assert "block_allocated" in names
    assert "block_freed" in names

    _assert_block_lifecycle_is_clear(events, pool_size=8)

    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert events[0].event == "request_admitted"
    assert events[0].request_id == "active"
    assert events[0].prompt_tokens == 1
    assert events[0].reserved_blocks == 1

    long_starts = [
        event
        for event in events
        if event.event == "prefill_chunk_started" and event.request_id == "long"
    ]
    assert [(event.start_pos, event.end_pos) for event in long_starts] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert [event.total_prompt_tokens for event in long_starts] == [5, 5, 5]

    long_progress = [
        event
        for event in events
        if event.event == "prefill_chunk_progress" and event.request_id == "long"
    ]
    assert [(event.start_pos, event.end_pos) for event in long_progress] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert [event.cached_tokens for event in long_progress] == [2, 4, 5]
    assert [event.completed for event in long_progress] == [False, False, True]

    decode_events = [event for event in events if event.event == "decode_step"]
    assert decode_events[0].request_ids == ("active",)
    assert decode_events[0].token_ids == (11,)
    assert decode_events[0].token_source == "prefill"
    assert any(event.request_ids == ("active", "long") for event in decode_events)

    finished = {event.request_id: event for event in events if event.event == "request_finished"}
    assert finished["active"].reason == "length"
    assert finished["active"].token_ids == (11, 12, 13, 14)
    assert finished["long"].generated_tokens == 2
    traced_tokens_by_request: dict[str, list[int]] = {"active": [], "long": []}
    for event in decode_events:
        if len(event.request_ids) == 1:
            traced_tokens_by_request[event.request_ids[0]].extend(event.token_ids)
            continue
        for request_id, token_id in zip(event.request_ids, event.token_ids, strict=True):
            traced_tokens_by_request[request_id].append(token_id)
    assert traced_tokens_by_request["active"] == [11, 12, 13, 14]
    assert traced_tokens_by_request["long"] == [15, 16]

    samples = [event for event in events if event.event == "tokens_per_second_sampled"]
    assert samples[-1].total_generated_tokens == 6
    assert samples[-1].tokens_per_second is not None
    assert samples[-1].tokens_per_second > 0

    first_json_event = json.loads(recorder.to_jsonl().splitlines()[0])
    assert first_json_event["schema_version"] == TRACE_SCHEMA_VERSION
    assert first_json_event["event"] == "request_admitted"


def _assert_block_lifecycle_is_clear(events: tuple, pool_size: int) -> None:
    """Every freed block was live, pool counts stay valid, and nothing frees before alloc."""
    live: set[int] = set()
    for event in events:
        if event.event not in {"block_allocated", "block_freed"}:
            continue
        block_ids = list(event.block_ids)
        assert block_ids, f"{event.event} must carry block_ids"
        assert event.block_count == len(block_ids)
        assert event.pool_used is not None and event.pool_free is not None
        assert event.pool_used >= 0 and event.pool_free >= 0
        assert event.pool_used + event.pool_free == pool_size

        if event.event == "block_allocated":
            for block in block_ids:
                assert block not in live, f"block {block} allocated while still live"
                live.add(block)
        else:
            for block in block_ids:
                assert block in live, f"free-before-alloc of block {block}"
                live.discard(block)
        assert len(live) == event.pool_used

    assert not live, f"blocks never returned to the pool: {sorted(live)}"


def test_preemption_emits_clear_preempt_and_resume_events() -> None:
    """Under a tight budget the engine evicts a victim, frees its KV, and resumes it by recompute.

    The victim's preemption fires a real ``block_freed`` (the allocator boundary), a
    ``request_preempted`` naming the pressure, and on resume a ``request_resumed`` plus replayed
    prefill_chunk events — the visible shape of recompute. The block lifecycle stays clear
    throughout (no free-before-alloc; every block returns to the pool).
    """
    recorder = TraceRecorder()
    # block_size 4, pool 3: three prompt-3 requests admit on footprint (1 block each), then must
    # evict as they grow past one block — a genuine forced preemption.
    engine = InferenceEngine(
        TraceModel(), block_size=4, num_blocks=3, preemption=True, trace=recorder
    )
    engine.add_request(Request("a", [1, 2, 3], 6, frozenset({63})))
    engine.add_request(Request("b", [4, 5, 6], 6, frozenset({63})))
    engine.add_request(Request("c", [7, 8, 9], 6, frozenset({63})))
    engine.run()

    events = recorder.events
    names = [event.event for event in events]
    assert "request_preempted" in names
    assert "request_resumed" in names

    _assert_block_lifecycle_is_clear(events, pool_size=3)

    preempts = [event for event in events if event.event == "request_preempted"]
    for event in preempts:
        assert event.preempt_reason == "kv_pressure"
        assert event.block_count >= 1
        assert event.generated_tokens is not None
        assert event.pool_used is not None and event.pool_free is not None
        assert event.pool_used + event.pool_free == 3

    # A preempted request resumes and replays prefill chunks (recompute) before decoding again.
    preempted_ids = {event.request_id for event in preempts}
    resumed_ids = {event.request_id for event in events if event.event == "request_resumed"}
    assert preempted_ids <= resumed_ids

    # Every request still finishes with all its tokens — preemption never drops output.
    finished = {
        event.request_id: event.generated_tokens
        for event in events
        if event.event == "request_finished"
    }
    assert set(finished) == {"a", "b", "c"}
    assert all(count == 6 for count in finished.values())


def test_block_lifecycle_is_clear_for_shared_prefix() -> None:
    """A shared prompt block is reported freed exactly once — by its last owner.

    Two prefix-group siblings share one full prompt block. The leader's free drops that
    block to refcount 1 (a sibling still owns it), so it is NOT reported freed there; only
    when the second sibling frees does the block truly return to the pool and get traced.
    """
    recorder = TraceRecorder()
    engine = InferenceEngine(TraceModel(), block_size=4, num_blocks=8, trace=recorder)
    engine.add_request(Request("sib-a", [1, 2, 3, 4], 3, frozenset({63}), prefix_group_id="g"))
    engine.add_request(Request("sib-b", [1, 2, 3, 4], 3, frozenset({63}), prefix_group_id="g"))
    engine.run()

    events = recorder.events
    _assert_block_lifecycle_is_clear(events, pool_size=8)

    allocated = [event for event in events if event.event == "block_allocated"]
    freed = [event for event in events if event.event == "block_freed"]

    # The shared prompt block is allocated once (by the leader) and never re-allocated:
    # the sibling retains it by refcount, which emits no allocation.
    shared_block = allocated[0].block_ids[0]
    assert sum(shared_block in event.block_ids for event in allocated) == 1
    # And it is reported freed exactly once across both siblings — last-owner only.
    assert sum(shared_block in event.block_ids for event in freed) == 1
    free_owner = next(event.request_id for event in freed if shared_block in event.block_ids)
    assert free_owner == "sib-b"
