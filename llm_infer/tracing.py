"""Typed runtime traces for replaying real inference-engine events.

Tracing is opt-in: pass a :class:`TraceRecorder` to ``InferenceEngine(trace=...)``.
The recorder stores schema-versioned events in emission order and can serialize them as
JSON Lines for the static visualizer. Block allocation/free events are emitted from the real
physical boundary in :class:`~llm_infer.kv_cache.block_allocator.BlockAllocator` — a block is
``block_allocated`` only when it leaves the free pool and ``block_freed`` only when it truly
returns to it (refcount-0), so prefix-shared blocks retained by a sibling are never reported as
freed and a copy-on-write that allocates a new physical block is reported as an allocation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

TRACE_SCHEMA_VERSION = 3

TraceEventName = Literal[
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
]

FinishReason = Literal["eos", "length"]
PreemptReason = Literal["kv_pressure"]
TokenSource = Literal["prefill", "decode", "speculative"]


@dataclass(frozen=True)
class TraceEvent:
    """One schema-versioned event in the engine trace stream.

    Fields are intentionally explicit instead of a free-form payload. Unused fields stay
    ``None``/empty and are omitted by :meth:`to_dict`.
    """

    event: TraceEventName
    sequence: int
    step: int
    schema_version: int = field(default=TRACE_SCHEMA_VERSION, init=False)
    request_id: str | None = None
    request_ids: tuple[str, ...] = ()
    prompt_tokens: int | None = None
    max_new_tokens: int | None = None
    prefix_group_id: str | None = None
    reserved_blocks: int | None = None
    start_pos: int | None = None
    end_pos: int | None = None
    cached_tokens: int | None = None
    total_prompt_tokens: int | None = None
    completed: bool | None = None
    block_count: int | None = None
    block_ids: tuple[int, ...] = ()
    pool_used: int | None = None
    pool_free: int | None = None
    batch_size: int | None = None
    previous_batch_size: int | None = None
    waiting: int | None = None
    token_ids: tuple[int, ...] = ()
    tokens_emitted: int | None = None
    token_source: TokenSource | None = None
    generated_tokens: int | None = None
    reason: FinishReason | None = None
    preempt_reason: PreemptReason | None = None
    total_generated_tokens: int | None = None
    elapsed_seconds: float | None = None
    tokens_per_second: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a compact JSON-ready dict, omitting empty optional fields."""
        raw = asdict(self)
        return {
            key: value
            for key, value in raw.items()
            if value is not None and value != () and value != []
        }


class TraceRecorder:
    """In-memory trace sink for tests, scripts, and JSONL export."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def record(self, event: TraceEvent) -> None:
        self._events.append(event)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(event.to_dict(), sort_keys=True) for event in self._events)
