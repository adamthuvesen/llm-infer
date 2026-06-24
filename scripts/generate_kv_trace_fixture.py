"""Generate the committed schema-v2 trace used by the KV trace visualizer.

This is a **self-contained synthetic generator**: it does not import the inference
engine or torch. It runs a small, simplified continuous-batching simulation — FIFO
admission against a fixed block budget, chunked prefill, prefix-group sharing, batched
decode, and throughput sampling — and emits schema-v2 trace events that mirror the
shape ``InferenceEngine(trace=...)`` produces.

The trace is a labelled **synthetic sample**, not a measured benchmark: the prompts,
token ids, and per-step clock are all invented. Keeping the generator standalone lets
the visualizer ship on its own without the engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FIXTURE_PATH = Path("docs/assets/kv_trace_schema_v2.jsonl")

SCHEMA_VERSION = 2
BLOCK_SIZE = 4
NUM_BLOCKS = 12
PREFILL_CHUNK = 2
# Simulated per-sample step latency. Stands in for a plausible decode step (~6 ms) so the
# throughput samples are deterministic AND land in a realistic range, instead of reading
# as ~5 tok/s. This is invented time for a synthetic sample, not a measurement.
STEP_SECONDS = 0.006


def blocks_for(prompt_len: int, max_new: int) -> int:
    """Worst-case blocks: the prompt plus its decode budget, minus the prefill-sampled token."""
    max_positions = prompt_len + max_new - 1
    return -(-max_positions // BLOCK_SIZE)  # ceil division


@dataclass
class Req:
    """One simulated request: a prompt length, a decode budget, and a deterministic ramp."""

    request_id: str
    prompt_len: int
    max_new: int
    base_token: int
    group: str | None = None
    cached: int = 0
    prefilled: bool = False
    finished: bool = False
    generated: list[int] = field(default_factory=list)

    @property
    def reserved(self) -> int:
        return blocks_for(self.prompt_len, self.max_new)

    def emit_token(self) -> int:
        """Append the next token (a per-request ramp) and apply the length cap."""
        token = self.base_token + len(self.generated)
        self.generated.append(token)
        if len(self.generated) >= self.max_new:
            self.finished = True
        return token


class TraceBuilder:
    """Accumulates ordered schema-v2 events with a running sequence/step/throughput clock."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []
        self.seq = 0
        self.step = 0
        self.total_tokens = 0
        self.samples = 0
        self.last_batch = 0

    def emit(self, event: str, **fields: object) -> None:
        self.seq += 1
        self.events.append(
            {
                "event": event,
                "schema_version": SCHEMA_VERSION,
                "sequence": self.seq,
                "step": self.step,
                **fields,
            }
        )

    def batch_changed(self, size: int, waiting: int) -> None:
        if size == self.last_batch:
            return
        self.emit(
            "batch_size_changed",
            previous_batch_size=self.last_batch,
            batch_size=size,
            waiting=waiting,
        )
        self.last_batch = size

    def throughput(self, tokens_this_step: int) -> None:
        if tokens_this_step == 0:
            return
        self.total_tokens += tokens_this_step
        self.samples += 1
        elapsed = self.samples * STEP_SECONDS
        self.emit(
            "tokens_per_second_sampled",
            tokens_emitted=tokens_this_step,
            total_generated_tokens=self.total_tokens,
            elapsed_seconds=elapsed,
            tokens_per_second=self.total_tokens / elapsed,
        )

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(event, sort_keys=True) for event in self.events) + "\n"


def _admit(tb: TraceBuilder, waiting: list[Req], running: list[Req], committed: int) -> int:
    """FIFO admission with head-of-line blocking, mirroring the scheduler's block budget."""
    while waiting:
        need = waiting[0].reserved
        if committed + need > NUM_BLOCKS:
            break
        request = waiting.pop(0)
        running.append(request)
        committed += need
        fields = {"prefix_group_id": request.group} if request.group is not None else {}
        tb.emit(
            "request_admitted",
            request_id=request.request_id,
            prompt_tokens=request.prompt_len,
            max_new_tokens=request.max_new,
            reserved_blocks=need,
            **fields,
        )
    return committed


def _prefill_chunk(tb: TraceBuilder, leader: Req) -> bool:
    """Cache one prompt chunk for the leader; return True when its prompt is fully cached."""
    start = leader.cached
    end = min(leader.prompt_len, start + PREFILL_CHUNK)
    tb.emit(
        "prefill_chunk_started",
        request_id=leader.request_id,
        start_pos=start,
        end_pos=end,
        total_prompt_tokens=leader.prompt_len,
    )
    leader.cached = end
    completed = end == leader.prompt_len
    tb.emit(
        "prefill_chunk_progress",
        request_id=leader.request_id,
        start_pos=start,
        end_pos=end,
        cached_tokens=end,
        total_prompt_tokens=leader.prompt_len,
        completed=completed,
    )
    return completed


def _prefill(tb: TraceBuilder, to_prefill: list[Req]) -> int:
    """Advance prefill for unstarted requests; prefix siblings share the leader's cache."""
    tokens_emitted = 0
    handled: set[str] = set()
    for request in to_prefill:
        if request.request_id in handled:
            continue
        group = (
            [r for r in to_prefill if r.group == request.group]
            if request.group is not None
            else [request]
        )
        handled.update(r.request_id for r in group)

        leader = group[0]
        if not _prefill_chunk(tb, leader):
            continue

        # Prompt fully cached: siblings fork the leader's cache (no prefill events of their
        # own), and every group member samples its first token from the prefill logits.
        tokens = []
        for member in group:
            member.prefilled = True
            member.cached = leader.prompt_len
            tokens.append(member.emit_token())
        tb.emit(
            "decode_step",
            request_ids=[member.request_id for member in group],
            batch_size=len(group),
            token_ids=tokens,
            tokens_emitted=len(tokens),
            token_source="prefill",
        )
        tokens_emitted += len(tokens)
    return tokens_emitted


def _decode(tb: TraceBuilder, to_decode: list[Req]) -> int:
    """One batched decode step advancing every already-ready request by a token."""
    if not to_decode:
        return 0
    tokens = [request.emit_token() for request in to_decode]
    tb.emit(
        "decode_step",
        request_ids=[request.request_id for request in to_decode],
        batch_size=len(to_decode),
        token_ids=tokens,
        tokens_emitted=len(tokens),
        token_source="decode",
    )
    return len(tokens)


def build_trace_jsonl() -> str:
    """Run the synthetic scenario and return its schema-v2 JSONL.

    Six requests against twelve blocks: the first four fit at once and the last two queue,
    so the trace shows real waiting pressure, continuous-batching churn, a long chunked
    prefill (``code-gen``), prefix-shared rollouts, and a KV wall that fills to capacity.
    """
    tb = TraceBuilder()
    waiting = [
        Req("rollout-a", prompt_len=4, max_new=5, base_token=36, group="rollout"),
        Req("rollout-b", prompt_len=4, max_new=5, base_token=36, group="rollout"),
        Req("code-gen", prompt_len=12, max_new=8, base_token=37),
        Req("summarize", prompt_len=7, max_new=4, base_token=33),
        Req("chat-quick", prompt_len=2, max_new=6, base_token=26),
        Req("translate", prompt_len=9, max_new=6, base_token=32),
    ]
    running: list[Req] = []
    committed = 0
    step = 0

    while waiting or running:
        tb.step = step
        committed = _admit(tb, waiting, running, committed)
        tb.batch_changed(len(running), len(waiting))

        to_decode = [r for r in running if r.prefilled]
        to_prefill = [r for r in running if not r.prefilled]
        tokens_this_step = _prefill(tb, to_prefill) + _decode(tb, to_decode)

        for request in [r for r in running if r.finished]:
            tb.emit(
                "request_finished",
                request_id=request.request_id,
                token_ids=request.generated,
                generated_tokens=len(request.generated),
                reason="length",
            )
            committed -= request.reserved
            running.remove(request)
        tb.batch_changed(len(running), len(waiting))
        tb.throughput(tokens_this_step)
        step += 1

    return tb.to_jsonl()


def main() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(build_trace_jsonl(), encoding="utf-8")
    print(f"wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
