"""Generate the committed schema-v3 trace used by the KV trace visualizer.

This is a **self-contained synthetic generator**: it does not import the inference
engine or torch. It runs a small, simplified continuous-batching simulation — footprint
admission against a fixed pool, chunked prefill, prefix-group sharing, batched decode,
lazy physical block allocation, recompute **preemption** under KV pressure, and throughput
sampling — and emits schema-v3 trace events that mirror the shape
``InferenceEngine(trace=..., preemption=True)`` produces.

Preemption is simulated consistently with the engine: admission reserves only the current
footprint (over-committing the pool), and when a running request then needs a block the pool
cannot give, the most-recently-admitted request (LIFO) is evicted — its KV freed, its tokens
kept — then later resumed by recomputing prompt-plus-generated. So the trace shape (free →
requeue → re-prefill → continue) matches what the real engine emits.

Block lifecycle is emitted honestly. A small ``BlockPool`` mirrors the real refcounted
allocator: a ``block_allocated`` fires only when a block leaves the free pool, a
``block_freed`` fires only when a block truly returns to it (refcount-0), and a shared
prefix block retained by a forked sibling is never re-allocated and is freed once, by its
last owner. A preemption frees the victim's blocks the same way.

The trace is a labelled **synthetic sample**, not a measured benchmark: the prompts,
token ids, and per-step clock are all invented. Keeping the generator standalone lets
the visualizer ship on its own without the engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

FIXTURE_PATH = Path("docs/assets/kv_trace_schema_v3.jsonl")

SCHEMA_VERSION = 3
BLOCK_SIZE = 4
NUM_BLOCKS = 12
PREFILL_CHUNK = 2
# Simulated per-sample step latency. Stands in for a plausible decode step (~6 ms) so the
# throughput samples are deterministic AND land in a realistic range, instead of reading
# as ~5 tok/s. This is invented time for a synthetic sample, not a measurement.
STEP_SECONDS = 0.006


def blocks_for_length(length: int) -> int:
    """How many physical blocks ``length`` cached positions occupy."""
    return -(-length // BLOCK_SIZE)  # ceil division


class BlockPool:
    """A refcounted free list mirroring ``BlockAllocator`` so block events stay honest.

    Hands out ids from the end (LIFO) and frees only the blocks whose refcount reaches
    zero, exactly like the engine's allocator. ``retain`` shares a block without a new
    allocation, so a forked prefix sibling emits no ``block_allocated`` for shared blocks.
    """

    def __init__(self, num_blocks: int) -> None:
        self.num_blocks = num_blocks
        self._free: list[int] = list(range(num_blocks))
        self._refcounts: list[int] = [0] * num_blocks

    @property
    def used(self) -> int:
        return self.num_blocks - len(self._free)

    @property
    def free(self) -> int:
        return len(self._free)

    def allocate(self, count: int) -> list[int]:
        out = self._free[-count:]
        del self._free[-count:]
        for block in out:
            self._refcounts[block] = 1
        return out

    def retain(self, blocks: list[int]) -> None:
        for block in blocks:
            self._refcounts[block] += 1

    def free_blocks(self, blocks: list[int]) -> list[int]:
        """Drop one reference each; return only the ids that truly returned to the pool."""
        returned: list[int] = []
        for block in blocks:
            self._refcounts[block] -= 1
            if self._refcounts[block] == 0:
                self._free.append(block)
                returned.append(block)
        return returned


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
    blocks: list[int] = field(default_factory=list)
    length: int = 0
    preempted: bool = False

    @property
    def footprint(self) -> int:
        """Blocks the request's KV occupies right now — its (re)admission footprint.

        Mirrors the engine's ``blocks_for_footprint``: a fresh request needs its prompt, a
        resumed one needs prompt-plus-generated (recompute rebuilds the whole prefix).
        """
        return blocks_for_length(self.prompt_len + len(self.generated))

    @property
    def resume_length(self) -> int:
        """Positions a recompute resume rebuilds: prompt plus all generated tokens but the last.

        The last generated token is re-fed by the resuming decode at its original position, so
        it is not re-prefilled — matching ``Request.recompute_prompt_ids`` in the engine.
        """
        return self.prompt_len + max(0, len(self.generated) - 1)

    def emit_token(self) -> int:
        """Append the next token (a per-request ramp) and apply the length cap."""
        token = self.base_token + len(self.generated)
        self.generated.append(token)
        if len(self.generated) >= self.max_new:
            self.finished = True
        return token

    def reset_for_recompute(self) -> None:
        """Drop cached-KV state for preemption, keeping generated tokens for later recompute."""
        self.blocks = []
        self.cached = 0
        self.length = 0
        self.prefilled = False
        self.preempted = True


class TraceBuilder:
    """Accumulates ordered schema-v3 events with a running sequence/step/throughput clock."""

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

    def block_allocated(self, request: Req, pool: BlockPool, block_ids: list[int]) -> None:
        if not block_ids:
            return
        self.emit(
            "block_allocated",
            request_id=request.request_id,
            block_count=len(block_ids),
            block_ids=block_ids,
            pool_used=pool.used,
            pool_free=pool.free,
        )

    def block_freed(self, request: Req, pool: BlockPool, block_ids: list[int]) -> None:
        if not block_ids:
            return
        self.emit(
            "block_freed",
            request_id=request.request_id,
            block_count=len(block_ids),
            block_ids=block_ids,
            pool_used=pool.used,
            pool_free=pool.free,
        )

    def preempted(self, request: Req, pool: BlockPool, freed_blocks: int) -> None:
        self.emit(
            "request_preempted",
            request_id=request.request_id,
            preempt_reason="kv_pressure",
            block_count=freed_blocks,
            generated_tokens=len(request.generated),
            pool_used=pool.used,
            pool_free=pool.free,
        )

    def resumed(self, request: Req, pool: BlockPool) -> None:
        self.emit(
            "request_resumed",
            request_id=request.request_id,
            prompt_tokens=request.prompt_len,
            generated_tokens=len(request.generated),
            cached_tokens=request.cached,
            pool_used=pool.used,
            pool_free=pool.free,
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


def _admit(tb: TraceBuilder, waiting: list[Req], running: list[Req], pool: BlockPool) -> None:
    """Footprint admission against the free pool, mirroring the preemption scheduler.

    Admits while each request's *current* footprint (prompt, plus generated for a resume) fits
    in the blocks actually free right now. This deliberately over-commits — admission no longer
    reserves worst-case decode budget — so the pool can be exhausted and the engine must preempt.
    A re-admitted request is not re-announced with ``request_admitted``; its later
    ``request_resumed`` marks its return.
    """
    free = pool.free
    while waiting:
        request = waiting[0]
        need = request.footprint
        if need > free:
            break
        free -= need
        waiting.pop(0)
        running.append(request)
        if request.preempted:
            continue  # a resume, not a first admission — request_resumed will mark it
        fields = {"prefix_group_id": request.group} if request.group is not None else {}
        tb.emit(
            "request_admitted",
            request_id=request.request_id,
            prompt_tokens=request.prompt_len,
            max_new_tokens=request.max_new,
            reserved_blocks=need,
            **fields,
        )


def _preempt(
    tb: TraceBuilder,
    victim: Req,
    running: list[Req],
    waiting: list[Req],
    pool: BlockPool,
) -> None:
    """Evict ``victim`` by recompute: free its KV honestly, keep its tokens, requeue it front."""
    returned = pool.free_blocks(victim.blocks)
    freed = len(victim.blocks)
    victim.reset_for_recompute()
    running.remove(victim)
    waiting.insert(0, victim)
    if returned:
        tb.block_freed(victim, pool, returned)
    tb.preempted(victim, pool, freed)


def _make_room(
    tb: TraceBuilder,
    demand: int,
    block_needers: list[Req],
    running: list[Req],
    waiting: list[Req],
    pool: BlockPool,
) -> None:
    """Preempt LIFO victims until the pool can satisfy ``demand`` new blocks for ``block_needers``.

    The victim is the most-recently-admitted running request not among the block-needers; if a
    needer is itself the newest, it can be evicted too (and drops out of its step). Forward
    progress holds because every request fits the empty pool alone.
    """
    needer_ids = {r.request_id for r in block_needers}
    while demand > pool.free:
        victim = next((r for r in reversed(running) if r.request_id not in needer_ids), None)
        if victim is None:
            # Only block-needers remain; evict the newest of them to shrink demand.
            victim = running[-1]
        _preempt(tb, victim, running, waiting, pool)


def _blocks_to_grow(request: Req, new_length: int) -> int:
    """How many *new* blocks ``request`` must pull to hold ``new_length`` positions."""
    return max(0, blocks_for_length(new_length) - len(request.blocks))


def _grow(tb: TraceBuilder, request: Req, pool: BlockPool, new_length: int) -> None:
    """Lazily allocate physical blocks so ``request`` can hold ``new_length`` positions.

    Mirrors ``BlockTable.reserve``: pulls fresh blocks from the pool only when the existing
    ones cannot cover the new length, emitting one ``block_allocated`` for the delta.
    """
    request.length = new_length
    needed = blocks_for_length(new_length)
    if needed > len(request.blocks):
        new_blocks = pool.allocate(needed - len(request.blocks))
        request.blocks.extend(new_blocks)
        tb.block_allocated(request, pool, new_blocks)


def _prefill_chunk(
    tb: TraceBuilder,
    leader: Req,
    pool: BlockPool,
    running: list[Req],
    waiting: list[Req],
) -> bool:
    """Cache one prompt chunk for the leader; return True when its prompt is fully cached."""
    start = leader.cached
    end = min(leader.prompt_len, start + PREFILL_CHUNK)
    # Before the chunk write, ensure the pool can cover its growth — preempting LIFO victims if
    # a fresh prompt's prefill would otherwise exhaust the pool (engine order: room first).
    _make_room(tb, _blocks_to_grow(leader, end), [leader], running, waiting, pool)
    tb.emit(
        "prefill_chunk_started",
        request_id=leader.request_id,
        start_pos=start,
        end_pos=end,
        total_prompt_tokens=leader.prompt_len,
    )
    _grow(tb, leader, pool, end)
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


def _prefill(
    tb: TraceBuilder,
    to_prefill: list[Req],
    pool: BlockPool,
    running: list[Req],
    waiting: list[Req],
) -> int:
    """Advance prefill for unstarted requests; prefix siblings share the leader's cache."""
    tokens_emitted = 0
    handled: set[str] = set()
    for request in to_prefill:
        if request.request_id in handled:
            continue
        if request not in running:
            continue  # preempted out while making room for an earlier prefill this step
        group = (
            [r for r in to_prefill if r.group == request.group]
            if request.group is not None
            else [request]
        )
        handled.update(r.request_id for r in group)

        leader = group[0]
        if not _prefill_chunk(tb, leader, pool, running, waiting):
            continue

        # Prompt fully cached: siblings fork the leader's cache by retaining its physical
        # blocks (a refcount bump, never a new allocation), and every group member samples
        # its first token from the prefill logits. The prefill-sampled token is written on
        # the next decode step, so it does not grow the cache here.
        tokens = []
        for member in group:
            member.prefilled = True
            member.cached = leader.prompt_len
            member.length = leader.length
            if member is not leader:
                member.blocks = list(leader.blocks)
                pool.retain(member.blocks)
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


def _resume(
    tb: TraceBuilder,
    to_resume: list[Req],
    pool: BlockPool,
    running: list[Req],
    waiting: list[Req],
) -> None:
    """Rebuild each preempted request's KV by recompute before it decodes again.

    Recompute, not swap: the request kept its generated tokens, so re-prefilling
    ``prompt + generated[:-1]`` reconstructs exactly the evicted state. No token is sampled —
    its generated ids already decide what it decodes next. Replayed prefill chunks (same shape
    as a first prefill) are what makes the resume visible in the trace.
    """
    for request in list(to_resume):
        if request not in running:
            continue  # re-preempted while making room for an earlier resume
        target = request.resume_length
        while request.cached < target:
            start = request.cached
            end = min(target, start + PREFILL_CHUNK)
            _make_room(tb, _blocks_to_grow(request, end), [request], running, waiting, pool)
            tb.emit(
                "prefill_chunk_started",
                request_id=request.request_id,
                start_pos=start,
                end_pos=end,
                total_prompt_tokens=target,
            )
            _grow(tb, request, pool, end)
            request.cached = end
            tb.emit(
                "prefill_chunk_progress",
                request_id=request.request_id,
                start_pos=start,
                end_pos=end,
                cached_tokens=end,
                total_prompt_tokens=target,
                completed=end == target,
            )
        request.prefilled = True
        tb.resumed(request, pool)


def _decode(
    tb: TraceBuilder,
    to_decode: list[Req],
    pool: BlockPool,
    running: list[Req],
    waiting: list[Req],
) -> int:
    """One batched decode step advancing every already-ready request by a token.

    ``decode_many`` allocates for the whole batch in one call, so room must cover the batch's
    *total* growth. We preempt LIFO victims (possibly batch members, which then drop out) until
    the pool can satisfy it, then grow and emit one token per surviving request.
    """
    survivors = [r for r in to_decode if r.prefilled and r in running]
    while True:
        survivors = [r for r in survivors if r.prefilled and r in running]
        demand = sum(_blocks_to_grow(r, r.length + 1) for r in survivors)
        if demand <= pool.free:
            break
        _make_room(tb, demand, [], running, waiting, pool)
    if not survivors:
        return 0
    for request in survivors:
        _grow(tb, request, pool, request.length + 1)
    tokens = [request.emit_token() for request in survivors]
    tb.emit(
        "decode_step",
        request_ids=[request.request_id for request in survivors],
        batch_size=len(survivors),
        token_ids=tokens,
        tokens_emitted=len(tokens),
        token_source="decode",
    )
    return len(tokens)


def build_trace_jsonl() -> str:
    """Run the synthetic scenario and return its schema-v3 JSONL.

    Six requests against a deliberately tight pool, under the **preemption** discipline:
    admission reserves only each request's current footprint, so the pool over-commits and is
    genuinely exhausted as requests decode. When a running request then needs a block, the
    engine evicts the most-recently-admitted one (LIFO) — freeing its KV honestly and keeping
    its tokens — then later resumes it by recompute, replaying its prefill. The scenario stays
    rich: a long chunked prefill (``code-gen``), two requests sharing one prompt
    (``sample-a``/``sample-b``), continuous-batching churn, a KV wall that fills and drains, and
    now at least one real preemption + resume cycle. Every request still finishes with all its
    tokens, and the block lifecycle stays honest throughout.
    """
    tb = TraceBuilder()
    pool = BlockPool(NUM_BLOCKS)
    waiting = [
        Req("sample-a", prompt_len=4, max_new=5, base_token=36, group="shared prompt"),
        Req("sample-b", prompt_len=4, max_new=5, base_token=36, group="shared prompt"),
        Req("code-gen", prompt_len=12, max_new=8, base_token=37),
        Req("summarize", prompt_len=7, max_new=4, base_token=33),
        Req("chat-quick", prompt_len=2, max_new=6, base_token=26),
        Req("translate", prompt_len=9, max_new=6, base_token=32),
    ]
    running: list[Req] = []
    step = 0

    while waiting or running:
        tb.step = step
        _admit(tb, waiting, running, pool)
        tb.batch_changed(len(running), len(waiting))

        to_resume = [r for r in running if not r.prefilled and r.generated]
        to_prefill = [r for r in running if not r.prefilled and not r.generated]
        _resume(tb, to_resume, pool, running, waiting)
        prefill_tokens = _prefill(tb, to_prefill, pool, running, waiting)
        to_decode = [r for r in running if r.prefilled]
        decode_tokens = _decode(tb, to_decode, pool, running, waiting)
        tokens_this_step = prefill_tokens + decode_tokens

        for request in [r for r in running if r.finished]:
            tb.emit(
                "request_finished",
                request_id=request.request_id,
                token_ids=request.generated,
                generated_tokens=len(request.generated),
                reason="length",
            )
            # Free the request's physical blocks; only last-owner blocks return to the pool,
            # so a shared prefix block held by a still-running sibling is not reported freed.
            returned = pool.free_blocks(request.blocks)
            tb.block_freed(request, pool, returned)
            request.blocks = []
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
