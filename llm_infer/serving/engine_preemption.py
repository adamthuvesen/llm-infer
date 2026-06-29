"""KV preemption and recompute for the continuous-batching decode loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_infer.kv_cache.block_allocator import OutOfBlocksError
from llm_infer.serving.request import Request

if TYPE_CHECKING:
    from llm_infer.serving.engine import StepResult


class EnginePreemptionMixin:
    # --- preemption (recompute) ---------------------------------------------------------
    #
    # Trigger points: the two places a running request needs the allocator to hand out a
    # *new* physical block. Decode appends one token per step (a new block only when the
    # current one fills); a prefill chunk caches up to ``chunk_size`` new positions. Before
    # either forward we ensure the free pool can cover that growth, preempting victims until
    # it can. Reserving here — at the engine boundary, before the model touches the cache —
    # keeps recovery clean: no forward runs half-done, so a preempted-then-resumed request
    # rebuilds from a pristine state and stays token-exact.

    def _blocks_to_grow(self, request: Request, new_tokens: int) -> int:
        """How many *new* physical blocks ``request`` must pull to cache ``new_tokens`` more.

        Delegates to the cache's dry-run cost so the count includes copy-on-write: a request
        whose last block is a prefix-shared partial block copies it private on append, which
        pulls a block the bare capacity-growth math misses (and would otherwise OOM mid-write).
        """
        table = request.block_table
        if table is None:
            # Fresh request: no blocks yet and nothing shared, so just capacity growth from empty.
            return -(-new_tokens // self.scheduler.block_size)  # ceil division
        return self.cache.append_cost(table, new_tokens)

    def _ensure_pool_room(self, request: Request, new_tokens: int) -> None:
        """Free enough blocks for ``request`` to grow by ``new_tokens``, preempting LIFO victims.

        Forward-progress invariant: a request fits in the empty pool alone (admission rejects
        any that does not), so once every *other* running request is preempted ``request`` can
        always grow — the loop cannot spin forever. ``request`` is the block-needer and is passed
        as ``exclude`` so it never evicts itself; we stop once the free pool covers the growth.
        """
        need = self._blocks_to_grow(request, new_tokens)
        while self.cache.allocator.num_free < need:
            victim = self.scheduler.preemption_victim(exclude=request)
            if victim is None:
                raise OutOfBlocksError(
                    f"request {request.request_id!r} needs {need} block(s) to grow but the "
                    f"pool has {self.cache.allocator.num_free} free and no other request to "
                    "preempt — the forward-progress invariant was violated"
                )
            self._preempt(victim)

    def _preempt(self, victim: Request) -> None:
        """Evict ``victim`` by recompute: free its KV honestly, keep its tokens, requeue it."""
        self.preemption_count += 1
        freed_blocks = victim.block_table.num_blocks if victim.block_table is not None else 0
        if victim.block_table is not None:
            # Fires honest block_freed at the allocator boundary.
            self._free_block_table(victim.block_table)
        victim.reset_for_recompute()
        self.scheduler.requeue(victim)
        self._emit_trace(
            "request_preempted",
            request_id=victim.request_id,
            preempt_reason="kv_pressure",
            block_count=freed_blocks,
            generated_tokens=len(victim.generated),
            pool_used=self.cache.allocator.num_used,
            pool_free=self.cache.allocator.num_free,
        )

    def _is_running(self, request: Request) -> bool:
        """Whether ``request`` is in the running set *right now*, tested by identity.

        Prefilling or resuming an earlier request can preempt a later candidate via
        ``_ensure_pool_room``, moving it to ``waiting`` partway through the loop. Both loops
        therefore re-check membership live rather than against a set snapshotted once before the
        loop: a stale snapshot would let an already-evicted request be prefilled while it sits in
        ``waiting``, double-allocating its KV and breaking forward progress.
        """
        return any(candidate is request for candidate in self.scheduler.running)

    def _resume_requests(self, requests: list[Request], result: StepResult) -> None:
        """Rebuild each preempted request's KV by recompute before it decodes again.

        Recompute, not swap: the request kept its generated tokens through preemption, so a
        fresh prefill over ``recompute_prompt_ids`` reconstructs exactly the KV state it was
        evicted in. No token is sampled — the request already owns its generated ids — so on the
        next step it decodes its next token, identical to the uninterrupted run.
        """
        for request in requests:
            # A queued resume can be preempted again while making room for an earlier one this
            # step; check membership live (see _is_running) so a request evicted mid-loop is
            # skipped and re-picked once it is re-admitted, never resumed out of the waiting queue.
            if not self._is_running(request):
                continue
            self._recompute_prefill(request)
            self._emit_trace(
                "request_resumed",
                request_id=request.request_id,
                prompt_tokens=len(request.prompt_ids),
                generated_tokens=len(request.generated),
                cached_tokens=request.prompt_cached_tokens,
                pool_used=self.cache.allocator.num_used,
                pool_free=self.cache.allocator.num_free,
            )

    def _recompute_prefill(self, request: Request) -> None:
        """Re-prefill the request's pre-feed sequence into a fresh table, sampling nothing.

        The sequence is ``prompt + generated[:-1]`` (see ``Request.recompute_prompt_ids``): the
        last generated token is *not* written here, it is re-fed by the resuming decode at its
        original position. Walks the sequence in the same bounded chunks as a normal prefill (so
        a long resume shares the loop and can itself be re-preempted between chunks), rebuilding
        KV at the original absolute positions. The final logits are discarded — the request's
        generated tokens already determine what it decodes next.
        """
        sequence = request.recompute_prompt_ids
        if request.block_table is None:
            request.block_table = self.cache.new_request()
            request.block_table.owner = request.request_id

        total = len(sequence)
        chunk_size = self.prefill_chunk_size or total
        while request.prompt_cached_tokens < total:
            start = request.prompt_cached_tokens
            step = min(chunk_size, total - start)
            self._ensure_pool_room(request, step)
            self._emit_trace(
                "prefill_chunk_started",
                request_id=request.request_id,
                start_pos=start,
                end_pos=start + step,
                total_prompt_tokens=total,
            )
            with self._record_time("prefill"):
                self._write_recompute_chunk(request, sequence, start, step)
            request.prompt_cached_tokens = start + step
            self._emit_trace(
                "prefill_chunk_progress",
                request_id=request.request_id,
                start_pos=start,
                end_pos=start + step,
                cached_tokens=request.prompt_cached_tokens,
                total_prompt_tokens=total,
                completed=request.prompt_cached_tokens == total,
            )
        request.prefilled = True

    def _write_recompute_chunk(
        self, request: Request, sequence: list[int], start: int, count: int
    ) -> None:
        """Write KV for ``sequence[start:start+count]`` via the model's cached prefill path."""
        if start == 0 and count == len(sequence):
            self.model.prefill(sequence, self.cache, request.block_table)
            return
        self.model.prefill_chunk(
            sequence,
            self.cache,
            request.block_table,
            start_pos=start,
            chunk_size=count,
        )
