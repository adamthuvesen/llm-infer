"""Prompt prefill and prefix sharing for the continuous-batching decode loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llm_infer.model.interface import BatchedPrefillBackend
from llm_infer.scheduler.scheduler import blocks_for_footprint, max_blocks_for
from llm_infer.serving.request import Request

if TYPE_CHECKING:
    from llm_infer.serving.engine import StepResult


class EnginePrefillMixin:
    def _reserved_blocks(self, request: Request) -> int:
        """Blocks the admit event reports reserved — footprint under preemption, else worst case."""
        if self.preemption:
            return blocks_for_footprint(request, self.scheduler.block_size)
        return max_blocks_for(request, self.scheduler.block_size)

    def _prefill_requests(self, requests: list[Request], result: StepResult) -> None:
        """Prefill unstarted requests, sharing prompt blocks for declared sibling groups."""
        if self.prefix_cache is not None:
            for request in requests:
                self._apply_prefix_cache_hit(request)
        if self._can_prefill_many(requests):
            self._prefill_many(requests, result)
            return

        handled: set[str] = set()
        for request in requests:
            if request.request_id in handled:
                continue
            # A request can be preempted out of the running set while we make pool room for an
            # earlier one this step; check membership live (see _is_running) so it is skipped and
            # re-prefills once re-admitted, never prefilled while sitting in the waiting queue.
            if self.preemption and not self._is_running(request):
                continue
            if request.prefix_group_id is None:
                self._prefill_one(request, result)
                handled.add(request.request_id)
                continue

            group = [
                candidate
                for candidate in self._live_running(requests)
                if candidate.prefix_group_id == request.prefix_group_id
            ]
            if len(group) == 1:
                self._prefill_one(request, result)
                handled.add(request.request_id)
                continue
            self._prefill_shared_group(group, result)
            handled.update(candidate.request_id for candidate in group)

    def _apply_prefix_cache_hit(self, request: Request) -> None:
        """Seed a fresh request's block table from the longest cached block-aligned prefix.

        Runs before prefill for a plain (non-group, un-started) request when the cross-turn
        prefix cache is on. A hit builds the request's table over the reused blocks and advances
        ``prompt_cached_tokens`` to the hit length, so ``_cache_prompt_chunk`` then prefills only
        the uncached tail via ``prefill_chunk(start_pos=k, ...)``. Because hits are block-aligned,
        the reused blocks stay read-only and every fresh write lands in a private block.
        """
        if self.prefix_cache is None:
            return
        if request.block_table is not None or request.prefix_group_id is not None:
            return
        hit = self.prefix_cache.lookup(request.prompt_ids)
        if hit is None:
            return
        table = self.cache.new_request()
        table.owner = request.request_id
        table.blocks = list(hit.block_ids)
        table.length = hit.tokens
        request.block_table = table
        request.prompt_cached_tokens = hit.tokens

    def _can_prefill_many(self, requests: list[Request]) -> bool:
        return (
            self.batched_prefill
            and self.prefix_cache is None
            and len(requests) >= 2
            and self.capabilities.batched_prefill
            and isinstance(self.model, BatchedPrefillBackend)
            and self.prefill_chunk_size is None
            and not self.preemption
            and all(request.prefix_group_id is None for request in requests)
        )

    def _prefill_many(self, requests: list[Request], result: StepResult) -> None:
        tables = []
        for request in requests:
            if request.block_table is None:
                request.block_table = self.cache.new_request()
                request.block_table.owner = request.request_id
            if request.block_table.length != 0 or request.prompt_cached_tokens != 0:
                raise ValueError(
                    f"request {request.request_id!r} must have an empty cache for batched prefill"
                )
            end_pos = len(request.prompt_ids)
            result.prefill_chunks[request.request_id] = (0, end_pos)
            self._trace_prefill_chunk_started(
                request_id=request.request_id,
                start_pos=0,
                end_pos=end_pos,
                total_prompt_tokens=end_pos,
            )
            tables.append(request.block_table)

        with self._record_time("prefill"):
            logits = self.model.prefill_many(
                [request.prompt_ids for request in requests], self.cache, tables
            )

        for request in requests:
            end_pos = len(request.prompt_ids)
            request.prompt_cached_tokens = end_pos
            request.prefilled = True
            self._trace_prefill_chunk_progress(
                request_id=request.request_id,
                start_pos=0,
                end_pos=end_pos,
                cached_tokens=end_pos,
                total_prompt_tokens=end_pos,
                completed=True,
            )

        with self._record_time("sampling"):
            tokens = self._sample_rows(logits, requests)
        token_rows = list(tokens.unbind())
        eos_flags, host_ids = self._eos_flags_and_ids(tokens, requests)
        for request, token, is_eos, host_id in zip(
            requests, token_rows, eos_flags, host_ids, strict=True
        ):
            self._record(request, token, is_eos, result, host_token=host_id)
        self._trace_decode_step(requests, token_rows, token_source="prefill")
        self._release_finished_in(requests, result)

    def _prefill_one(self, request: Request, result: StepResult) -> None:
        logits = self._cache_prompt_chunk(request, result)
        if logits is None:
            return
        request.prefilled = True
        with self._record_time("sampling"):
            token = self._sample_one(logits, request)
        eos_flags, host_ids = self._eos_flags_and_ids(token.reshape(1), [request])
        self._record(request, token, eos_flags[0], result, host_token=host_ids[0])
        self._trace_decode_step([request], [token], token_source="prefill")
        self._release_finished_in([request], result)

    def _prefill_shared_group(self, requests: list[Request], result: StepResult) -> None:
        prompt_ids = requests[0].prompt_ids
        if any(request.prompt_ids != prompt_ids for request in requests):
            raise ValueError(
                f"prefix group {requests[0].prefix_group_id!r} contains different prompts"
            )

        leaders = [request for request in requests if request.block_table is not None]
        if len(leaders) > 1:
            raise ValueError(
                f"prefix group {requests[0].prefix_group_id!r} has multiple active leaders"
            )
        leader = leaders[0] if leaders else requests[0]
        logits = self._cache_prompt_chunk(leader, result)
        if logits is None:
            return

        # Leader prefill can preempt a sibling to make pool room (under preemption): that sibling
        # has been reset and returned to the waiting queue. Re-filter the group to the still-running
        # members before forking/sampling — forking onto a preempted sibling would resurrect it
        # out of the waiting queue with a live block table and a sampled token. It re-prefills
        # next step instead. The leader is never its own victim, so it always survives.
        members = [leader, *(r for r in self._live_running(requests) if r is not leader)]

        leader.prefilled = True
        leader.prompt_cached_tokens = len(prompt_ids)
        for request in members:
            if request is leader:
                continue
            request.block_table = self.cache.fork_request(leader.block_table)
            request.block_table.owner = request.request_id
            request.prompt_cached_tokens = leader.prompt_cached_tokens
            request.prefilled = True

        with self._record_time("sampling"):
            tokens = [self._sample_one(logits, request) for request in members]
        eos_flags, host_ids = self._eos_flags_and_ids(torch.stack(tokens), members)
        for request, token, is_eos, host_id in zip(
            members, tokens, eos_flags, host_ids, strict=True
        ):
            self._record(request, token, is_eos, result, host_token=host_id)
        self._trace_decode_step(members, tokens, token_source="prefill")
        self._release_finished_in(members, result)

    def _cache_prompt_chunk(self, request: Request, result: StepResult) -> torch.Tensor | None:
        """Cache one prompt chunk and return final-prompt logits when ready to sample."""
        if request.block_table is None:
            request.block_table = self.cache.new_request()
            request.block_table.owner = request.request_id

        start_pos = request.prompt_cached_tokens
        if request.block_table.length != start_pos:
            raise ValueError(
                f"request {request.request_id!r} block table length "
                f"{request.block_table.length} != cached prompt length {start_pos}"
            )
        remaining = len(request.prompt_ids) - start_pos
        if remaining < 1:
            raise ValueError(f"request {request.request_id!r} has no prompt tokens left")

        chunk_size = self.prefill_chunk_size or len(request.prompt_ids)
        chunk_size = min(chunk_size, remaining)
        end_pos = start_pos + chunk_size
        if self.preemption:
            # A fresh prompt's prefill can also exhaust the pool — preempt LIFO victims so the
            # chunk's blocks are available before the model touches the cache.
            self._ensure_pool_room(request, new_tokens=chunk_size)
        result.prefill_chunks[request.request_id] = (start_pos, end_pos)
        self._trace_prefill_chunk_started(
            request_id=request.request_id,
            start_pos=start_pos,
            end_pos=end_pos,
            total_prompt_tokens=len(request.prompt_ids),
        )

        with self._record_time("prefill"):
            if start_pos == 0 and end_pos == len(request.prompt_ids):
                logits = self.model.prefill(request.prompt_ids, self.cache, request.block_table)
            else:
                logits = self.model.prefill_chunk(
                    request.prompt_ids,
                    self.cache,
                    request.block_table,
                    start_pos=start_pos,
                    chunk_size=chunk_size,
                )
        request.prompt_cached_tokens = end_pos
        self._trace_prefill_chunk_progress(
            request_id=request.request_id,
            start_pos=start_pos,
            end_pos=end_pos,
            cached_tokens=end_pos,
            total_prompt_tokens=len(request.prompt_ids),
            completed=end_pos == len(request.prompt_ids),
        )
        if end_pos < len(request.prompt_ids):
            return None
        return logits
