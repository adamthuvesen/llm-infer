"""Prompt prefill and prefix sharing for the continuous-batching decode loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llm_infer.scheduler.scheduler import blocks_for_footprint, max_blocks_for
from llm_infer.serving.engine_contract import EngineMixinHost
from llm_infer.serving.request import Request

if TYPE_CHECKING:
    from llm_infer.serving.engine import StepResult


class EnginePrefillMixin(EngineMixinHost):
    def _reserved_blocks(self, request: Request) -> int:
        """Blocks the admit event reports reserved — footprint under preemption, else worst case."""
        if self.preemption:
            return blocks_for_footprint(request, self.scheduler.block_size)
        return max_blocks_for(request, self.scheduler.block_size)

    def _prefill_requests(self, requests: list[Request], result: StepResult) -> None:
        """Prefill unstarted requests, sharing prompt blocks for declared sibling groups."""
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

    def _prefill_one(self, request: Request, result: StepResult) -> None:
        logits = self._cache_prompt_chunk(request, result)
        if logits is None:
            return
        request.prefilled = True
        with self._record_time("sampling"):
            token = self._sample_one(logits, request)
        self._record(request, token, self._eos_flags(token.reshape(1), [request])[0], result)
        self._trace_decode_step([request], [token], token_source="prefill")
        self._release_finished_in([request])

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
        eos_flags = self._eos_flags(torch.stack(tokens), members)
        for request, token, is_eos in zip(members, tokens, eos_flags, strict=True):
            self._record(request, token, is_eos, result)
        self._trace_decode_step(members, tokens, token_source="prefill")
        self._release_finished_in(members)

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
