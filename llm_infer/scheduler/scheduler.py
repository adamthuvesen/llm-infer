"""Continuous-batching admission over a fixed block budget.

At decode-step boundaries finished requests leave and queued requests enter. Admission is
a reservation scheme — a request is admitted only if the worst-case blocks it could ever
need fit alongside everything already running, so a running request never runs out of
blocks mid-decode and never has to be evicted. Serving may execute prefill in chunks, but
this scheduler still reserves the request's full prompt-plus-decode budget up front.

Physical block allocation stays lazy (the block table pulls blocks as the sequence
grows); this scheduler only reserves a *budget* against the pool size, which upper-
bounds the lazy allocation and so keeps the allocator from ever going dry.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_infer.serving.request import Request


def max_blocks_for(request: Request, block_size: int) -> int:
    """Worst-case blocks a request can occupy: prompt plus its full decode budget.

    A request caches ``prompt`` positions on prefill, then one more per decode step.
    The first generated token comes from prefill, so the most positions it can ever
    cache is ``len(prompt) + max_new_tokens - 1``.
    """
    max_positions = len(request.prompt_ids) + request.max_new_tokens - 1
    return -(-max_positions // block_size)  # ceil division


class Scheduler:
    """Holds the waiting queue and the running set, admitting within a block budget."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1; got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.committed_blocks = 0

    def add(self, request: Request) -> None:
        """Queue a request. Rejects one too large to ever fit the whole pool."""
        need = max_blocks_for(request, self.block_size)
        if need > self.num_blocks:
            raise ValueError(
                f"request {request.request_id!r} needs up to {need} blocks but the pool "
                f"holds {self.num_blocks}; it can never be admitted"
            )
        self.waiting.append(request)

    def admit(self) -> list[Request]:
        """Move waiting requests into the running set while their budget fits.

        FIFO and head-of-line blocking: stops at the first request that does not fit,
        rather than skipping ahead, to keep admission order predictable.
        """
        admitted: list[Request] = []
        while self.waiting:
            request = self.waiting[0]
            need = max_blocks_for(request, self.block_size)
            if self.committed_blocks + need > self.num_blocks:
                break
            self.waiting.popleft()
            self.running.append(request)
            self.committed_blocks += need
            admitted.append(request)
        return admitted

    def release(self, request: Request) -> None:
        """Drop a finished request from the running set and return its block budget."""
        self.running.remove(request)
        self.committed_blocks -= max_blocks_for(request, self.block_size)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
