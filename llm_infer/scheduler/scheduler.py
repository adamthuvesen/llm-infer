"""Continuous-batching admission, with optional preemption under memory pressure.

At decode-step boundaries finished requests leave and queued requests enter. Two admission
disciplines share this class, selected by ``preemption``:

* **Reserve (default).** A request is admitted only if the worst-case blocks it could ever
  need fit alongside everything already running, so a running request never runs out of
  blocks mid-decode and never has to be evicted. Serving may execute prefill in chunks, but
  this still reserves the full prompt-plus-decode budget up front. ``OutOfBlocksError`` then
  cannot fire — the allocator never goes dry.

* **Preempt.** Admission instead reserves only the request's *current* footprint (the blocks
  its prompt needs right now), deliberately over-committing the pool so it can actually be
  exhausted. When a running request then needs a block the pool cannot give, the engine
  preempts a victim — frees its KV and requeues it to be rebuilt later by recompute — until
  the block is available. This is the canonical eviction technique; it lets the engine pack
  more requests than the reservation discipline admits, at the cost of recompute on resume.

Either way physical block allocation stays lazy (the block table pulls blocks as the
sequence grows); the scheduler tracks a *budget* against the pool size.
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


def blocks_for_footprint(request: Request, block_size: int) -> int:
    """Blocks the request's KV occupies *right now* — its rebuild footprint on (re)admit.

    The recompute resume path rebuilds KV for the prompt plus every already-generated
    token, so a resumed request's immediate footprint is ``len(prompt) + len(generated)``
    positions; a fresh request has no generated tokens, so this is just its prompt.
    """
    positions = len(request.prompt_ids) + len(request.generated)
    return -(-positions // block_size)  # ceil division


class Scheduler:
    """Holds the waiting queue and the running set, admitting within a block budget.

    With ``preemption`` off this is the strict reservation scheduler: admission reserves
    each request's worst-case block budget. With it on, admission reserves only the current
    footprint and may over-commit the pool, leaving the engine to preempt under pressure.
    """

    def __init__(self, num_blocks: int, block_size: int, *, preemption: bool = False) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1; got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.preemption = preemption
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.committed_blocks = 0

    def add(self, request: Request) -> None:
        """Queue a request. Rejects one too large to ever fit the whole pool.

        The worst-case-fits check is the forward-progress guarantee in *both* modes: a
        request that fits in the empty pool alone can always be made to fit by preempting
        every other request, so the block-needer can never livelock.
        """
        need = max_blocks_for(request, self.block_size)
        if need > self.num_blocks:
            raise ValueError(
                f"request {request.request_id!r} needs up to {need} blocks but the pool "
                f"holds {self.num_blocks}; it can never be admitted"
            )
        self.waiting.append(request)

    def requeue(self, request: Request) -> None:
        """Return a preempted request to the *front* of the queue for prompt re-admission.

        Front, not back: a preempted request has already made progress, so it should
        reclaim a slot ahead of never-started newcomers rather than starve behind them.
        """
        self.running.remove(request)
        self.committed_blocks -= blocks_for_footprint(request, self.block_size)
        self.waiting.appendleft(request)

    def admit(self, *, free_blocks: int | None = None) -> list[Request]:
        """Move waiting requests into the running set while their budget fits.

        Reserve mode reserves the worst-case budget; preempt mode reserves only the current
        footprint and additionally requires that footprint to fit in the blocks actually free
        right now (``free_blocks``), so admission tracks real occupancy and the engine never
        admits a prompt it cannot even prefill. FIFO with head-of-line blocking in both modes:
        admission stops at the first request that does not fit, keeping order predictable.
        """
        if self.preemption and free_blocks is None:
            raise ValueError("preemption admission needs the current free-block count")
        admitted: list[Request] = []
        remaining_free = free_blocks if free_blocks is not None else 0
        while self.waiting:
            request = self.waiting[0]
            if self.preemption:
                need = blocks_for_footprint(request, self.block_size)
                if need > remaining_free:
                    break
                remaining_free -= need
            else:
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
        budget = (
            blocks_for_footprint(request, self.block_size)
            if self.preemption
            else max_blocks_for(request, self.block_size)
        )
        self.committed_blocks -= budget

    def preemption_victim(self, exclude: Request | None = None) -> Request | None:
        """The request to evict under pressure: most-recently-admitted, never ``exclude``.

        LIFO by admission order. The newest running request has decoded the fewest tokens, so
        freeing and later recomputing it wastes the least work. ``exclude`` is the request that
        needs the block (when a single request drives the pressure), which must not evict itself;
        pass ``None`` when the whole running batch is the block-needer.
        """
        for request in reversed(self.running):
            if request is not exclude:
                return request
        return None

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)
