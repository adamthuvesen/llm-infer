"""The block allocator: a refcounted free list over physical KV-cache blocks.

The cache is paged — token K/V is stored in fixed-size blocks, and a request owns a
list of physical block ids (its block table). The allocator is the single owner of which
blocks are free and which live blocks are shared by more than one table. Nothing here knows
about tensors — it is pure bookkeeping over integer ids.
"""

from __future__ import annotations


class OutOfBlocksError(RuntimeError):
    """Raised when more blocks are requested than the pool has free.

    Under the v1 scheduler this should never fire: admission reserves a block budget
    so a running request can always grow. If it fires, an invariant broke (the
    scheduler admitted beyond capacity), so it is loud rather than silently dropping
    tokens.
    """


class BlockAllocator:
    """Allocates and frees physical block ids from a fixed pool.

    Freed blocks return to the pool and are re-handed-out (LIFO) to later requests —
    the reallocation path the Phase B vertical slice exercises when one request
    finishes and a queued one is admitted.
    """

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1; got {num_blocks}")
        self.num_blocks = num_blocks
        # Stack of free ids; pop/extend from the end so freed blocks are reused soon.
        self._free: list[int] = list(range(num_blocks))
        self._refcounts: list[int] = [0] * num_blocks

    @property
    def num_free(self) -> int:
        return len(self._free)

    def allocate(self, count: int = 1) -> list[int]:
        """Hand out ``count`` free block ids, removing them from the pool."""
        if count < 1:
            raise ValueError(f"count must be >= 1; got {count}")
        if count > len(self._free):
            raise OutOfBlocksError(
                f"requested {count} blocks but only {len(self._free)} free "
                f"(pool size {self.num_blocks})"
            )
        out = self._free[-count:]
        del self._free[-count:]
        for block in out:
            self._refcounts[block] = 1
        return out

    def retain(self, blocks: list[int]) -> None:
        """Increment refcounts for blocks added to another block table."""
        self._validate_live_blocks(blocks, action="retain")
        for block in blocks:
            self._refcounts[block] += 1

    def free(self, blocks: list[int]) -> None:
        """Release table references, returning only last-owner blocks to the pool.

        Validates the *entire* batch before mutating ``_free`` — out-of-range ids,
        already-free ids, and duplicates within the argument all raise before any refcount
        changes. A failed free therefore leaves the free list (and the next allocation)
        untouched, never half-applied: a partially-applied release could return a still-owned
        block to the pool and alias another request's KV pages.
        """
        self._validate_live_blocks(blocks, action="free")
        for block in blocks:
            self._refcounts[block] -= 1
            if self._refcounts[block] == 0:
                self._free.append(block)

    def refcount(self, block: int) -> int:
        """Current owner count for one physical block."""
        if not 0 <= block < self.num_blocks:
            raise ValueError(f"block id {block} out of range [0, {self.num_blocks})")
        return self._refcounts[block]

    def _validate_live_blocks(self, blocks: list[int], *, action: str) -> None:
        seen: set[int] = set()
        for block in blocks:
            if not 0 <= block < self.num_blocks:
                raise ValueError(f"block id {block} out of range [0, {self.num_blocks})")
            if self._refcounts[block] == 0:
                if action == "free":
                    raise ValueError(f"double free of block id {block}")
                raise ValueError(f"cannot {action} free block id {block}")
            if block in seen:
                raise ValueError(f"duplicate block id {block} in {action}() argument")
            seen.add(block)
