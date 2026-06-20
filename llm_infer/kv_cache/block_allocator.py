"""The block allocator: a free list over a fixed pool of physical KV-cache blocks.

The cache is paged — token K/V is stored in fixed-size blocks, and a request owns a
list of physical block ids (its block table). The allocator is the single owner of
which blocks are free; it hands blocks out on prefill/decode growth and takes them
back when a request finishes, so a finished request's blocks are reused by the next
one. Nothing here knows about tensors — it is pure bookkeeping over integer ids.
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
        return out

    def free(self, blocks: list[int]) -> None:
        """Return blocks to the pool. Rejects out-of-range ids and double-frees loudly."""
        free_set = set(self._free)
        for block in blocks:
            if not 0 <= block < self.num_blocks:
                raise ValueError(f"block id {block} out of range [0, {self.num_blocks})")
            if block in free_set:
                raise ValueError(f"double free of block id {block}")
            free_set.add(block)
            self._free.append(block)
