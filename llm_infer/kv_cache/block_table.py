"""A request's block table: maps its logical token positions to physical cache slots.

Each request owns one of these. Logical position ``pos`` lives in the request's
``pos // block_size``-th block at offset ``pos % block_size``; the table grows by
pulling fresh blocks from the shared allocator as the sequence lengthens, and returns
them all when the request finishes. ``length`` is the number of tokens currently
cached (the next token to write goes at position ``length``).
"""

from __future__ import annotations

from llm_infer.kv_cache.block_allocator import BlockAllocator


class BlockTable:
    """Per-request logical-position → physical-slot mapping over paged blocks."""

    def __init__(self, allocator: BlockAllocator, block_size: int) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        self.allocator = allocator
        self.block_size = block_size
        self.blocks: list[int] = []
        self.length = 0
        # Set by the engine to the owning request id so allocator pool events can be
        # attributed in the trace. Pure bookkeeping — the table still owns its blocks.
        self.owner: str | None = None

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    @property
    def capacity(self) -> int:
        """How many tokens the currently-allocated blocks can hold."""
        return len(self.blocks) * self.block_size

    def reserve(self, num_new_tokens: int) -> None:
        """Allocate blocks so the table can hold ``length + num_new_tokens`` tokens."""
        if num_new_tokens < 0:
            raise ValueError(f"num_new_tokens must be >= 0; got {num_new_tokens}")
        target = self.length + num_new_tokens
        needed_blocks = -(-target // self.block_size)  # ceil division
        if needed_blocks > len(self.blocks):
            self.blocks.extend(
                self.allocator.allocate(needed_blocks - len(self.blocks), owner=self.owner)
            )

    def fork_shared(self) -> BlockTable:
        """Create another table pointing at the same physical blocks.

        Used after one sibling request has prefetched a prompt. The fork starts at the same
        logical length and shares every current block by refcount; generated-token writes make
        the last partial block private before mutation.
        """
        fork = BlockTable(self.allocator, self.block_size)
        fork.blocks = list(self.blocks)
        fork.length = self.length
        fork.owner = self.owner
        if fork.blocks:
            self.allocator.retain(fork.blocks)
        return fork

    def physical_slot(self, pos: int) -> int:
        """Flat slot index (``block * block_size + offset``) for logical position ``pos``."""
        if not 0 <= pos < self.capacity:
            raise ValueError(
                f"position {pos} out of allocated range [0, {self.capacity}); reserve first"
            )
        return self.blocks[pos // self.block_size] * self.block_size + pos % self.block_size

    def physical_slots(self, start: int, count: int) -> list[int]:
        """Flat slot indices for ``count`` consecutive positions starting at ``start``."""
        return [self.physical_slot(pos) for pos in range(start, start + count)]

    def free(self) -> None:
        """Return all blocks to the allocator and reset to empty."""
        if self.blocks:
            self.allocator.free(self.blocks, owner=self.owner)
        self.blocks = []
        self.length = 0
