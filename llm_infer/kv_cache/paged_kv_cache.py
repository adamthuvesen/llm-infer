"""The paged KV-cache store: the K/V tensors plus scatter/gather over block tables.

Layout is one unified page tensor shaped
``(num_layers, num_blocks, 2, block_size, num_kv_heads, head_dim)``. K/V are stored
*after* RoPE but *before* GQA expansion (one row per KV head, not per query head) —
each token's rotation is fixed by its absolute position, so it is computed once at
write time and never re-rotated. The K and V views preserve the public API, while native
paged-attention kernels read the unified layout directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llm_infer.kv_cache.block_allocator import BlockAllocator
from llm_infer.kv_cache.block_table import BlockTable


@dataclass(frozen=True)
class KVPagePlan:
    """Page-table metadata for a batched decode step."""

    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor
    page_size: int


@dataclass(frozen=True)
class KVReadPlan:
    """Layer-independent metadata for one batched decode step.

    ``idx`` is absent when a native paged backend reads the cache directly.
    """

    idx: torch.Tensor | None
    cu_seqlens: torch.Tensor | None
    lengths: list[int]
    max_len: int
    page_plan: KVPagePlan | None = None


class PagedKVCache:
    """Fixed-size paged store of per-token K/V, addressed via block tables."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.allocator = BlockAllocator(num_blocks)
        shape = (num_layers, num_blocks, 2, block_size, num_kv_heads, head_dim)
        self.kv = torch.zeros(shape, dtype=dtype, device=device)
        self.key = self.kv[:, :, 0]
        self.value = self.kv[:, :, 1]

    def new_request(self) -> BlockTable:
        """A fresh, empty block table bound to this cache's allocator and block size."""
        return BlockTable(self.allocator, self.block_size)

    def fork_request(self, table: BlockTable) -> BlockTable:
        """Fork a request's current block table by shared physical-block ownership."""
        return table.fork_shared()

    def append_cost(self, table: BlockTable, count: int) -> int:
        """Physical blocks an append of ``count`` tokens will pull from the pool — a dry run.

        The engine must reserve two costs before a forward writes: **capacity growth** (new blocks
        to extend the table past its current capacity) and **copy-on-write** (a shared partial
        block the append lands in is copied private before mutation; see :meth:`prepare_write`).
        Counting only capacity growth under-reserves when prefix-shared siblings decode — a COW
        append could then find the pool empty mid-write. Allocates nothing; pure accounting.
        """
        if count < 1:
            return 0
        start_pos = table.length
        end_pos = start_pos + count
        needed_blocks = -(-end_pos // self.block_size)  # ceil
        growth = max(0, needed_blocks - len(table.blocks))
        # COW copies only blocks that already exist (freshly grown blocks are private) and are
        # shared (refcount > 1). The append touches the blocks holding positions [start_pos,
        # end_pos); only the already-allocated ones among them can be shared and so be copied.
        first_block = start_pos // self.block_size
        last_existing = min((end_pos - 1) // self.block_size, len(table.blocks) - 1)
        cow = sum(
            1
            for block_index in range(first_block, last_existing + 1)
            if self.allocator.refcount(table.blocks[block_index]) > 1
        )
        return growth + cow

    def prepare_write(self, table: BlockTable, start_pos: int, count: int) -> None:
        """Make append writes safe when a shared prompt ends in a partial block.

        Prefix caching shares prompt blocks by reference. If generation starts inside the last
        prompt block, the first generated-token append would otherwise overwrite that shared
        physical page for every sibling. The only allowed copy-on-write is that append into a
        shared partial block; full prompt blocks are never copied.
        """
        if count < 1:
            raise ValueError(f"count must be >= 1; got {count}")
        end_pos = start_pos + count
        if end_pos > table.capacity:
            raise ValueError(
                f"positions [{start_pos}, {end_pos}) out of allocated range "
                f"[0, {table.capacity}); reserve first"
            )

        first_block = start_pos // self.block_size
        last_block = (end_pos - 1) // self.block_size
        for block_index in range(first_block, last_block + 1):
            block = table.blocks[block_index]
            if self.allocator.refcount(block) == 1:
                continue

            block_start = block_index * self.block_size
            if start_pos != table.length or start_pos == block_start:
                raise ValueError(
                    "cannot write into a shared prompt block; only appending into the "
                    "last partial prompt block may copy-on-write"
                )
            new_block = self.allocator.allocate(1, owner=table.owner)[0]
            self._copy_block(block, new_block)
            table.blocks[block_index] = new_block
            self.allocator.free([block], owner=table.owner)

    def write(
        self,
        table: BlockTable,
        layer: int,
        start_pos: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store K/V for ``n`` consecutive positions starting at ``start_pos``.

        ``key``/``value`` are ``(n, num_kv_heads, head_dim)`` — RoPE already applied,
        GQA expansion not yet. The block table must already cover these positions.
        """
        n = key.shape[0]
        self.prepare_write(table, start_pos, n)
        idx = torch.as_tensor(
            table.physical_slots(start_pos, n), dtype=torch.long, device=key.device
        )
        self._write_slots(layer, idx, key, value)

    def write_many(
        self,
        tables: list[BlockTable],
        layer: int,
        positions: list[int],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store one K/V row per request for a batched decode step.

        ``key``/``value`` are ``(B, num_kv_heads, head_dim)`` and row ``i`` is written
        into ``tables[i]`` at ``positions[i]``.
        """
        if not (len(tables) == len(positions) == key.shape[0] == value.shape[0]):
            raise ValueError(
                f"tables/positions/key/value batch mismatch: "
                f"{len(tables)}, {len(positions)}, {key.shape[0]}, {value.shape[0]}"
            )
        for table, pos in zip(tables, positions, strict=True):
            self.prepare_write(table, pos, 1)
        slots = [table.physical_slot(pos) for table, pos in zip(tables, positions, strict=True)]
        idx = torch.as_tensor(slots, dtype=torch.long, device=key.device)
        self._write_slots(layer, idx, key, value)

    def write_rows(
        self,
        layer: int,
        slots: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Store one K/V row per request at precomputed physical ``slots`` — no table walk.

        The planned decode window computes every write slot once at window open, so per layer
        this is just two indexed stores. The caller owns copy-on-write safety: slots must come
        from unshared tables (``build_decode_window_plan`` refuses shared blocks up front).
        """
        self._write_slots(layer, slots, key, value)

    def read(self, table: BlockTable, layer: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for positions ``0 .. length-1`` as ``(length, num_kv_heads, head_dim)``."""
        idx = torch.as_tensor(
            table.physical_slots(0, length), dtype=torch.long, device=self.kv.device
        )
        return self._read_slots(layer, idx)

    def plan_read_many(
        self,
        tables: list[BlockTable],
        lengths: list[int],
        *,
        include_pages: bool = False,
        include_packed: bool = True,
    ) -> KVReadPlan:
        """Build the requested packed or native-page metadata for a decode step."""
        if len(tables) != len(lengths):
            raise ValueError(f"tables/lengths mismatch: {len(tables)} vs {len(lengths)}")
        if not tables:
            raise ValueError("a batched read needs at least one table")
        if any(length < 1 for length in lengths):
            raise ValueError(f"lengths must be positive; got {lengths}")
        if not include_pages and not include_packed:
            raise ValueError("a batched read needs packed indices or native page metadata")

        idx: torch.Tensor | None = None
        if include_packed:
            slots: list[int] = []
            for table, length in zip(tables, lengths, strict=True):
                slots.extend(table.physical_slots(0, length))
            idx = torch.as_tensor(slots, dtype=torch.long, device=self.kv.device)

        cu_seqlens: torch.Tensor | None = None
        if include_packed:
            cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=self.kv.device)
            cu_seqlens[1:] = torch.as_tensor(
                lengths, dtype=torch.int32, device=self.kv.device
            ).cumsum(0)
        return KVReadPlan(
            idx=idx,
            cu_seqlens=cu_seqlens,
            lengths=list(lengths),
            max_len=max(lengths),
            page_plan=self.plan_pages(tables, lengths) if include_pages else None,
        )

    def plan_pages(self, tables: list[BlockTable], lengths: list[int]) -> KVPagePlan:
        """Build page-table metadata for direct paged-attention backends."""
        if len(tables) != len(lengths):
            raise ValueError(f"tables/lengths mismatch: {len(tables)} vs {len(lengths)}")
        if not tables:
            raise ValueError("a paged read needs at least one table")
        if any(length < 1 for length in lengths):
            raise ValueError(f"lengths must be positive; got {lengths}")

        page_counts = [-(-length // self.block_size) for length in lengths]
        indptr_host = [0]
        indices_host: list[int] = []
        last_page_len_host: list[int] = []
        for table, length, page_count in zip(tables, lengths, page_counts, strict=True):
            if page_count > len(table.blocks):
                raise ValueError(
                    f"table has {len(table.blocks)} blocks but length {length} needs {page_count}"
                )
            indices_host.extend(table.blocks[:page_count])
            indptr_host.append(indptr_host[-1] + page_count)
            last_page_len_host.append(((length - 1) % self.block_size) + 1)

        device = self.kv.device
        return KVPagePlan(
            indptr=torch.tensor(indptr_host, dtype=torch.int32, device=device),
            indices=torch.tensor(indices_host, dtype=torch.int32, device=device),
            last_page_len=torch.tensor(last_page_len_host, dtype=torch.int32, device=device),
            page_size=self.block_size,
        )

    def read_many_plan(self, layer: int, plan: KVReadPlan) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather packed K/V for ``layer`` using a prebuilt :class:`KVReadPlan`."""
        if plan.idx is None:
            raise ValueError("packed K/V read needs packed indices")
        return self._read_slots(layer, plan.idx)

    def layer_kv(self, layer: int) -> torch.Tensor:
        """Return one layer's unified K/V page tensor for direct paged kernels."""
        return self.kv[layer]

    def _copy_block(self, source: int, target: int) -> None:
        self.kv[:, target].copy_(self.kv[:, source])

    def _slot_parts(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.div(slots, self.block_size, rounding_mode="floor"), slots % self.block_size

    def _read_slots(self, layer: int, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        blocks, offsets = self._slot_parts(slots)
        return self.kv[layer, blocks, 0, offsets], self.kv[layer, blocks, 1, offsets]

    def _write_slots(
        self,
        layer: int,
        slots: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        blocks, offsets = self._slot_parts(slots)
        self.kv[layer, blocks, 0, offsets] = key
        self.kv[layer, blocks, 1, offsets] = value
