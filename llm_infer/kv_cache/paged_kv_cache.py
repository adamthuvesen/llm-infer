"""The paged KV-cache store: the K/V tensors plus scatter/gather over block tables.

Layout is one contiguous tensor per side, shaped
``(num_layers, num_blocks, block_size, num_kv_heads, head_dim)``. K/V are stored
*after* RoPE but *before* GQA expansion (one row per KV head, not per query head) —
each token's rotation is fixed by its absolute position, so it is computed once at
write time and never re-rotated, and the cheaper KV-head layout is expanded to query
heads on read by the model. Reads/writes go through a request's :class:`BlockTable`,
which translates logical positions to physical slots, so two requests sharing the
pool never collide.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llm_infer.kv_cache.block_allocator import BlockAllocator
from llm_infer.kv_cache.block_table import BlockTable


@dataclass(frozen=True)
class KVReadPlan:
    """Layer-independent packed-read metadata for one batched decode step."""

    idx: torch.Tensor
    cu_seqlens: torch.Tensor
    lengths: list[int]
    max_len: int


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
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.key = torch.zeros(shape, dtype=dtype, device=device)
        self.value = torch.zeros(shape, dtype=dtype, device=device)

    def new_request(self) -> BlockTable:
        """A fresh, empty block table bound to this cache's allocator and block size."""
        return BlockTable(self.allocator, self.block_size)

    def fork_request(self, table: BlockTable) -> BlockTable:
        """Fork a request's current block table by shared physical-block ownership."""
        return table.fork_shared()

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
        self.key[layer].view(-1, self.num_kv_heads, self.head_dim)[idx] = key
        self.value[layer].view(-1, self.num_kv_heads, self.head_dim)[idx] = value

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
        slots = [
            table.physical_slot(pos) for table, pos in zip(tables, positions, strict=True)
        ]
        idx = torch.as_tensor(slots, dtype=torch.long, device=key.device)
        self.key[layer].view(-1, self.num_kv_heads, self.head_dim)[idx] = key
        self.value[layer].view(-1, self.num_kv_heads, self.head_dim)[idx] = value

    def read(
        self, table: BlockTable, layer: int, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for positions ``0 .. length-1`` as ``(length, num_kv_heads, head_dim)``."""
        idx = torch.as_tensor(
            table.physical_slots(0, length), dtype=torch.long, device=self.key.device
        )
        key = self.key[layer].view(-1, self.num_kv_heads, self.head_dim)[idx]
        value = self.value[layer].view(-1, self.num_kv_heads, self.head_dim)[idx]
        return key, value

    def read_many(
        self,
        tables: list[BlockTable],
        layer: int,
        lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Gather ragged histories for several requests with one indexed read per side.

        Returns packed token-major K/V plus ``cu_seqlens`` for varlen attention:
        K/V ``(sum(lengths), num_kv_heads, head_dim)``, ``cu_seqlens`` ``(B + 1,)``.
        """
        plan = self.plan_read_many(tables, lengths)
        return (*self.read_many_plan(layer, plan), plan.cu_seqlens, plan.max_len)

    def plan_read_many(self, tables: list[BlockTable], lengths: list[int]) -> KVReadPlan:
        """Build reusable packed-read indices for a batched decode step."""
        if len(tables) != len(lengths):
            raise ValueError(f"tables/lengths mismatch: {len(tables)} vs {len(lengths)}")
        if not tables:
            raise ValueError("read_many needs at least one table")
        if any(length < 1 for length in lengths):
            raise ValueError(f"lengths must be positive; got {lengths}")

        slots: list[int] = []
        for table, length in zip(tables, lengths, strict=True):
            slots.extend(table.physical_slots(0, length))
        idx = torch.as_tensor(slots, dtype=torch.long, device=self.key.device)

        cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=self.key.device)
        cu_seqlens[1:] = torch.as_tensor(lengths, dtype=torch.int32, device=self.key.device).cumsum(
            0
        )
        return KVReadPlan(
            idx=idx,
            cu_seqlens=cu_seqlens,
            lengths=list(lengths),
            max_len=max(lengths),
        )

    def read_many_plan(
        self, layer: int, plan: KVReadPlan
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather packed K/V for ``layer`` using a prebuilt :class:`KVReadPlan`."""
        key = self.key[layer].view(-1, self.num_kv_heads, self.head_dim)[plan.idx]
        value = self.value[layer].view(-1, self.num_kv_heads, self.head_dim)[plan.idx]
        return key, value

    def _copy_block(self, source: int, target: int) -> None:
        self.key[:, target].copy_(self.key[:, source])
        self.value[:, target].copy_(self.value[:, source])
