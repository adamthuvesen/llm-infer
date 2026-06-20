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

import torch

from llm_infer.kv_cache.block_allocator import BlockAllocator
from llm_infer.kv_cache.block_table import BlockTable


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
        idx = torch.as_tensor(table.physical_slots(start_pos, n), dtype=torch.long)
        self.key[layer].view(-1, self.num_kv_heads, self.head_dim)[idx] = key
        self.value[layer].view(-1, self.num_kv_heads, self.head_dim)[idx] = value

    def read(
        self, table: BlockTable, layer: int, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather K/V for positions ``0 .. length-1`` as ``(length, num_kv_heads, head_dim)``."""
        idx = torch.as_tensor(table.physical_slots(0, length), dtype=torch.long)
        key = self.key[layer].view(-1, self.num_kv_heads, self.head_dim)[idx]
        value = self.value[layer].view(-1, self.num_kv_heads, self.head_dim)[idx]
        return key, value
