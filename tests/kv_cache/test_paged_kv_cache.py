"""Unit tests for the paged KV-cache: allocator bookkeeping, block tables, and the store.

These need no model — they prove the page mechanics the scoping doc calls out
(allocate → write → read → free → reallocate) and, crucially, that two requests
sharing the pool never read each other's K/V. A block-gather/contamination bug here
is the likeliest cause of any cached-vs-recompute divergence, so it is pinned directly.
"""

from __future__ import annotations

import pytest
import torch

from llm_infer.kv_cache import BlockAllocator, BlockTable, OutOfBlocksError, PagedKVCache


def test_allocator_allocate_and_free_round_trip() -> None:
    alloc = BlockAllocator(num_blocks=4)
    assert alloc.num_free == 4
    a = alloc.allocate(3)
    assert len(a) == 3
    assert alloc.num_free == 1
    alloc.free(a)
    assert alloc.num_free == 4


def test_allocator_reuses_freed_blocks() -> None:
    """Freeing then reallocating must hand the same physical blocks back out."""
    alloc = BlockAllocator(num_blocks=4)
    first = alloc.allocate(4)
    alloc.free(first[:2])
    reused = alloc.allocate(2)
    assert set(reused) == set(first[:2])
    assert alloc.num_free == 0


def test_allocator_out_of_blocks_is_loud() -> None:
    alloc = BlockAllocator(num_blocks=2)
    with pytest.raises(OutOfBlocksError):
        alloc.allocate(3)


def test_allocator_rejects_double_free_and_bad_ids() -> None:
    alloc = BlockAllocator(num_blocks=2)
    blocks = alloc.allocate(2)
    # A duplicate id within one free() call (distinct from freeing an already-free id).
    with pytest.raises(ValueError, match="duplicate block id"):
        alloc.free([blocks[0], blocks[0]])
    with pytest.raises(ValueError, match="out of range"):
        alloc.free([99])
    # Freeing a block that is genuinely already in the pool is the double-free case.
    alloc.free([blocks[0]])
    with pytest.raises(ValueError, match="double free"):
        alloc.free([blocks[0]])


def test_block_table_grows_at_block_boundaries() -> None:
    alloc = BlockAllocator(num_blocks=8)
    table = BlockTable(alloc, block_size=4)
    table.reserve(4)
    assert table.num_blocks == 1
    table.length = 4
    table.reserve(1)  # crossing into the second block
    assert table.num_blocks == 2
    assert table.capacity == 8


def test_block_table_slot_mapping() -> None:
    alloc = BlockAllocator(num_blocks=8)
    table = BlockTable(alloc, block_size=4)
    table.reserve(6)
    b0, b1 = table.blocks
    # positions 0..3 live in the first block, 4..5 in the second.
    assert table.physical_slot(0) == b0 * 4 + 0
    assert table.physical_slot(3) == b0 * 4 + 3
    assert table.physical_slot(4) == b1 * 4 + 0
    assert table.physical_slots(0, 6) == [
        b0 * 4,
        b0 * 4 + 1,
        b0 * 4 + 2,
        b0 * 4 + 3,
        b1 * 4,
        b1 * 4 + 1,
    ]


def test_block_table_slot_out_of_range_is_loud() -> None:
    alloc = BlockAllocator(num_blocks=8)
    table = BlockTable(alloc, block_size=4)
    table.reserve(2)
    with pytest.raises(ValueError, match="reserve first"):
        table.physical_slot(4)


def test_block_table_free_returns_blocks() -> None:
    alloc = BlockAllocator(num_blocks=8)
    table = BlockTable(alloc, block_size=4)
    table.reserve(6)
    assert alloc.num_free == 6
    table.free()
    assert alloc.num_free == 8
    assert table.num_blocks == 0
    assert table.length == 0


def _ramp(n: int, kv_heads: int, head_dim: int, offset: float) -> torch.Tensor:
    return torch.arange(n * kv_heads * head_dim, dtype=torch.float32).reshape(
        n, kv_heads, head_dim
    ) + offset


def test_cache_write_read_round_trip() -> None:
    cache = PagedKVCache(
        num_layers=2, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=3, dtype=torch.float32
    )
    table = cache.new_request()
    table.reserve(6)
    key = _ramp(6, 2, 3, offset=0.0)
    value = _ramp(6, 2, 3, offset=100.0)
    cache.write(table, layer=0, start_pos=0, key=key, value=value)
    got_k, got_v = cache.read(table, layer=0, length=6)
    assert torch.equal(got_k, key)
    assert torch.equal(got_v, value)


def test_cache_incremental_append_like_decode() -> None:
    """Append one position at a time (the decode pattern) and read the growing history."""
    cache = PagedKVCache(
        num_layers=1, num_blocks=4, block_size=2, num_kv_heads=1, head_dim=2, dtype=torch.float32
    )
    table = cache.new_request()
    for pos in range(5):
        table.reserve(1)
        row_k = torch.full((1, 1, 2), float(pos))
        row_v = torch.full((1, 1, 2), float(pos) + 50.0)
        cache.write(table, layer=0, start_pos=pos, key=row_k, value=row_v)
        table.length = pos + 1
        got_k, _ = cache.read(table, layer=0, length=pos + 1)
        assert got_k.shape[0] == pos + 1
        assert torch.equal(got_k[-1], row_k[0])
    final_k, final_v = cache.read(table, layer=0, length=5)
    assert torch.equal(final_k[:, 0, 0], torch.arange(5, dtype=torch.float32))
    assert torch.equal(final_v[:, 0, 0], torch.arange(5, dtype=torch.float32) + 50.0)


def test_two_requests_do_not_collide() -> None:
    """Distinct block tables sharing the pool must read back exactly their own K/V."""
    cache = PagedKVCache(
        num_layers=2, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=3, dtype=torch.float32
    )
    a = cache.new_request()
    b = cache.new_request()
    a.reserve(6)
    b.reserve(6)
    assert set(a.blocks).isdisjoint(b.blocks)

    ka, va = _ramp(6, 2, 3, 0.0), _ramp(6, 2, 3, 100.0)
    kb, vb = _ramp(6, 2, 3, 1000.0), _ramp(6, 2, 3, 2000.0)
    cache.write(a, layer=0, start_pos=0, key=ka, value=va)
    cache.write(b, layer=0, start_pos=0, key=kb, value=vb)

    got_ak, got_av = cache.read(a, layer=0, length=6)
    got_bk, got_bv = cache.read(b, layer=0, length=6)
    assert torch.equal(got_ak, ka)
    assert torch.equal(got_av, va)
    assert torch.equal(got_bk, kb)
    assert torch.equal(got_bv, vb)


def test_read_many_packs_ragged_histories() -> None:
    cache = PagedKVCache(
        num_layers=1, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=3, dtype=torch.float32
    )
    a = cache.new_request()
    b = cache.new_request()
    a.reserve(3)
    b.reserve(5)

    ka, va = _ramp(3, 2, 3, 0.0), _ramp(3, 2, 3, 100.0)
    kb, vb = _ramp(5, 2, 3, 1000.0), _ramp(5, 2, 3, 2000.0)
    cache.write(a, layer=0, start_pos=0, key=ka, value=va)
    cache.write(b, layer=0, start_pos=0, key=kb, value=vb)

    key, value, cu_seqlens, max_len = cache.read_many([a, b], layer=0, lengths=[3, 5])

    assert torch.equal(key, torch.cat([ka, kb], dim=0))
    assert torch.equal(value, torch.cat([va, vb], dim=0))
    assert torch.equal(cu_seqlens.cpu(), torch.tensor([0, 3, 8], dtype=torch.int32))
    assert max_len == 5


def test_read_many_plan_reuses_indices_across_layers() -> None:
    cache = PagedKVCache(
        num_layers=2, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=3, dtype=torch.float32
    )
    a = cache.new_request()
    b = cache.new_request()
    a.reserve(3)
    b.reserve(5)

    layer0_a, layer0_av = _ramp(3, 2, 3, 0.0), _ramp(3, 2, 3, 100.0)
    layer0_b, layer0_bv = _ramp(5, 2, 3, 1000.0), _ramp(5, 2, 3, 2000.0)
    layer1_a, layer1_av = _ramp(3, 2, 3, 3000.0), _ramp(3, 2, 3, 4000.0)
    layer1_b, layer1_bv = _ramp(5, 2, 3, 5000.0), _ramp(5, 2, 3, 6000.0)
    cache.write(a, layer=0, start_pos=0, key=layer0_a, value=layer0_av)
    cache.write(b, layer=0, start_pos=0, key=layer0_b, value=layer0_bv)
    cache.write(a, layer=1, start_pos=0, key=layer1_a, value=layer1_av)
    cache.write(b, layer=1, start_pos=0, key=layer1_b, value=layer1_bv)

    plan = cache.plan_read_many([a, b], lengths=[3, 5])
    key0, value0 = cache.read_many_plan(layer=0, plan=plan)
    key1, value1 = cache.read_many_plan(layer=1, plan=plan)

    assert torch.equal(plan.cu_seqlens.cpu(), torch.tensor([0, 3, 8], dtype=torch.int32))
    assert plan.lengths == [3, 5]
    assert plan.max_len == 5
    assert torch.equal(key0, torch.cat([layer0_a, layer0_b], dim=0))
    assert torch.equal(value0, torch.cat([layer0_av, layer0_bv], dim=0))
    assert torch.equal(key1, torch.cat([layer1_a, layer1_b], dim=0))
    assert torch.equal(value1, torch.cat([layer1_av, layer1_bv], dim=0))


def test_write_many_stores_one_decode_row_per_request() -> None:
    cache = PagedKVCache(
        num_layers=1, num_blocks=8, block_size=4, num_kv_heads=2, head_dim=3, dtype=torch.float32
    )
    a = cache.new_request()
    b = cache.new_request()
    a.reserve(4)
    b.reserve(4)
    a.length = 2
    b.length = 3

    key = _ramp(2, 2, 3, 10.0)
    value = _ramp(2, 2, 3, 100.0)
    cache.write_many([a, b], layer=0, positions=[2, 3], key=key, value=value)

    got_a, got_av = cache.read(a, layer=0, length=3)
    got_b, got_bv = cache.read(b, layer=0, length=4)
    assert torch.equal(got_a[2], key[0])
    assert torch.equal(got_b[3], key[1])
    assert torch.equal(got_av[2], value[0])
    assert torch.equal(got_bv[3], value[1])


def test_layers_are_independent() -> None:
    cache = PagedKVCache(
        num_layers=2, num_blocks=4, block_size=4, num_kv_heads=1, head_dim=2, dtype=torch.float32
    )
    table = cache.new_request()
    table.reserve(3)
    cache.write(table, layer=0, start_pos=0, key=_ramp(3, 1, 2, 1.0), value=_ramp(3, 1, 2, 1.0))
    layer1_k, _ = cache.read(table, layer=1, length=3)
    assert torch.count_nonzero(layer1_k) == 0


def test_new_request_binds_cache_allocator() -> None:
    cache = PagedKVCache(
        num_layers=1, num_blocks=4, block_size=2, num_kv_heads=1, head_dim=2, dtype=torch.float32
    )
    table = cache.new_request()
    assert isinstance(table, BlockTable)
    assert table.allocator is cache.allocator
    assert table.block_size == 2
