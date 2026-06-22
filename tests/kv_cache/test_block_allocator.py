"""Allocator hardening (audit 2026-06-21): a failed free is atomic — the pool is unchanged.

``BlockAllocator.free`` validates the whole batch before mutating the free list, so a bad
free (out-of-range, already-free, or a duplicate within the argument) raises while leaving
``num_free`` and the next allocation exactly as they were. A half-applied free could return
a still-owned block to the pool and alias another request's KV pages — hence the atomicity.
"""

from __future__ import annotations

import pytest

from llm_infer.kv_cache.block_allocator import BlockAllocator


def test_valid_free_returns_all_blocks() -> None:
    allocator = BlockAllocator(4)
    held = allocator.allocate(3)
    allocator.free(held)
    assert allocator.num_free == 4


def test_refcounted_free_waits_for_last_owner() -> None:
    allocator = BlockAllocator(4)
    held = allocator.allocate(2)
    allocator.retain(held)
    assert allocator.refcount(held[0]) == 2
    assert allocator.refcount(held[1]) == 2

    allocator.free([held[0]])
    assert allocator.refcount(held[0]) == 1
    assert allocator.num_free == 2

    allocator.free([held[0]])
    assert allocator.refcount(held[0]) == 0
    assert allocator.num_free == 3


def test_duplicate_in_argument_raises_without_mutation() -> None:
    allocator = BlockAllocator(4)
    held = allocator.allocate(2)  # owns 2 blocks; 2 remain free
    before = allocator.num_free
    with pytest.raises(ValueError, match="duplicate block id"):
        allocator.free([held[0], held[0]])
    assert allocator.num_free == before  # neither copy returned
    # The held blocks are still owned: a full reallocation never hands them back out.
    reallocated = allocator.allocate(before)
    assert held[0] not in reallocated and held[1] not in reallocated


def test_already_free_id_raises_before_applying_earlier_valid_id() -> None:
    allocator = BlockAllocator(4)
    held = allocator.allocate(1)  # owns 1 block; 3 remain free
    before = allocator.num_free
    free_id = next(b for b in range(4) if b != held[0])  # some currently-free block
    with pytest.raises(ValueError, match="double free"):
        allocator.free([held[0], free_id])  # valid id first, then an already-free id
    assert allocator.num_free == before  # the valid held[0] was NOT returned


def test_out_of_range_id_raises_without_mutation() -> None:
    allocator = BlockAllocator(4)
    held = allocator.allocate(2)
    before = allocator.num_free
    with pytest.raises(ValueError, match="out of range"):
        allocator.free([held[0], 99])
    assert allocator.num_free == before
