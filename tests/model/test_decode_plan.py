"""Planned decode-window buffers equal the classic per-step block-table bookkeeping.

``DecodeWindowPlan`` replaces per-step Python walks (``physical_slot`` per request,
``plan_read_many`` re-listing every history) with device buffers built once per window.
These tests pin the equivalence step by step against the classic cache methods on ragged
batches, plus the plan-safety refusals (shared blocks) and the window budget bound.
"""

from __future__ import annotations

import pytest
import torch

from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.decode_plan import build_decode_window_plan


def _cache(num_blocks: int = 32, block_size: int = 4) -> PagedKVCache:
    return PagedKVCache(
        num_layers=1,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
    )


def _prefilled_table(cache: PagedKVCache, length: int):
    table = cache.new_request()
    table.reserve(length)
    table.length = length
    return table


def test_plan_matches_classic_bookkeeping_step_by_step() -> None:
    """Write slots and packed read plans equal the per-step block-table walks, every step."""
    cache = _cache()
    # Ragged lengths across block boundaries: 3 (partial), 5 (crosses), 9 (multi-block).
    tables = [_prefilled_table(cache, length) for length in (3, 5, 9)]
    budget = 6
    plan = build_decode_window_plan(cache, tables, budget)
    assert plan is not None

    for _ in range(budget):
        expected_writes = [table.physical_slot(table.length) for table in tables]
        write_slots, read_plan = plan.begin_step()
        assert write_slots.tolist() == expected_writes

        new_lengths = [table.length + 1 for table in tables]
        reference = cache.plan_read_many(tables, new_lengths)
        assert read_plan.idx.tolist() == reference.idx.tolist()
        assert read_plan.cu_seqlens.tolist() == reference.cu_seqlens.tolist()
        assert read_plan.lengths == reference.lengths
        assert read_plan.max_len == reference.max_len

        plan.complete_step()
        assert [table.length for table in tables] == new_lengths


def test_plan_refuses_shared_blocks() -> None:
    """A fork-shared table (refcount > 1) is not plan-safe: build returns None."""
    cache = _cache()
    leader = _prefilled_table(cache, 5)
    sibling = cache.fork_request(leader)
    assert build_decode_window_plan(cache, [leader, sibling], budget=2) is None


def test_plan_reserves_whole_window_and_bounds_steps() -> None:
    """The build reserves every window block up front and refuses steps past the budget."""
    cache = _cache()
    table = _prefilled_table(cache, 3)  # one partial block; budget 4 crosses into a second
    plan = build_decode_window_plan(cache, [table], budget=4)
    assert plan is not None
    assert table.capacity >= 3 + 4

    for _ in range(4):
        plan.begin_step()
        plan.complete_step()
    with pytest.raises(ValueError, match="window exhausted"):
        plan.begin_step()
