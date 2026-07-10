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
    plan = build_decode_window_plan(cache, tables, budget, include_pages=True, include_packed=True)
    assert plan is not None

    for _ in range(budget):
        expected_writes = [table.physical_slot(table.length) for table in tables]
        write_slots, read_plan = plan.begin_step(include_pages=True, include_packed=True)
        assert write_slots.tolist() == expected_writes

        new_lengths = [table.length + 1 for table in tables]
        reference = cache.plan_read_many(tables, new_lengths, include_pages=True)
        assert read_plan.idx is not None
        assert reference.idx is not None
        assert read_plan.idx.tolist() == reference.idx.tolist()
        assert read_plan.cu_seqlens.tolist() == reference.cu_seqlens.tolist()
        assert read_plan.lengths == reference.lengths
        assert read_plan.max_len == reference.max_len
        assert read_plan.page_plan is not None
        assert reference.page_plan is not None
        assert read_plan.page_plan.indptr.tolist() == reference.page_plan.indptr.tolist()
        assert read_plan.page_plan.indices.tolist() == reference.page_plan.indices.tolist()
        assert (
            read_plan.page_plan.last_page_len.tolist() == reference.page_plan.last_page_len.tolist()
        )
        assert read_plan.page_plan.page_size == reference.page_plan.page_size

        plan.complete_step()
        assert [table.length for table in tables] == new_lengths


def test_native_page_metadata_matches_classic_without_packed_indices() -> None:
    """Native pages stay equivalent while avoiding the packed physical-slot gather."""
    cache = _cache()
    tables = [_prefilled_table(cache, length) for length in (3, 5, 9)]
    plan = build_decode_window_plan(
        cache, tables, budget=4, include_pages=True, include_packed=False
    )
    assert plan is not None
    assert plan.read_slots is None
    assert plan.cu_seqlens is None
    assert plan.cu_step is None
    assert plan.cu_seqlens_long is None
    assert plan.cu_step_long is None
    assert plan.flat_arange is None

    for _ in range(plan.budget):
        _, read_plan = plan.begin_step(include_pages=True, include_packed=False)
        expected = cache.plan_pages(tables, [table.length + 1 for table in tables])

        assert read_plan.idx is None
        assert read_plan.cu_seqlens is None
        assert read_plan.page_plan is not None
        assert read_plan.page_plan.indptr.tolist() == expected.indptr.tolist()
        assert read_plan.page_plan.indices.tolist() == expected.indices.tolist()
        assert read_plan.page_plan.last_page_len.tolist() == expected.last_page_len.tolist()
        plan.complete_step()


def test_native_page_buffers_keep_addresses_across_page_growth() -> None:
    """Crossing a block boundary changes values and length, never metadata addresses."""
    cache = _cache(block_size=4)
    tables = [_prefilled_table(cache, length) for length in (3, 4)]
    plan = build_decode_window_plan(
        cache, tables, budget=3, include_pages=True, include_packed=False
    )
    assert plan is not None
    assert plan.page_indptr is not None
    assert plan.native_page_indices is not None
    assert plan.page_last_page_len is not None

    addresses = (
        plan.page_indptr.data_ptr(),
        plan.native_page_indices.data_ptr(),
        plan.page_last_page_len.data_ptr(),
    )
    observed_page_counts: list[int] = []
    for _ in range(plan.budget):
        _, read_plan = plan.begin_step(include_pages=True, include_packed=False)
        assert read_plan.page_plan is not None
        observed_page_counts.append(int(read_plan.page_plan.indices.numel()))
        assert read_plan.page_plan.indptr.data_ptr() == addresses[0]
        assert read_plan.page_plan.indices.data_ptr() == addresses[1]
        assert read_plan.page_plan.last_page_len.data_ptr() == addresses[2]
        plan.complete_step()

    assert observed_page_counts == [3, 4, 4]


def test_packed_fallback_still_builds_exact_read_indices() -> None:
    """Backends without native pages retain the existing request-major packed plan."""
    cache = _cache()
    tables = [_prefilled_table(cache, length) for length in (2, 6)]
    plan = build_decode_window_plan(cache, tables, budget=2)
    assert plan is not None
    assert plan.page_indptr_matrix is None
    assert plan.page_indices_matrix is None
    assert plan.page_last_page_len_matrix is None
    assert plan.page_indptr is None
    assert plan.native_page_indices is None
    assert plan.page_last_page_len is None

    for _ in range(plan.budget):
        _, read_plan = plan.begin_step()
        expected = cache.plan_read_many(tables, [table.length + 1 for table in tables])
        assert read_plan.page_plan is None
        assert read_plan.idx is not None
        assert read_plan.cu_seqlens is not None
        assert expected.idx is not None
        assert expected.cu_seqlens is not None
        assert read_plan.idx.tolist() == expected.idx.tolist()
        assert read_plan.cu_seqlens.tolist() == expected.cu_seqlens.tolist()
        plan.complete_step()


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
