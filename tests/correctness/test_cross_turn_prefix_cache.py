"""Cross-turn prefix cache: reuse a finished turn's KV, token-for-token identical to a cold run.

The engine-level checks run on the tiny ``llm_pretrain_dense_v1`` bundle so a real paged
prefill/decode path is exercised on CPU with zero spend. The store-level checks pin the LRU,
eviction, partial-match, and cap behavior directly on :class:`PrefixCacheStore`.
"""

from __future__ import annotations

from pathlib import Path

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as _write_tiny_bundle
from llm_infer.kv_cache.block_allocator import BlockAllocator
from llm_infer.kv_cache.prefix_cache import PrefixCacheStore
from llm_infer.model.decode import greedy_decode
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.request import Request

_BLOCK_SIZE = 4
_EOS: frozenset[int] = frozenset()


def _engine(tmp_path: Path, *, prefix_cache: bool, num_blocks: int = 64) -> InferenceEngine:
    runtime = load_model_runtime("esme", bundle_path=_write_tiny_bundle(tmp_path))
    return InferenceEngine(
        runtime.model,
        block_size=_BLOCK_SIZE,
        num_blocks=num_blocks,
        capabilities=runtime.capabilities,
        prefix_cache=prefix_cache,
    )


def _run_one(engine: InferenceEngine, request_id: str, prompt: list[int], steps: int) -> list[int]:
    engine.add_request(Request(request_id, prompt, steps, _EOS))
    return engine.run()[request_id]


def test_cache_hit_matches_cold_run_and_reuses_block_aligned_prefix(tmp_path: Path) -> None:
    runtime = load_model_runtime("esme", bundle_path=_write_tiny_bundle(tmp_path))
    engine = InferenceEngine(
        runtime.model,
        block_size=_BLOCK_SIZE,
        num_blocks=64,
        capabilities=runtime.capabilities,
        prefix_cache=True,
    )
    assert engine.prefix_cache is not None

    # Turn A: prompt P, greedy to the length cap; it donates its block-aligned prefix on finish.
    prompt_a = [4, 5, 6, 7, 8]
    output_a = _run_one(engine, "A", prompt_a, steps=5)
    assert engine.prefix_cache.entry_count == 1

    # Turn B continues the conversation: P + A's output + one more token. Its leading positions
    # are identical to A's, so the cache should serve a block-aligned prefix.
    prompt_b = prompt_a + output_a + [9]
    output_b = _run_one(engine, "B", prompt_b, steps=5)

    # The reference is the full-recompute greedy oracle over B's whole prompt — the cache must
    # not change a single token.
    reference_b = greedy_decode(runtime.model, prompt_b, max_new_tokens=5, eos_token_ids=set())
    assert output_b == reference_b

    # B reused a block-aligned prefix: A cached 10 positions (prompt 5 + 5 generated), floored to
    # 8 (two 4-token blocks); B's match caps at len(prompt_b) - 1 = 10 -> still the 8-token prefix.
    assert engine.prefix_cache.hits == 1
    assert engine.prefix_cache.hit_tokens == 8


def test_blocks_return_to_pool_after_clear(tmp_path: Path) -> None:
    engine = _engine(tmp_path, prefix_cache=True)
    assert engine.prefix_cache is not None
    total = engine.cache.allocator.num_blocks

    output_a = _run_one(engine, "A", [4, 5, 6, 7, 8], steps=5)
    _run_one(engine, "B", [4, 5, 6, 7, 8] + output_a + [9], steps=5)

    # Finished tables freed, but donated blocks are still held by the store — so the pool is not
    # yet whole. Clearing the store returns every cached block.
    assert engine.cache.allocator.num_free < total
    engine.prefix_cache.clear()
    assert engine.cache.allocator.num_free == total
    assert engine.prefix_cache.entry_count == 0


def _finish_and_donate(
    allocator: BlockAllocator, store: PrefixCacheStore, token_ids: list[int]
) -> None:
    """Simulate one request finishing: allocate its blocks, donate, then free its own table ref."""
    block_count = len(token_ids) // store.block_size
    blocks = allocator.allocate(block_count, owner="req")
    store.donate(token_ids, blocks)
    allocator.free(blocks, owner="req")  # the request's table free; the store keeps its retain


def test_eviction_frees_lru_blocks_for_a_legal_allocation() -> None:
    allocator = BlockAllocator(2)
    store = PrefixCacheStore(allocator, block_size=2, max_entries=8)

    _finish_and_donate(allocator, store, [10, 11])  # LRU entry (one block), held outside free list
    _finish_and_donate(allocator, store, [20, 21])  # MRU entry (one block)
    assert allocator.num_free == 0
    assert store.entry_count == 2

    # A one-block allocation with an empty free list: the store must give a block back. It evicts
    # only the LRU entry (its freed block covers the one-block shortfall).
    handed = allocator.allocate(1, owner="new")
    assert len(handed) == 1
    assert store.entry_count == 1
    assert store.cached_prefixes() == [(20, 21)]  # the LRU (10, 11) entry was evicted


def test_partial_match_reuses_common_block_aligned_prefix() -> None:
    allocator = BlockAllocator(8)
    store = PrefixCacheStore(allocator, block_size=2, max_entries=8)
    _finish_and_donate(allocator, store, [1, 2, 3, 4])  # two blocks: [1,2] and [3,4]

    # A prompt that agrees on [1, 2] then diverges: only the first whole block is reusable.
    hit = store.lookup([1, 2, 9, 8, 7])
    assert hit is not None
    assert hit.tokens == 2
    assert len(hit.block_ids) == 1
    assert store.hit_tokens == 2


def test_cap_evicts_lru_entry_on_the_ninth_donation() -> None:
    allocator = BlockAllocator(16)
    store = PrefixCacheStore(allocator, block_size=1, max_entries=8)
    for index in range(9):
        _finish_and_donate(allocator, store, [100 + index])

    assert store.entry_count == 8
    prefixes = store.cached_prefixes()
    assert (100,) not in prefixes  # the first (LRU) donation was evicted by the ninth
    assert prefixes[-1] == (108,)  # the ninth donation is the most-recently-used entry
