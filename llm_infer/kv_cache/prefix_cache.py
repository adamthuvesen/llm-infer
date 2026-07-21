"""Cross-request ("cross-turn") prefix cache over already-computed KV blocks.

A chat turn re-prefills the whole conversation: the webui resends every message and the
finished turn's KV blocks are freed. The engine's existing ``prefix_group_id`` sharing only
covers *concurrent* identical prompts, so nothing survives one turn to the next.

This store keeps a small set of finished sequences' block-aligned prompt-prefix blocks alive
between requests. A finishing request *donates* its cached prefix blocks (the store retains
them so the request's own table free does not return them to the pool); a new request's prefill
*looks up* the longest block-aligned prefix it shares with a stored entry and reuses those exact
blocks instead of recomputing them. Reused KV rows are the model's own earlier outputs for the
same absolute positions, so a greedy continuation is token-for-token identical to a cold prefill.

Entries are LRU-ordered and capped at a small count. Cached idle blocks sit with a refcount at
or above one, *outside* the allocator's free list, so a legal admission can find the pool dry
even though the scheduler's budget said it would fit. The store registers itself as the
allocator's ``eviction_source`` and gives blocks back under pressure (see
:meth:`_evict_for_shortfall`), which keeps the reserve scheduler's loud over-commit invariant
intact while still caching.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_infer.kv_cache.block_allocator import BlockAllocator

# Owner tag for allocator pool events the store triggers, so a KV trace can attribute a block
# that left the pool to the prefix cache rather than to any request's table.
_CACHE_OWNER = "prefix-cache"


@dataclass
class PrefixCacheEntry:
    """One cached sequence prefix: ``block_ids`` covers exactly ``token_ids``' positions."""

    token_ids: tuple[int, ...]
    block_ids: list[int]


@dataclass(frozen=True)
class PrefixCacheHit:
    """A lookup match. ``block_ids`` are already retained for the caller's new block table."""

    block_ids: list[int]
    tokens: int


def _common_prefix_len(cached: tuple[int, ...], prompt: list[int]) -> int:
    length = 0
    for cached_id, prompt_id in zip(cached, prompt, strict=False):
        if cached_id != prompt_id:
            break
        length += 1
    return length


def _is_prefix(shorter: tuple[int, ...], longer: tuple[int, ...]) -> bool:
    """Whether ``shorter`` is a (possibly equal-length) leading run of ``longer``."""
    return len(shorter) <= len(longer) and longer[: len(shorter)] == shorter


class PrefixCacheStore:
    """LRU store of donated prompt-prefix KV blocks, capped at ``max_entries`` entries.

    All entries are block-aligned: ``len(token_ids)`` is a multiple of ``block_size`` and each
    block holds exactly ``block_size`` positions, so a reused prefix always ends on a block
    boundary and a new request's first real write lands in a fresh private block — copy-on-write
    over a shared prefix block never has to run.
    """

    def __init__(self, allocator: BlockAllocator, block_size: int, *, max_entries: int = 8) -> None:
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1; got {block_size}")
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1; got {max_entries}")
        self.allocator = allocator
        self.block_size = block_size
        self.max_entries = max_entries
        # Front is least-recently-used, back is most-recently-used.
        self._entries: list[PrefixCacheEntry] = []
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0
        # Give blocks back when a legal allocation finds the pool dry because idle cached blocks
        # hold refs outside the free list.
        allocator.eviction_source = self._evict_for_shortfall

    @property
    def entry_count(self) -> int:
        """Number of cached prefixes currently held."""
        return len(self._entries)

    def cached_prefixes(self) -> list[tuple[int, ...]]:
        """Token ids of the current entries, least- to most-recently-used (introspection)."""
        return [entry.token_ids for entry in self._entries]

    def lookup(self, prompt_ids: list[int]) -> PrefixCacheHit | None:
        """Longest block-aligned prefix of ``prompt_ids`` held in the store, or ``None``.

        The match is capped at ``len(prompt_ids) - 1`` (then floored to a block multiple) so at
        least one prompt token is always really prefilled and the sampler still has final-token
        logits. On a hit the reused blocks are retained for the caller's new table and the entry
        becomes most-recently-used.
        """
        cap = ((len(prompt_ids) - 1) // self.block_size) * self.block_size
        if cap < self.block_size:
            self.misses += 1
            return None

        best_entry: PrefixCacheEntry | None = None
        best_k = 0
        for entry in self._entries:
            common = _common_prefix_len(entry.token_ids, prompt_ids)
            k = min((common // self.block_size) * self.block_size, cap)
            if k > best_k:
                best_entry = entry
                best_k = k

        if best_entry is None or best_k < self.block_size:
            self.misses += 1
            return None

        reused = best_entry.block_ids[: best_k // self.block_size]
        self.allocator.retain(reused)
        # Touch: move to most-recently-used so a burst of new prompts evicts cold entries first.
        self._entries.remove(best_entry)
        self._entries.append(best_entry)
        self.hits += 1
        self.hit_tokens += best_k
        return PrefixCacheHit(block_ids=list(reused), tokens=best_k)

    def donate(self, token_ids: list[int], block_ids: list[int]) -> None:
        """Cache ``block_ids`` (covering exactly ``token_ids``) from a finishing request.

        The store retains the blocks so they survive the donor table's own free, replaces any
        existing entry the donation supersedes (a stored prefix of the new tokens), and evicts
        the LRU entry when the cap is exceeded. Blocks are validated block-aligned up front.
        """
        if len(block_ids) * self.block_size != len(token_ids):
            raise ValueError(
                f"donation not block-aligned: {len(token_ids)} tokens, {len(block_ids)} blocks "
                f"at block_size {self.block_size}"
            )
        if not block_ids:
            raise ValueError("donation must cover at least one block")
        tokens = tuple(token_ids)
        # Retain before dropping duplicates: a superseded entry may share a physical block with
        # this donation, and its free must never drive a still-cached block to refcount zero.
        self.allocator.retain(block_ids)
        survivors: list[PrefixCacheEntry] = []
        for entry in self._entries:
            if _is_prefix(entry.token_ids, tokens):
                self.allocator.free(entry.block_ids, owner=_CACHE_OWNER)
            else:
                survivors.append(entry)
        survivors.append(PrefixCacheEntry(tokens, list(block_ids)))
        self._entries = survivors
        while len(self._entries) > self.max_entries:
            self._pop_lru_and_free()

    def clear(self) -> None:
        """Drop every entry, returning its blocks to the pool — for shutdown and tests."""
        for entry in self._entries:
            self.allocator.free(entry.block_ids, owner=_CACHE_OWNER)
        self._entries = []

    def _evict_for_shortfall(self, needed_blocks: int) -> None:
        """Free LRU entries until ``needed_blocks`` blocks actually return to the pool, or empty.

        An evicted entry's blocks return to the free list only when no live request still shares
        them, so eviction counts blocks that *actually* freed (via the pool delta) rather than
        entries dropped, and keeps going until the shortfall is genuinely covered.
        """
        freed = 0
        while freed < needed_blocks and self._entries:
            before = self.allocator.num_free
            self._pop_lru_and_free()
            freed += self.allocator.num_free - before

    def _pop_lru_and_free(self) -> None:
        entry = self._entries.pop(0)
        self.allocator.free(entry.block_ids, owner=_CACHE_OWNER)
