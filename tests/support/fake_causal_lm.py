"""Shared test doubles implementing the causal-LM backend surface."""

from __future__ import annotations

from llm_infer.kv_cache.block_table import BlockTable


class FakeCausalLMBase:
    """Minimal backend fields plus ``release_table`` for engine lifecycle tests."""

    num_layers: int = 1
    num_kv_heads: int = 1
    head_dim: int = 2
    dtype = None
    device = None
    profiler = None
    backend = None

    def release_table(self, table: BlockTable) -> None:
        del table
