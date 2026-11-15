"""Shared test doubles implementing the causal-LM backend surface."""

from __future__ import annotations

from typing import cast

from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.model.interface import BackendCapabilities, CausalLMBackend


class FakeCausalLMBase:
    """Minimal backend fields plus ``release_table`` for engine lifecycle tests."""

    num_layers: int = 1
    num_kv_heads: int = 1
    head_dim: int = 2
    dtype = None
    device = None
    profiler = None
    backend = None
    capabilities = BackendCapabilities(
        paged_kv=True,
        prefix_caching=True,
        speculative=True,
        flash_attention=False,
    )

    def release_table(self, table: BlockTable) -> None:
        del table


def as_backend(fake: object) -> CausalLMBackend:
    """Hand a partial test double to code typed against the full backend protocol."""
    return cast(CausalLMBackend, fake)
