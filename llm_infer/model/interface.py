"""Backend contract shared by model implementations and the inference engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import torch

from llm_infer.kernels.base import AttentionBackend
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.profiling import TimingProfiler


class CausalLMBackend(Protocol):
    """Minimal causal-LM surface the scheduler/serving stack needs."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    backend: AttentionBackend
    profiler: TimingProfiler | None

    def logits(self, token_ids: list[int]) -> torch.Tensor:
        """Return next-token logits for each input position."""

    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        """Cache a full prompt and return logits for the last prompt token."""

    def prefill_chunk(
        self,
        prompt_ids: list[int],
        cache: PagedKVCache,
        table: BlockTable,
        *,
        start_pos: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """Cache one prompt chunk and return logits for the last chunk token."""

    def decode_one(
        self, cache: PagedKVCache, table: BlockTable, token_id: int | torch.Tensor
    ) -> torch.Tensor:
        """Append one token for one request and return its next-token logits."""

    def decode_many(
        self,
        cache: PagedKVCache,
        tables: list[BlockTable],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Append one token per request and return one logits row per request."""


class TokenizerLike(Protocol):
    """Tokenizer methods used by the OpenAI-compatible server."""

    def encode(self, text: str) -> list[int]:
        """Return token ids for raw text."""

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Return text for token ids."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> object:
        """Render or tokenize chat messages."""


@dataclass(frozen=True)
class ModelRuntime:
    """Loaded model plus the tokenizer/provenance needed by serving and benchmarks."""

    backend_id: str
    model_id: str
    model: CausalLMBackend
    tokenizer: TokenizerLike
    eos_token_ids: frozenset[int]
    metadata: dict[str, object] = field(default_factory=dict)
    bundle_path: Path | None = None
