"""A single generation request and its mutable runtime state.

Created by the caller with its prompt and stop config; the engine fills in the block
table on admission and grows ``generated`` one token per step. Stop semantics mirror
the Phase A ``greedy_decode`` exactly (EOS token included, capped at ``max_new_tokens``)
so the cached/paged path reproduces the full-recompute reference token-for-token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from llm_infer.kv_cache.block_table import BlockTable


@dataclass
class Request:
    """One greedy generation request, carrying its own KV block table and output."""

    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    eos_token_ids: frozenset[int]

    block_table: BlockTable | None = None
    prefilled: bool = False
    finished: bool = False
    _generated_tokens: list[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _generated_cache: list[int] | None = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1; got {self.max_new_tokens}")

    @property
    def generated(self) -> list[int]:
        """Generated ids materialized as Python ints at output/test boundaries."""
        if self._generated_cache is None:
            if not self._generated_tokens:
                self._generated_cache = []
            else:
                stacked = torch.stack([token.reshape(()) for token in self._generated_tokens])
                self._generated_cache = [int(token) for token in stacked.cpu().tolist()]
        return list(self._generated_cache)

    @property
    def last_token(self) -> int:
        """The most recently produced token as a Python int, for boundary callers."""
        if not self._generated_tokens:
            raise ValueError(f"request {self.request_id!r} has produced no tokens yet")
        return self.generated[-1]

    @property
    def last_token_tensor(self) -> torch.Tensor:
        """The most recently produced token, still on its original device."""
        if not self._generated_tokens:
            raise ValueError(f"request {self.request_id!r} has produced no tokens yet")
        return self._generated_tokens[-1]

    def record(self, token_id: int | torch.Tensor, *, is_eos: bool | None = None) -> None:
        """Append a sampled token and apply the stop rule (EOS or length cap)."""
        token = torch.as_tensor(token_id, dtype=torch.long).reshape(())
        self._generated_tokens.append(token.detach())
        self._generated_cache = None

        if is_eos is None:
            is_eos = int(token.cpu().item()) in self.eos_token_ids
        if is_eos or len(self._generated_tokens) >= self.max_new_tokens:
            self.finished = True
