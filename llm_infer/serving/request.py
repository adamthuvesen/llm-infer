"""A single generation request and its mutable runtime state.

Created by the caller with its prompt and stop config; the engine fills in the block
table on admission and grows ``generated`` one token per step. Stop semantics mirror
the Phase A ``greedy_decode`` exactly (EOS token included, capped at ``max_new_tokens``)
so the cached/paged path reproduces the full-recompute reference token-for-token.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_infer.kv_cache.block_table import BlockTable


@dataclass
class Request:
    """One greedy generation request, carrying its own KV block table and output."""

    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    eos_token_ids: frozenset[int]

    block_table: BlockTable | None = None
    generated: list[int] = field(default_factory=list)
    prefilled: bool = False
    finished: bool = False

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1; got {self.max_new_tokens}")

    @property
    def last_token(self) -> int:
        """The most recently produced token — the one a decode step feeds back in."""
        if not self.generated:
            raise ValueError(f"request {self.request_id!r} has produced no tokens yet")
        return self.generated[-1]

    def record(self, token_id: int) -> None:
        """Append a sampled token and apply the stop rule (EOS or length cap)."""
        self.generated.append(token_id)
        if token_id in self.eos_token_ids or len(self.generated) >= self.max_new_tokens:
            self.finished = True
