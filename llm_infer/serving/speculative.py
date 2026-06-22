"""Prompt-lookup draft tokens for the first speculative-decoding slice."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeculativeDecodingConfig:
    """Configuration for prompt-lookup speculative decoding.

    The draft source is deliberately narrow: find a repeated n-gram suffix in the
    already-seen prompt/history and copy the following tokens as a cheap draft.
    """

    max_draft_tokens: int = 4
    max_ngram_size: int = 4

    def __post_init__(self) -> None:
        if self.max_draft_tokens < 1:
            raise ValueError(f"max_draft_tokens must be >= 1; got {self.max_draft_tokens}")
        if self.max_ngram_size < 1:
            raise ValueError(f"max_ngram_size must be >= 1; got {self.max_ngram_size}")


@dataclass(frozen=True)
class PromptLookupDraft:
    """Draft tokens by copying the continuation after a prior matching suffix."""

    config: SpeculativeDecodingConfig

    def draft(self, context: list[int], *, max_tokens: int | None = None) -> list[int]:
        """Return up to ``max_tokens`` draft ids copied from earlier in ``context``."""
        limit = self.config.max_draft_tokens if max_tokens is None else max_tokens
        limit = min(limit, self.config.max_draft_tokens)
        if limit < 1 or len(context) < 2:
            return []

        max_ngram = min(self.config.max_ngram_size, len(context) - 1)
        for ngram_size in range(max_ngram, 0, -1):
            suffix_start = len(context) - ngram_size
            suffix = context[suffix_start:]
            for start in range(suffix_start - 1, -1, -1):
                end = start + ngram_size
                if context[start:end] != suffix:
                    continue
                draft = context[end : min(len(context), end + limit)]
                if draft:
                    return draft
        return []
