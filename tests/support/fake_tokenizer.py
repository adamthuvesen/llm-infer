"""Hand a partial tokenizer double to code typed against the full tokenizer protocol."""

from __future__ import annotations

from typing import cast

from llm_infer.model.interface import TokenizerLike


def as_tokenizer(fake: object) -> TokenizerLike:
    return cast(TokenizerLike, fake)
