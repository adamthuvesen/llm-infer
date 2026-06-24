"""Incremental detokenization: turn a growing token-id stream into clean text deltas.

Decoding tokens one at a time and concatenating is wrong — many tokenizers merge across
token boundaries (a leading space, a byte-pair split across two ids, a multibyte UTF-8
character spanning several tokens). The streaming-safe pattern is to decode the *whole*
id list each step and emit only the text that is genuinely new, while holding back any
trailing fragment that is not yet a complete, valid character.

So each :meth:`feed` decodes the accumulated ids, finds the common prefix already emitted,
and releases the new suffix only up to the last codepoint boundary — a half-formed
multibyte tail stays buffered until the token that completes it arrives. The server never
emits broken UTF-8.
"""

from __future__ import annotations


class IncrementalDetokenizer:
    """Decode an append-only token stream into UTF-8-safe text deltas, one request's worth."""

    def __init__(self, tokenizer: object) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._emitted_len = 0

    def feed(self, token_id: int) -> str:
        """Append one token; return the newly decodable text (possibly empty)."""
        self._token_ids.append(token_id)
        text = self._decode(self._token_ids)
        # A trailing U+FFFD means the last token only carried part of a multibyte character;
        # hold the whole tail back until the completing token lands and the replacement clears.
        if text.endswith("�"):
            return ""
        delta = text[self._emitted_len :]
        self._emitted_len = len(text)
        return delta

    def finalize(self) -> str:
        """Flush any buffered tail once the stream is complete (e.g. a stop right after a split)."""
        text = self._decode(self._token_ids)
        delta = text[self._emitted_len :]
        self._emitted_len = len(text)
        return delta

    def _decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
