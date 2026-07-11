"""Incremental detokenization: turn a growing token-id stream into clean text deltas.

Decoding tokens one at a time and concatenating is wrong — many tokenizers merge across
token boundaries (a leading space, a byte-pair split across two ids, a multibyte UTF-8
character spanning several tokens). Decoding the *whole* history each step is correct but
quadratic in output length: token 500 pays for re-decoding 499 already-emitted ids.

The streaming-safe and bounded pattern decodes a suffix window with a stable left anchor:
``_prefix_offset`` marks where the window starts and ``_read_offset`` marks the ids whose
text has already been emitted. Each :meth:`feed` decodes the window twice — once up to
``_read_offset`` and once to the end — and the emitted delta is the difference. Both
decodes share the same left boundary, so any boundary artifact the tokenizer introduces at
the anchor cancels out of the delta. The anchor advances only when text is emitted, so a
half-formed multibyte tail (a trailing U+FFFD) stays buffered — with its full left context —
until the completing token arrives. The server never emits broken UTF-8, and each feed
decodes only the last emit's worth of ids, not the whole history.
"""

from __future__ import annotations

from llm_infer.model.interface import TokenizerLike


class IncrementalDetokenizer:
    """Decode an append-only token stream into UTF-8-safe text deltas, one request's worth."""

    def __init__(self, tokenizer: TokenizerLike) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        # Suffix-window bounds: ids before _prefix_offset are settled text and never decoded
        # again; ids in [_prefix_offset, _read_offset) were emitted by the previous feed and
        # anchor the delta comparison; ids from _read_offset on are not yet emitted.
        self._prefix_offset = 0
        self._read_offset = 0

    def feed(self, token_id: int) -> str:
        """Append one token; return the newly decodable text (possibly empty)."""
        self._token_ids.append(token_id)
        window = self._decode(self._token_ids[self._prefix_offset :])
        # A trailing U+FFFD means the last token only carried part of a multibyte character;
        # hold the whole tail back until the completing token lands and the replacement clears.
        if window.endswith("�"):
            return ""
        return self._emit(window)

    def finalize(self) -> str:
        """Flush any buffered tail once the stream is complete (e.g. a stop right after a split)."""
        window = self._decode(self._token_ids[self._prefix_offset :])
        return self._emit(window)

    def _emit(self, window: str) -> str:
        """Release the window text past what the previous emit already covered, then advance."""
        emitted = self._decode(self._token_ids[self._prefix_offset : self._read_offset])
        delta = window[len(emitted) :]
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._token_ids)
        return delta

    def _decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=True)
