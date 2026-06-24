"""Incremental detokenization must emit clean text deltas and never broken UTF-8.

The streaming-safe contract: decoding token-by-token and concatenating is wrong when a
multibyte character spans several tokens. These tests pin that a character split across tokens
is held back until complete, that the deltas always reassemble to the full text, and that a
finish right after a split is flushed by ``finalize``.
"""

from __future__ import annotations

from llm_infer.serving.server.detokenizer import IncrementalDetokenizer


class ByteSplitTokenizer:
    """A tokenizer whose ids are raw UTF-8 bytes — so a multibyte char spans several tokens.

    ``decode`` joins the bytes and decodes UTF-8 with ``errors="replace"``, exactly the failure
    mode a real tokenizer exhibits when a codepoint is split across token boundaries: a partial
    tail decodes to U+FFFD until the completing byte arrives.
    """

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")


def test_multibyte_character_is_not_emitted_until_complete() -> None:
    # "é" is U+00E9 -> two UTF-8 bytes 0xC3 0xA9; feeding them one at a time must not leak U+FFFD.
    detok = IncrementalDetokenizer(ByteSplitTokenizer())
    first = detok.feed(0xC3)
    assert first == "", "a half-formed multibyte character must be buffered, not emitted"
    assert "�" not in first
    second = detok.feed(0xA9)
    assert second == "é"


def test_emoji_split_across_four_tokens() -> None:
    # "😀" is U+1F600 -> four UTF-8 bytes; only the final byte completes the character.
    detok = IncrementalDetokenizer(ByteSplitTokenizer())
    emoji_bytes = "😀".encode()
    deltas = [detok.feed(b) for b in emoji_bytes]
    assert deltas[:-1] == ["", "", ""]
    assert "".join(deltas) == "😀"
    assert all("�" not in d for d in deltas)


def test_deltas_reassemble_to_full_text() -> None:
    detok = IncrementalDetokenizer(ByteSplitTokenizer())
    text = "café — déjà vu 🚀"
    deltas = [detok.feed(b) for b in text.encode()]
    assert "".join(deltas) == text


def test_finalize_flushes_buffered_tail() -> None:
    # If the stream ends mid-character (truncated), finalize releases whatever decoded so far.
    detok = IncrementalDetokenizer(ByteSplitTokenizer())
    assert detok.feed(ord("a")) == "a"
    assert detok.feed(0xC3) == ""  # dangling lead byte of a 2-byte char, held back
    tail = detok.finalize()
    # The lead byte alone decodes to the replacement char; finalize surfaces it rather than
    # silently dropping output — honest about a genuinely truncated stream.
    assert tail == "�"
