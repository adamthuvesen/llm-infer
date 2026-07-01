"""Output-text stop sequences, layered over incremental detokenization without leaking text.

OpenAI ``stop`` halts generation when any configured stop string appears in the *generated*
text; the stop string and everything after it are never returned, and the request finishes with
``finish_reason="stop"``. Stop matches the output only — the prompt is never inspected here.

The subtlety is streaming. A naive "emit each token's delta, then check for a stop" leaks: a
stop string can straddle two tokens (so it is never visible in any single delta), and even when
the whole stop has not yet arrived, a trailing fragment of an already-emitted delta might turn
out to be its start. The fix is a small hold-back buffer around the existing
:class:`IncrementalDetokenizer`: decoded text accumulates in ``_pending``, and on each feed we
emit only the prefix that *cannot* be the start of any stop string, holding back up to
``max(len(stop)) - 1`` trailing characters. When a stop completes we emit the text up to it and
report the hit; when the stream ends with no stop we flush whatever is held back. So no delta —
streamed or accumulated — ever contains a stop string or any text past it.

This is purely an output-text stop on top of the engine's token-level EOS / max-tokens
stopping, which is left untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_infer.model.interface import TokenizerLike
from llm_infer.serving.server.detokenizer import IncrementalDetokenizer


@dataclass
class StopFeed:
    """The result of feeding one token through the stop-aware detokenizer.

    ``text`` is the safe-to-emit delta (possibly empty). ``stopped`` is true once a stop string
    has completed: the caller emits ``text`` (the run up to but excluding the stop), then halts.
    """

    text: str
    stopped: bool


class StopSequenceDetokenizer:
    """Wrap :class:`IncrementalDetokenizer` with leak-proof stop-string truncation.

    With no stop strings this is a thin pass-through over the inner detokenizer. With stop
    strings it buffers decoded text and releases only what can never be part of, or follow, a
    stop match — so the stop string itself and any text after it are never emitted, including
    when a stop straddles token or chunk boundaries.
    """

    def __init__(self, tokenizer: TokenizerLike, stop: list[str]) -> None:
        # Empty stop strings can never match meaningfully (and a 0-length hold-back is a no-op),
        # so drop them; the server validates the count/length before we get here.
        self._stop = [s for s in stop if s]
        self._detok = IncrementalDetokenizer(tokenizer)
        self._pending = ""
        self._hold_back = max((len(s) for s in self._stop), default=0)
        self._stopped = False

    def feed(self, token_id: int) -> StopFeed:
        """Append one token; return the safe delta and whether a stop string just completed."""
        if self._stopped:
            return StopFeed(text="", stopped=True)
        self._pending += self._detok.feed(token_id)
        return self._consume()

    def finalize(self) -> str:
        """Flush remaining held-back text once the stream ends with no stop hit.

        After a stop has fired there is nothing to flush — the buffer was truncated at the stop
        and the rest discarded. Otherwise drain the inner detokenizer's tail and release all
        pending text, since no further token can complete a stop.
        """
        if self._stopped:
            return ""
        self._pending += self._detok.finalize()
        flushed = self._pending
        self._pending = ""
        return flushed

    def _consume(self) -> StopFeed:
        """Cut ``_pending`` at the earliest stop, or release all but a possible stop prefix."""
        if not self._stop:
            emit, self._pending = self._pending, ""
            return StopFeed(text=emit, stopped=False)

        cut = self._earliest_stop()
        if cut is not None:
            emit = self._pending[:cut]
            self._pending = ""
            self._stopped = True
            return StopFeed(text=emit, stopped=True)

        safe = len(self._pending) - self._unsafe_tail_len()
        emit = self._pending[:safe]
        self._pending = self._pending[safe:]
        return StopFeed(text=emit, stopped=False)

    def _earliest_stop(self) -> int | None:
        """Index of the earliest complete stop occurrence in ``_pending``, or ``None``."""
        first: int | None = None
        for stop in self._stop:
            index = self._pending.find(stop)
            if index != -1 and (first is None or index < first):
                first = index
        return first

    def _unsafe_tail_len(self) -> int:
        """How many trailing chars to hold back: the longest stop-prefix the tail could begin.

        If the buffer ends with a string that is a *prefix* of some stop string, the next token
        might complete that stop, so those trailing chars are not yet safe to emit. We hold back
        the longest such suffix-that-is-a-stop-prefix (bounded by ``max len(stop) - 1``, since a
        full match would have been caught by :meth:`_earliest_stop`).
        """
        max_check = min(len(self._pending), self._hold_back - 1)
        for length in range(max_check, 0, -1):
            tail = self._pending[-length:]
            if any(stop.startswith(tail) for stop in self._stop):
                return length
        return 0
