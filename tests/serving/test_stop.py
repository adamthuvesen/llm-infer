"""Stop sequences must truncate output before the stop and never leak it — even under streaming.

Two layers. First, unit tests pin the :class:`StopSequenceDetokenizer` hold-back logic directly,
where we control the exact token stream: a stop string that straddles two tokens is never present
in any emitted delta, earliest-stop-wins, and absent-stop output is untouched. Second,
HTTP-surface tests drive the real app handlers through a *scripted* async engine that emits a
fixed token sequence, so we can construct the straddling case end to end and assert the
streamed deltas equal the non-streaming truncated text exactly, the engine request is aborted on
the stop, and an over-limit ``stop`` is a 400. Token ids are already validated by the oracle; the
scripted engine is only a deterministic token source for the transport-level stop contract.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest

from llm_infer.serving.server import create_app
from llm_infer.serving.server.async_engine import TokenStreamItem
from llm_infer.serving.server.stop import StopSequenceDetokenizer


class CharTokenizer:
    """One token id per ASCII codepoint: ``decode`` is ``chr`` per id, so tokens map to chars.

    This makes the token/text boundary explicit — each id is exactly one character — so a stop
    string is "split across two tokens" precisely when its characters arrive on separate feeds.
    """

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text] or [ord(" ")]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(t) for t in token_ids)

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True, tokenize: bool = True
    ) -> list[int]:
        return self.encode(messages[-1]["content"])


def _ids(text: str) -> list[int]:
    return [ord(c) for c in text]


# --- unit: the hold-back buffer, in isolation ---------------------------------------------


def test_stop_split_across_two_tokens_is_never_leaked() -> None:
    """A stop string whose chars arrive on separate feeds must never appear in any delta."""
    detok = StopSequenceDetokenizer(CharTokenizer(), ["END"])
    emitted: list[str] = []
    stopped_at: int | None = None
    # "ab" then the stop "END" one char per token, then "zz" that must never be reached.
    for i, char in enumerate("abENDzz"):
        fed = detok.feed(ord(char))
        emitted.append(fed.text)
        if fed.stopped:
            stopped_at = i
            break
    text = "".join(emitted)
    assert text == "ab"  # everything before the stop, nothing of the stop or after
    assert "END" not in text
    assert "E" not in text[2:]  # the held-back "E"/"EN" prefix was discarded, not emitted
    assert stopped_at == 4  # the 'D' that completes "END"


def test_partial_stop_prefix_is_held_then_released_when_it_does_not_complete() -> None:
    """A trailing run that looks like a stop prefix is held, then flushed once it cannot match."""
    detok = StopSequenceDetokenizer(CharTokenizer(), ["END"])
    # "EN" looks like the start of "END" and must be held back...
    assert detok.feed(ord("E")).text == ""
    assert detok.feed(ord("N")).text == ""
    # ...but the next char is 'X', so "ENX" can never be a stop — release the held-back text.
    fed = detok.feed(ord("X"))
    assert not fed.stopped
    assert fed.text == "ENX"


def test_no_stop_passes_through_and_finalizes() -> None:
    detok = StopSequenceDetokenizer(CharTokenizer(), [])
    assert detok.feed(ord("h")).text == "h"
    assert detok.feed(ord("i")).text == "i"
    assert detok.finalize() == ""


def test_finalize_flushes_held_back_prefix_when_stream_ends_without_stop() -> None:
    """If the stream ends while a stop *prefix* is buffered, finalize releases it (no stop hit)."""
    detok = StopSequenceDetokenizer(CharTokenizer(), ["END"])
    assert detok.feed(ord("a")).text == "a"
    assert detok.feed(ord("E")).text == ""  # possible start of "END", held back
    assert detok.feed(ord("N")).text == ""  # still a prefix, held back
    assert detok.finalize() == "EN"  # stream ended; "EN" never completed, so it is real output


def test_earliest_of_multiple_stops_wins() -> None:
    detok = StopSequenceDetokenizer(CharTokenizer(), ["XY", "B"])
    emitted: list[str] = []
    for char in "aBXY":
        fed = detok.feed(ord(char))
        emitted.append(fed.text)
        if fed.stopped:
            break
    # "B" occurs before "XY", so the earlier match wins and truncates at "a".
    assert "".join(emitted) == "a"


def test_feeds_reassemble_to_truncated_text_across_a_straddle() -> None:
    """The concatenated feed deltas equal the text truncated at the first stop, exactly."""
    detok = StopSequenceDetokenizer(CharTokenizer(), ["<stop>"])
    full = "hello <stop> world"
    emitted: list[str] = []
    for char in full:
        fed = detok.feed(ord(char))
        emitted.append(fed.text)
        if fed.stopped:
            break
    assert "".join(emitted) == "hello "
    assert "<stop>" not in "".join(emitted)


# --- HTTP surface: scripted engine, real handlers -----------------------------------------


class ScriptedAsyncEngine:
    """A stand-in for :class:`AsyncInferenceEngine` that emits a fixed token sequence.

    Implements only what :func:`create_app` uses — ``start``/``stop``/``next_request_id``/
    ``stream`` — so the real handlers, detokenizer, and SSE shaping run unchanged over a
    deterministic token source. ``aborted`` records that a stop hit closed the stream (the abort
    path), proving generation halts instead of running to ``max_new_tokens``.
    """

    def __init__(self, token_ids: list[int], eos_id: int) -> None:
        self._token_ids = token_ids
        self._eos_id = eos_id
        self._counter = 0
        self.aborted = False

    def start(self) -> None:  # noqa: D401 - lifecycle no-op
        return None

    def stop(self) -> None:
        return None

    def next_request_id(self) -> str:
        self._counter += 1
        return f"req-{self._counter}"

    async def stream(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        max_new_tokens: int,
        eos_token_ids,
        sampling=None,
    ) -> AsyncIterator[TokenStreamItem]:
        try:
            for index, token_id in enumerate(self._token_ids[:max_new_tokens]):
                await asyncio.sleep(0)  # yield control, mimicking the real async boundary
                is_last = index == min(len(self._token_ids), max_new_tokens) - 1
                reason = None
                if token_id == self._eos_id:
                    reason = "stop"
                    is_last = True
                elif is_last:
                    reason = "length"
                yield TokenStreamItem(token_id=token_id, finish_reason=reason)
                if reason is not None:
                    return
        finally:
            # The handler closes the generator on a stop hit; record that the abort path ran.
            self.aborted = True


def _scripted_app(text: str, *, eos_id: int = 0):
    """An app whose engine emits ``text`` one character-token at a time (no EOS unless given)."""
    engine = ScriptedAsyncEngine(_ids(text), eos_id=eos_id)
    app = create_app(
        async_engine=engine,
        tokenizer=CharTokenizer(),
        model_id="scripted",
        eos_token_ids=frozenset({eos_id}),
    )
    return app, engine


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _parse_sse(text: str) -> list:
    chunks = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            chunks.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return chunks


def _stream_content(raw: str) -> str:
    return "".join(
        c["choices"][0]["delta"].get("content", "") for c in _parse_sse(raw) if c != "[DONE]"
    )


def test_non_streaming_truncates_before_stop_with_finish_reason_stop() -> None:
    async def go() -> None:
        # The model "would" emit this, but the stop "STOP" cuts it before the stop string.
        app, engine = _scripted_app("hello STOP everything after is dropped")
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "scripted",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                    "stop": "STOP",
                },
            )
            assert resp.status_code == 200
            choice = resp.json()["choices"][0]
            assert choice["message"]["content"] == "hello "
            assert "STOP" not in choice["message"]["content"]
            assert choice["finish_reason"] == "stop"
        # Generation halted on the stop, not at max_tokens — the abort path ran.
        assert engine.aborted

    asyncio.run(go())


def test_streaming_deltas_equal_non_streaming_and_never_leak_the_stop() -> None:
    """The straddle case end to end: stop spans tokens/deltas; concatenated deltas == truncation.

    "STOP" arrives as four separate character-tokens, so it can only straddle delta boundaries.
    The streamed content must equal the non-streaming truncated text and must never contain the
    stop string in any single delta.
    """

    async def go() -> None:
        body = {
            "model": "scripted",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100,
            "stop": "STOP",
        }

        app, _ = _scripted_app("abc STOP xyz")
        async with app.router.lifespan_context(app), _client(app) as client:
            full = await client.post("/v1/chat/completions", json=body)
            non_streaming = full.json()["choices"][0]["message"]["content"]

        app, engine = _scripted_app("abc STOP xyz")
        async with app.router.lifespan_context(app), _client(app) as client:
            async with client.stream(
                "POST", "/v1/chat/completions", json={**body, "stream": True}
            ) as resp:
                raw = ""
                async for piece in resp.aiter_text():
                    raw += piece
            chunks = [c for c in _parse_sse(raw) if c != "[DONE]"]

        streamed = _stream_content(raw)
        assert streamed == non_streaming == "abc "
        # No individual delta contains the stop string or any character after it.
        for chunk in chunks:
            assert "STOP" not in chunk["choices"][0]["delta"].get("content", "")
            assert "xyz" not in chunk["choices"][0]["delta"].get("content", "")
        # The finish chunk reports the stop, and the engine request was aborted.
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert engine.aborted

    asyncio.run(go())


def test_stop_absent_leaves_completion_unaffected() -> None:
    async def go() -> None:
        app, _ = _scripted_app("plain output here", eos_id=0)
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post(
                "/v1/completions",
                json={"model": "scripted", "prompt": "go", "max_tokens": 100},
            )
            assert resp.status_code == 200
            choice = resp.json()["choices"][0]
            assert choice["text"] == "plain output here"
            assert choice["finish_reason"] == "length"

    asyncio.run(go())


def test_stop_as_string_and_as_list_both_truncate() -> None:
    async def go() -> None:
        for stop in ("END", ["END"]):
            app, _ = _scripted_app("keep END drop")
            async with app.router.lifespan_context(app), _client(app) as client:
                resp = await client.post(
                    "/v1/completions",
                    json={"model": "scripted", "prompt": "go", "max_tokens": 100, "stop": stop},
                )
                assert resp.status_code == 200
                choice = resp.json()["choices"][0]
                assert choice["text"] == "keep ", f"stop={stop!r} did not truncate"
                assert choice["finish_reason"] == "stop"

    asyncio.run(go())


def test_earliest_of_multiple_stops_wins_over_http() -> None:
    async def go() -> None:
        app, _ = _scripted_app("aa FIRST bb SECOND cc")
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post(
                "/v1/completions",
                json={
                    "model": "scripted",
                    "prompt": "go",
                    "max_tokens": 100,
                    "stop": ["SECOND", "FIRST"],
                },
            )
            assert resp.status_code == 200
            choice = resp.json()["choices"][0]
            assert choice["text"] == "aa "  # FIRST occurs earliest, so it wins
            assert choice["finish_reason"] == "stop"

    asyncio.run(go())


def test_responses_endpoint_honors_stop() -> None:
    async def go() -> None:
        app, _ = _scripted_app("answer HALT secret")
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post(
                "/v1/responses",
                json={
                    "model": "scripted",
                    "input": "ask",
                    "max_output_tokens": 100,
                    "stop": "HALT",
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["output_text"] == "answer "
            assert "HALT" not in body["output_text"]

    asyncio.run(go())


@pytest.mark.parametrize(
    "stop",
    [
        ["a", "b", "c", "d", "e"],  # too many sequences (> 4)
        "x" * 257,  # single sequence too long (> 256 chars)
    ],
)
def test_over_limit_stop_is_400(stop) -> None:
    async def go() -> None:
        app, _ = _scripted_app("whatever")
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "scripted",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stop": stop,
                },
            )
            assert resp.status_code == 400

    asyncio.run(go())
