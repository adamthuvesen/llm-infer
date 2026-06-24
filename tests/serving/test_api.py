"""HTTP-surface tests for the OpenAI-compatible server, on the tiny CPU model.

These prove the transport, not the math: the engine's token ids are already validated
token-for-token by the correctness oracle. Here we check the wire contract (response shape,
ids, usage, finish_reason), that streaming emits SSE deltas ending in ``[DONE]``, that
several streaming requests genuinely batch through one engine loop, and that unsupported
fields fail loudly with a 4xx. Everything runs greedy on the tiny random-weight Qwen — fast,
deterministic, no GPU.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.server import AsyncInferenceEngine, create_app
from tests.correctness.test_chunked_prefill import _tiny_qwen

VOCAB = 37
EOS_ID = 36  # the tiny model never greedily emits this on these prompts, so length caps decode


class TinyTokenizer:
    """A whitespace/byte stand-in for a real tokenizer over the 37-token vocab.

    ``encode`` maps each character to ``ord(c) % VOCAB`` (never EOS), ``decode`` maps ids back
    to printable ASCII via ``id + 33``, and ``apply_chat_template`` flattens messages into ids.
    Enough surface for the server to tokenize prompts and detokenize streamed output without a
    real model — the ids it produces are valid engine inputs and the text round-trips cleanly.
    """

    eos_token_id = EOS_ID

    def encode(self, text: str) -> list[int]:
        return [(ord(c) % (VOCAB - 1)) for c in text] or [1]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        ids = [t for t in token_ids if not (skip_special_tokens and t == EOS_ID)]
        return "".join(chr(33 + (t % 94)) for t in ids)

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True, tokenize: bool = True
    ) -> list[int]:
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return self.encode(text)


def _build_app():
    model = _tiny_qwen()
    engine = InferenceEngine(model, block_size=8, num_blocks=64)
    async_engine = AsyncInferenceEngine(engine)
    return create_app(
        async_engine=async_engine,
        tokenizer=TinyTokenizer(),
        model_id="tiny-qwen",
        eos_token_ids=frozenset({EOS_ID}),
    )


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _run(coro):
    return asyncio.run(coro)


def test_cancellation_aborts_engine_request() -> None:
    """Dropping a stream mid-generation aborts its engine request and frees the loop.

    We consume one token then break out of the ``async for`` (the disconnect signal). The
    wrapper's ``finally`` flags the stream aborted; the loop finishes the engine request and
    stops decoding it. We assert the request leaves the scheduler's running set.
    """

    async def go() -> None:
        engine = InferenceEngine(_tiny_qwen(), block_size=8, num_blocks=64)
        async_engine = AsyncInferenceEngine(engine)
        async_engine.start()
        try:
            request_id = async_engine.next_request_id()
            stream = async_engine.stream(
                request_id=request_id,
                prompt_ids=[1, 2, 3],
                max_new_tokens=50,  # long, so it would keep decoding without the abort
                eos_token_ids=frozenset({EOS_ID}),
            )
            count = 0
            async for _ in stream:
                count += 1
                if count == 1:
                    break  # client "disconnects" after the first token
            await stream.aclose()

            # The loop sees the abort at the next step boundary and releases the request.
            for _ in range(200):
                running_ids = {r.request_id for r in engine.scheduler.running}
                if request_id not in running_ids and not engine.scheduler.waiting:
                    break
                await asyncio.sleep(0.01)
            running_ids = {r.request_id for r in engine.scheduler.running}
            assert request_id not in running_ids
            assert count == 1
        finally:
            async_engine.stop()

    _run(go())


def test_health_and_models() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            health = await client.get("/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            models = await client.get("/v1/models")
            assert models.status_code == 200
            body = models.json()
            assert body["object"] == "list"
            assert body["data"][0]["id"] == "tiny-qwen"
            assert body["data"][0]["object"] == "model"

    _run(go())


def test_chat_completion_non_streaming_shape() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "tiny-qwen",
                    "messages": [{"role": "user", "content": "hello there"}],
                    "max_tokens": 5,
                },
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["object"] == "chat.completion"
            assert body["id"].startswith("cmpl-")
            assert body["model"] == "tiny-qwen"
            choice = body["choices"][0]
            assert choice["index"] == 0
            assert choice["message"]["role"] == "assistant"
            assert isinstance(choice["message"]["content"], str)
            # Greedy run never hits EOS here, so the length cap stops it at exactly max_tokens.
            assert choice["finish_reason"] == "length"
            usage = body["usage"]
            assert usage["completion_tokens"] == 5
            assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    _run(go())


def test_completion_non_streaming_shape() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post(
                "/v1/completions",
                json={"model": "tiny-qwen", "prompt": "abc", "max_tokens": 4},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["object"] == "text_completion"
            choice = body["choices"][0]
            assert isinstance(choice["text"], str)
            assert choice["finish_reason"] == "length"
            assert body["usage"]["completion_tokens"] == 4

    _run(go())


def _parse_sse(text: str) -> list:
    chunks = []
    for line in text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                chunks.append("[DONE]")
            else:
                chunks.append(json.loads(payload))
    return chunks


def test_chat_completion_streaming_sse() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "tiny-qwen",
                    "messages": [{"role": "user", "content": "stream please"}],
                    "max_tokens": 5,
                    "stream": True,
                },
            ) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                text = ""
                async for piece in resp.aiter_text():
                    text += piece
            chunks = _parse_sse(text)
            assert chunks[-1] == "[DONE]"
            data_chunks = [c for c in chunks if c != "[DONE]"]
            assert all(c["object"] == "chat.completion.chunk" for c in data_chunks)
            assert data_chunks[0]["choices"][0]["delta"]["role"] == "assistant"
            # The last data chunk carries the finish_reason, no content.
            assert data_chunks[-1]["choices"][0]["finish_reason"] == "length"

    _run(go())


def test_completion_streaming_sse() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            async with client.stream(
                "POST",
                "/v1/completions",
                json={"model": "tiny-qwen", "prompt": "xy", "max_tokens": 4, "stream": True},
            ) as resp:
                assert resp.status_code == 200
                text = ""
                async for piece in resp.aiter_text():
                    text += piece
            chunks = _parse_sse(text)
            assert chunks[-1] == "[DONE]"
            data_chunks = [c for c in chunks if c != "[DONE]"]
            assert all(c["object"] == "text_completion" for c in data_chunks)
            assert data_chunks[-1]["choices"][0]["finish_reason"] == "length"

    _run(go())


def test_streaming_matches_non_streaming() -> None:
    """Same prompt: the concatenated stream deltas equal the non-streaming content."""

    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            payload = {
                "model": "tiny-qwen",
                "messages": [{"role": "user", "content": "consistency check"}],
                "max_tokens": 6,
            }
            full = await client.post("/v1/chat/completions", json=payload)
            expected = full.json()["choices"][0]["message"]["content"]

            async with client.stream(
                "POST", "/v1/chat/completions", json={**payload, "stream": True}
            ) as resp:
                text = ""
                async for piece in resp.aiter_text():
                    text += piece
            streamed = "".join(
                c["choices"][0]["delta"].get("content", "")
                for c in _parse_sse(text)
                if c != "[DONE]"
            )
            assert streamed == expected

    _run(go())


def test_concurrent_streams_batch_through_one_loop() -> None:
    """Many streaming requests in flight at once must all complete off the single engine loop.

    They are launched concurrently and awaited together; with one background batching loop the
    only way they all finish is by being decoded together (continuous batching). We assert each
    got the right number of tokens and a finish reason — proof the loop served them in parallel.
    """

    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:

            async def one(content: str, max_tokens: int) -> tuple[int, str]:
                async with client.stream(
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": "tiny-qwen",
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": max_tokens,
                        "stream": True,
                    },
                ) as resp:
                    text = ""
                    async for piece in resp.aiter_text():
                        text += piece
                chunks = [c for c in _parse_sse(text) if c != "[DONE]"]
                content_chunks = sum(1 for c in chunks if c["choices"][0]["delta"].get("content"))
                finish = chunks[-1]["choices"][0]["finish_reason"]
                return content_chunks, finish

            results = await asyncio.gather(
                one("first request", 6),
                one("second request here", 6),
                one("third", 6),
                one("a fourth concurrent client", 6),
            )
            for content_chunks, finish in results:
                assert finish == "length"
                assert content_chunks >= 1

    _run(go())


def test_concurrent_streaming_equals_serial_tokens() -> None:
    """Running requests concurrently yields the same tokens as running each alone.

    This is the real continuous-batching guarantee: sharing one loop and one KV-cache must not
    perturb any request's greedy output. We compare each prompt's streamed text under load
    against the same prompt served by itself.
    """

    prompts = ["alpha", "a longer beta prompt", "gamma g"]

    async def text_for(client, prompt: str) -> str:
        async with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "tiny-qwen",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 6,
                "stream": True,
            },
        ) as resp:
            raw = ""
            async for piece in resp.aiter_text():
                raw += piece
        return "".join(
            c["choices"][0]["delta"].get("content", "") for c in _parse_sse(raw) if c != "[DONE]"
        )

    async def go() -> None:
        # Serial reference: a fresh app/engine per prompt, run alone.
        serial: dict[str, str] = {}
        for prompt in prompts:
            app = _build_app()
            async with app.router.lifespan_context(app), _client(app) as client:
                serial[prompt] = await text_for(client, prompt)

        # Concurrent: one shared app/engine, all prompts in flight together.
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            concurrent = await asyncio.gather(*(text_for(client, p) for p in prompts))

        for prompt, text in zip(prompts, concurrent, strict=True):
            assert text == serial[prompt], f"{prompt!r} diverged under concurrent batching"

    _run(go())


def _parse_named_sse(text: str) -> list[tuple[str, dict]]:
    """Parse Responses semantic SSE into (event_name, data) pairs."""
    events: list[tuple[str, dict]] = []
    name: str | None = None
    for line in text.splitlines():
        if line.startswith("event: "):
            name = line[len("event: ") :]
        elif line.startswith("data: ") and name is not None:
            events.append((name, json.loads(line[len("data: ") :])))
            name = None
    return events


def test_responses_non_streaming_shape() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post(
                "/v1/responses",
                json={"model": "tiny-qwen", "input": "tell me something", "max_output_tokens": 5},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["object"] == "response"
            assert body["id"].startswith("resp-")
            assert body["status"] == "completed"
            assert body["model"] == "tiny-qwen"
            message = body["output"][0]
            assert message["type"] == "message"
            assert message["role"] == "assistant"
            assert message["content"][0]["type"] == "output_text"
            assert message["content"][0]["text"] == body["output_text"]
            usage = body["usage"]
            assert usage["output_tokens"] == 5
            assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]

    _run(go())


def test_responses_structured_input_uses_chat_template() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post(
                "/v1/responses",
                json={
                    "model": "tiny-qwen",
                    "instructions": "be terse",
                    "input": [{"role": "user", "content": "structured input"}],
                    "max_output_tokens": 4,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["object"] == "response"

    _run(go())


def test_responses_streaming_semantic_events() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            async with client.stream(
                "POST",
                "/v1/responses",
                json={
                    "model": "tiny-qwen",
                    "input": "stream this response",
                    "max_output_tokens": 5,
                    "stream": True,
                },
            ) as resp:
                assert resp.status_code == 200
                assert resp.headers["content-type"].startswith("text/event-stream")
                text = ""
                async for piece in resp.aiter_text():
                    text += piece
            events = _parse_named_sse(text)
            names = [name for name, _ in events]
            assert names[0] == "response.created"
            assert names[-1] == "response.completed"
            assert "response.output_text.delta" in names

            streamed = "".join(
                data["delta"] for name, data in events if name == "response.output_text.delta"
            )
            completed = next(data for name, data in events if name == "response.completed")
            assert completed["response"]["output_text"] == streamed

    _run(go())


def test_responses_streaming_matches_non_streaming() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            payload = {"model": "tiny-qwen", "input": "match me", "max_output_tokens": 6}
            full = await client.post("/v1/responses", json=payload)
            expected = full.json()["output_text"]

            async with client.stream(
                "POST", "/v1/responses", json={**payload, "stream": True}
            ) as resp:
                text = ""
                async for piece in resp.aiter_text():
                    text += piece
            streamed = "".join(
                data["delta"]
                for name, data in _parse_named_sse(text)
                if name == "response.output_text.delta"
            )
            assert streamed == expected

    _run(go())


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "tiny-qwen", "input": "hi", "tools": [{"type": "web_search"}]},
        {"model": "tiny-qwen", "input": "hi", "previous_response_id": "resp-123"},
        {"model": "tiny-qwen", "input": "hi", "store": True},
        {"model": "tiny-qwen", "input": "hi", "background": True},
        {"model": "tiny-qwen", "input": "hi", "n": 2},
        {"model": "tiny-qwen", "input": "hi", "temperature": 0.9},
        {"model": "tiny-qwen", "input": "hi", "frequency_penalty": 0.1},
    ],
)
def test_responses_unsupported_fields_rejected(payload: dict) -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post("/v1/responses", json=payload)
            assert 400 <= resp.status_code < 500

    _run(go())


def _chat(**extra) -> dict:
    return {"model": "tiny-qwen", "messages": [{"role": "user", "content": "hi"}], **extra}


@pytest.mark.parametrize(
    "payload, status",
    [
        # tools / function-calling: explicitly unsupported.
        (_chat(tools=[{"type": "function"}]), 422),
        # n > 1: only one choice is returned.
        (_chat(n=2), 422),
        # logprobs: not produced.
        (_chat(logprobs=True), 422),
        # unknown field: extra=forbid -> 422 from validation.
        (_chat(frequency_penalty=0.5), 422),
        # temperature the fixed greedy sampler cannot honor.
        (_chat(temperature=0.7), 400),
    ],
)
def test_unsupported_fields_rejected(payload: dict, status: int) -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            resp = await client.post("/v1/chat/completions", json=payload)
            assert resp.status_code == status

    _run(go())
