"""The FastAPI app: OpenAI-shaped HTTP over the async engine. Transport only.

Built by :func:`create_app` around an injected engine + tokenizer, so a test wires the tiny
CPU model and production wires the real Qwen through the same code. The handlers tokenize
(chat template for ``/v1/chat/completions``, raw encode for ``/v1/completions``), stream
tokens off the one background batching loop, detokenize incrementally, and shape the result
as OpenAI responses. No decoding logic lives here — token ids come straight from the engine.

Sampling is **per request**: each request's ``temperature``/``top_p``/``top_k``/penalties/
``seed`` are mapped to :class:`SamplingParams` and carried on its engine request, so concurrent
clients each decode under their own params off the one shared batching loop.

``stop`` sequences are an **output-text** stop layered here, where the text is available: the
:class:`StopSequenceDetokenizer` truncates the output before the first stop string (never
leaking it or anything after it, even across token boundaries under streaming), reports
``finish_reason="stop"``, and we abort the engine request so no compute runs past the stop. The
engine's token-level EOS / max-tokens stopping is untouched.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from llm_infer.serving.sampler import SamplingParams
from llm_infer.serving.server.async_engine import AsyncInferenceEngine, TokenStreamItem
from llm_infer.serving.server.metrics import ServerMetrics
from llm_infer.serving.server.protocol import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelList,
    Response,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponsesRequest,
    ResponseUsage,
    SamplingRequestBody,
    Usage,
)
from llm_infer.serving.server.stop import StopSequenceDetokenizer

# Bounds on the OpenAI `stop` field. Beyond these we 400 rather than do unbounded buffering work
# per token; OpenAI itself caps stop at 4 sequences, which is plenty for a from-scratch engine.
_MAX_STOP_SEQUENCES = 4
_MAX_STOP_LENGTH = 256


def create_app(
    *,
    async_engine: AsyncInferenceEngine,
    tokenizer: object,
    model_id: str,
    eos_token_ids: frozenset[int],
    metrics: ServerMetrics | None = None,
) -> FastAPI:
    """Build the serving app around an injected engine, tokenizer, and metrics.

    ``metrics`` should be the same :class:`ServerMetrics` the ``async_engine`` was built with,
    so ``/metrics`` renders the instruments those request/token choke points feed and the live
    gauges already bound to the engine. When omitted, ``/metrics`` reports an empty registry.
    """
    metrics = metrics or ServerMetrics()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async_engine.start()
        try:
            yield
        finally:
            async_engine.stop()

    app = FastAPI(title="llm-infer", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics_endpoint() -> PlainTextResponse:
        # Counters reflect traffic already served; gauges read live engine state at this scrape.
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=model_id, created=int(time.time()))])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        _reject_unsupported(request)
        _check_sampling(request)
        model = _resolve_model(request.model, model_id)
        prompt_ids = _apply_chat_template(tokenizer, request.messages)
        return await _serve(
            request_kind="chat",
            prompt_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            stream=request.stream,
            model=model,
            sampling=_sampling_params(request),
            stop=_stop_sequences(request.stop),
        )

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        _check_sampling(request)
        model = _resolve_model(request.model, model_id)
        if isinstance(request.prompt, list):
            raise HTTPException(400, "batched 'prompt' (list) is not supported; send one string")
        prompt_ids = tokenizer.encode(request.prompt)
        if not prompt_ids:
            raise HTTPException(400, "'prompt' encoded to zero tokens")
        return await _serve(
            request_kind="text",
            prompt_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            stream=request.stream,
            model=model,
            sampling=_sampling_params(request),
            stop=_stop_sequences(request.stop),
        )

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest):
        _reject_responses_unsupported(request)
        model = _resolve_model(request.model, model_id)
        prompt_ids = _responses_prompt_ids(tokenizer, request)
        _assert_capacity(async_engine, prompt_ids, request.max_output_tokens)
        stop = _stop_sequences(request.stop)
        request_id = async_engine.next_request_id()
        created = int(time.time())
        response_id = f"resp-{request_id}"
        token_stream = async_engine.stream(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=request.max_output_tokens,
            eos_token_ids=eos_token_ids,
            sampling=_sampling_params(request),
        )
        if request.stream:
            sse = _stream_responses_sse(
                token_stream=token_stream,
                detok=StopSequenceDetokenizer(tokenizer, stop),
                response_id=response_id,
                created=created,
                model=model,
                input_tokens=len(prompt_ids),
            )
            return StreamingResponse(sse, media_type="text/event-stream")
        return await _collect_response(
            token_stream=token_stream,
            detok=StopSequenceDetokenizer(tokenizer, stop),
            response_id=response_id,
            created=created,
            model=model,
            input_tokens=len(prompt_ids),
        )

    async def _serve(
        *,
        request_kind: str,
        prompt_ids: list[int],
        max_new_tokens: int,
        stream: bool,
        model: str,
        sampling: SamplingParams,
        stop: list[str],
    ):
        _assert_capacity(async_engine, prompt_ids, max_new_tokens)
        request_id = async_engine.next_request_id()
        created = int(time.time())
        completion_id = f"cmpl-{request_id}"
        token_stream = async_engine.stream(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            sampling=sampling,
        )
        if stream:
            sse = _stream_sse(
                request_kind=request_kind,
                token_stream=token_stream,
                detok=StopSequenceDetokenizer(tokenizer, stop),
                completion_id=completion_id,
                created=created,
                model=model,
            )
            return StreamingResponse(sse, media_type="text/event-stream")
        return await _collect(
            request_kind=request_kind,
            token_stream=token_stream,
            detok=StopSequenceDetokenizer(tokenizer, stop),
            completion_id=completion_id,
            created=created,
            model=model,
            prompt_tokens=len(prompt_ids),
        )

    return app


async def _collect(
    *,
    request_kind: str,
    token_stream: AsyncIterator[TokenStreamItem],
    detok: StopSequenceDetokenizer,
    completion_id: str,
    created: int,
    model: str,
    prompt_tokens: int,
):
    """Drain the token stream and assemble a single non-streaming response.

    A stop string truncates the text before the stop, sets ``finish_reason="stop"``, and aborts
    the engine request so no compute is wasted past the stop (see :func:`_finish_on_stop`).
    """
    text_parts: list[str] = []
    finish_reason = "length"
    completion_tokens = 0
    async for item in token_stream:
        completion_tokens += 1
        fed = detok.feed(item.token_id)
        text_parts.append(fed.text)
        if fed.stopped:
            finish_reason = "stop"
            await _finish_on_stop(token_stream)
            break
        if item.finish_reason is not None:
            finish_reason = item.finish_reason
    else:
        text_parts.append(detok.finalize())
    text = "".join(text_parts)
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    if request_kind == "chat":
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=text), finish_reason=finish_reason
                )
            ],
            usage=usage,
        )
    return CompletionResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[CompletionChoice(text=text, finish_reason=finish_reason)],
        usage=usage,
    )


async def _collect_response(
    *,
    token_stream: AsyncIterator[TokenStreamItem],
    detok: StopSequenceDetokenizer,
    response_id: str,
    created: int,
    model: str,
    input_tokens: int,
):
    """Drain the stream into one Responses ``response`` object (a single assistant message)."""
    text_parts: list[str] = []
    output_tokens = 0
    async for item in token_stream:
        output_tokens += 1
        fed = detok.feed(item.token_id)
        text_parts.append(fed.text)
        if fed.stopped:
            await _finish_on_stop(token_stream)
            break
    else:
        text_parts.append(detok.finalize())
    text = "".join(text_parts)
    return Response(
        id=response_id,
        created_at=created,
        model=model,
        output=[
            ResponseOutputMessage(content=[ResponseOutputText(text=text)]),
        ],
        output_text=text,
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


async def _stream_responses_sse(
    *,
    token_stream: AsyncIterator[TokenStreamItem],
    detok: StopSequenceDetokenizer,
    response_id: str,
    created: int,
    model: str,
    input_tokens: int,
) -> AsyncIterator[str]:
    """Emit the Responses semantic SSE events, not chat chunks.

    The minimal text-generation lifecycle: ``response.created`` once, a
    ``response.output_text.delta`` per decodable text chunk (carrying ``delta``), then a
    terminal ``response.completed`` whose payload includes the assembled ``response`` object. A
    stop string truncates the text before the stop and halts generation; the stop string is never
    present in any emitted delta or in the completed payload.
    """
    created_response = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "model": model,
        "status": "in_progress",
    }
    yield _sse_event("response.created", {"response": created_response})

    text_parts: list[str] = []
    output_tokens = 0
    async for item in token_stream:
        output_tokens += 1
        fed = detok.feed(item.token_id)
        if fed.text:
            text_parts.append(fed.text)
            yield _sse_event(
                "response.output_text.delta",
                {"response_id": response_id, "delta": fed.text},
            )
        if fed.stopped:
            await _finish_on_stop(token_stream)
            break
    else:
        tail = detok.finalize()
        if tail:
            text_parts.append(tail)
            yield _sse_event(
                "response.output_text.delta", {"response_id": response_id, "delta": tail}
            )

    text = "".join(text_parts)
    completed = Response(
        id=response_id,
        created_at=created,
        model=model,
        output=[ResponseOutputMessage(content=[ResponseOutputText(text=text)])],
        output_text=text,
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )
    yield _sse_event("response.completed", {"response": completed.model_dump()})


def _sse_event(event: str, payload: dict) -> str:
    """A named SSE event (``event:``/``data:`` pair) for the Responses semantic stream."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def _stream_sse(
    *,
    request_kind: str,
    token_stream: AsyncIterator[TokenStreamItem],
    detok: StopSequenceDetokenizer,
    completion_id: str,
    created: int,
    model: str,
) -> AsyncIterator[str]:
    """Emit OpenAI SSE chunks: per-token deltas, a finish chunk, then ``[DONE]``.

    A stop string truncates the deltas before the stop, sets ``finish_reason="stop"``, and halts
    generation. The hold-back buffer guarantees the stop string never appears in any delta, even
    when it straddles two tokens, so the concatenated deltas equal the non-streaming text exactly.
    """
    if request_kind == "chat":
        first = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionDelta(role="assistant"))],
        )
        yield _sse(first)

    finish_reason: str | None = None
    async for item in token_stream:
        fed = detok.feed(item.token_id)
        if item.finish_reason is not None:
            finish_reason = item.finish_reason
        if fed.text:
            yield _sse(_delta_chunk(request_kind, completion_id, created, model, fed.text))
        if fed.stopped:
            finish_reason = "stop"
            await _finish_on_stop(token_stream)
            break
    else:
        tail = detok.finalize()
        if tail:
            yield _sse(_delta_chunk(request_kind, completion_id, created, model, tail))

    yield _sse(
        _finish_chunk(request_kind, completion_id, created, model, finish_reason or "length")
    )
    yield "data: [DONE]\n\n"


def _delta_chunk(
    request_kind: str, completion_id: str, created: int, model: str, text: str
) -> ChatCompletionChunk | CompletionChunk:
    if request_kind == "chat":
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionDelta(content=text))],
        )
    return CompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[CompletionChoice(text=text, finish_reason=None)],
    )


def _finish_chunk(
    request_kind: str, completion_id: str, created: int, model: str, finish_reason: str
) -> ChatCompletionChunk | CompletionChunk:
    if request_kind == "chat":
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChunkChoice(delta=ChatCompletionDelta(), finish_reason=finish_reason)
            ],
        )
    return CompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[CompletionChoice(text="", finish_reason=finish_reason)],
    )


def _sse(chunk: ChatCompletionChunk | CompletionChunk) -> str:
    # exclude_none keeps delta chunks faithful to OpenAI's streaming shape — a role-only or
    # finish-only chunk carries no null content key, so consumers concatenate deltas cleanly.
    return f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"


def _template_token_ids(rendered: object) -> list[int]:
    """Flatten ``apply_chat_template(tokenize=True)`` output into a list of int ids.

    Real HF tokenizers return a ``BatchEncoding`` (``{"input_ids": [...]}``); a stand-in may
    return a bare list. Iterating a mapping yields its *keys*, not its ids, so read
    ``input_ids`` explicitly, drop a leading batch dimension if present, and coerce to int.
    """
    ids = rendered["input_ids"] if isinstance(rendered, Mapping) else rendered
    ids = list(ids)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(token) for token in ids]


def _apply_chat_template(tokenizer: object, messages: list[ChatMessage]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        add_generation_prompt=True,
        tokenize=True,
    )
    ids = _template_token_ids(rendered)
    if not ids:
        raise HTTPException(400, "chat template produced zero tokens")
    return ids


def _resolve_model(requested: str, served: str) -> str:
    """Validate the requested model against the one this server actually serves.

    This is a single-model server, so any other id is a 404 (matching OpenAI / vLLM): a client
    never gets a response that silently claims a model we did not run. Returns the canonical
    served id, which every response then reports — we answer with what ran, not what was asked.
    """
    if requested != served:
        raise HTTPException(404, f"model {requested!r} not found; this server serves {served!r}")
    return served


def _assert_capacity(
    async_engine: AsyncInferenceEngine, prompt_ids: list[int], max_new_tokens: int
) -> None:
    """Reject a request too large for the KV pool with a clean 400, before it reaches the loop."""
    try:
        async_engine.assert_admissible(prompt_len=len(prompt_ids), max_new_tokens=max_new_tokens)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _reject_unsupported(request: ChatCompletionRequest) -> None:
    if request.tools is not None or request.functions is not None:
        raise HTTPException(422, "tools / function-calling are not supported")


def _check_sampling(request: SamplingRequestBody) -> None:
    """Reject the surface this engine cannot produce (``n > 1``, ``logprobs``).

    Temperature/top-p/top-k/penalties/seed are honored per request — see :func:`_sampling_params`
    — and ``stop`` is honored as output-text truncation (see :func:`_stop_sequences`), so neither
    is checked against any server-wide sampler.
    """
    if request.n != 1:
        raise HTTPException(422, "'n' > 1 is not supported; this server returns a single choice")
    if request.logprobs not in (None, False, 0):
        raise HTTPException(422, "'logprobs' is not supported")


def _stop_sequences(stop: str | list[str] | None) -> list[str]:
    """Normalize the OpenAI ``stop`` field (string, list, or absent) into a validated list.

    A bare string becomes a one-element list; ``None`` becomes empty. Non-string members, too
    many sequences, or an over-long sequence are rejected with a clear 400 — malformed stop input
    fails loudly rather than being silently coerced. Empty strings are dropped (they can never
    match meaningfully) by the stop-aware detokenizer.
    """
    if stop is None:
        return []
    sequences = [stop] if isinstance(stop, str) else list(stop)
    if len(sequences) > _MAX_STOP_SEQUENCES:
        raise HTTPException(
            400, f"'stop' accepts at most {_MAX_STOP_SEQUENCES} sequences; got {len(sequences)}"
        )
    for sequence in sequences:
        if not isinstance(sequence, str):
            raise HTTPException(400, "'stop' sequences must be strings")
        if len(sequence) > _MAX_STOP_LENGTH:
            raise HTTPException(
                400, f"each 'stop' sequence may be at most {_MAX_STOP_LENGTH} characters"
            )
    return sequences


async def _finish_on_stop(token_stream: AsyncIterator[TokenStreamItem]) -> None:
    """Halt generation on a stop hit: close the stream so the engine aborts the request.

    Closing the async generator fires its ``finally``, which flags the stream aborted; the
    background loop sees it at the next step boundary and calls ``engine.abort`` — freeing the
    request's KV so no compute is spent past the stop. This reuses the exact disconnect path, so
    a stop hit and a client disconnect halt the engine identically.
    """
    await token_stream.aclose()


def _sampling_params(request: SamplingRequestBody | ResponsesRequest) -> SamplingParams:
    """Map the request body's OpenAI sampling fields to per-request :class:`SamplingParams`.

    Out-of-range values raise in ``SamplingParams.__post_init__``; we surface that as a 400
    rather than letting it become a 500, so a malformed request fails loudly and clearly.
    """
    try:
        return SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            seed=request.seed,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _responses_prompt_ids(tokenizer: object, request: ResponsesRequest) -> list[int]:
    """Tokenize a Responses ``input`` (+ optional ``instructions``) into prompt ids.

    A bare string with no instructions is encoded raw — the caller asked for plain
    continuation. Anything structured (a list of input items) or carrying instructions goes
    through the chat template, with ``instructions`` mapped to a system message.
    """
    if isinstance(request.input, str) and request.instructions is None:
        prompt_ids = tokenizer.encode(request.input)
        if not prompt_ids:
            raise HTTPException(400, "'input' encoded to zero tokens")
        return list(prompt_ids)

    messages: list[dict[str, str]] = []
    if request.instructions is not None:
        messages.append({"role": "system", "content": request.instructions})
    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    else:
        # 'developer' is the Responses analogue of a system message; map it so the
        # template (which only knows system/user/assistant) renders it as guidance.
        for item in request.input:
            role = "system" if item.role == "developer" else item.role
            messages.append({"role": role, "content": item.content})
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    ids = _template_token_ids(rendered)
    if not ids:
        raise HTTPException(400, "chat template produced zero tokens")
    return ids


def _reject_responses_unsupported(request: ResponsesRequest) -> None:
    """Reject the stateful / hosted-tool Responses features a from-scratch engine cannot do."""
    if request.tools is not None:
        raise HTTPException(
            422, "hosted tools (web_search/file_search/code_interpreter) are not supported"
        )
    if request.previous_response_id is not None:
        raise HTTPException(
            422, "'previous_response_id' / server-side conversation state is not supported"
        )
    if request.store:
        raise HTTPException(422, "'store' (server-side response persistence) is not supported")
    if request.background:
        raise HTTPException(422, "'background' responses are not supported")
    if request.n != 1:
        raise HTTPException(422, "'n' > 1 is not supported; this server returns a single response")
