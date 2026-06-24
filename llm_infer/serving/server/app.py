"""The FastAPI app: OpenAI-shaped HTTP over the async engine. Transport only.

Built by :func:`create_app` around an injected engine + tokenizer, so a test wires the tiny
CPU model and production wires the real Qwen through the same code. The handlers tokenize
(chat template for ``/v1/chat/completions``, raw encode for ``/v1/completions``), stream
tokens off the one background batching loop, detokenize incrementally, and shape the result
as OpenAI responses. No decoding logic lives here — token ids come straight from the engine.

The engine runs one fixed sampler (greedy by default). Per-request ``temperature``/``top_p``
that disagree with it are rejected rather than silently ignored: this server does not vary
sampling per request, and pretending otherwise would be a lie.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from llm_infer.serving.sampler import Sampler
from llm_infer.serving.server.async_engine import AsyncInferenceEngine, TokenStreamItem
from llm_infer.serving.server.detokenizer import IncrementalDetokenizer
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
    SamplingParams,
    Usage,
)

_STOP_UNSUPPORTED = "'stop' sequences are not supported; generation stops on EOS/length"


def create_app(
    *,
    async_engine: AsyncInferenceEngine,
    tokenizer: object,
    model_id: str,
    eos_token_ids: frozenset[int],
    sampler: Sampler | None = None,
) -> FastAPI:
    """Build the serving app around an injected engine, tokenizer, and sampler config."""
    sampler = sampler or Sampler()

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

    @app.get("/v1/models")
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=model_id, created=int(time.time()))])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        _reject_unsupported(request)
        _check_sampling(request, sampler)
        prompt_ids = _apply_chat_template(tokenizer, request.messages)
        return await _serve(
            request_kind="chat",
            prompt_ids=prompt_ids,
            max_new_tokens=request.max_tokens,
            stream=request.stream,
            model=request.model,
        )

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        _check_sampling(request, sampler)
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
            model=request.model,
        )

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest):
        _reject_responses_unsupported(request)
        _check_responses_sampling(request, sampler)
        prompt_ids = _responses_prompt_ids(tokenizer, request)
        request_id = async_engine.next_request_id()
        created = int(time.time())
        response_id = f"resp-{request_id}"
        token_stream = async_engine.stream(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=request.max_output_tokens,
            eos_token_ids=eos_token_ids,
        )
        if request.stream:
            sse = _stream_responses_sse(
                token_stream=token_stream,
                detok=IncrementalDetokenizer(tokenizer),
                response_id=response_id,
                created=created,
                model=request.model,
                input_tokens=len(prompt_ids),
            )
            return StreamingResponse(sse, media_type="text/event-stream")
        return await _collect_response(
            token_stream=token_stream,
            detok=IncrementalDetokenizer(tokenizer),
            response_id=response_id,
            created=created,
            model=request.model,
            input_tokens=len(prompt_ids),
        )

    async def _serve(
        *, request_kind: str, prompt_ids: list[int], max_new_tokens: int, stream: bool, model: str
    ):
        request_id = async_engine.next_request_id()
        created = int(time.time())
        completion_id = f"cmpl-{request_id}"
        token_stream = async_engine.stream(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        if stream:
            sse = _stream_sse(
                request_kind=request_kind,
                token_stream=token_stream,
                detok=IncrementalDetokenizer(tokenizer),
                completion_id=completion_id,
                created=created,
                model=model,
            )
            return StreamingResponse(sse, media_type="text/event-stream")
        return await _collect(
            request_kind=request_kind,
            token_stream=token_stream,
            detok=IncrementalDetokenizer(tokenizer),
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
    detok: IncrementalDetokenizer,
    completion_id: str,
    created: int,
    model: str,
    prompt_tokens: int,
):
    """Drain the whole token stream and assemble a single non-streaming response."""
    text_parts: list[str] = []
    finish_reason = "length"
    completion_tokens = 0
    async for item in token_stream:
        completion_tokens += 1
        text_parts.append(detok.feed(item.token_id))
        if item.finish_reason is not None:
            finish_reason = item.finish_reason
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
    detok: IncrementalDetokenizer,
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
        text_parts.append(detok.feed(item.token_id))
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
    detok: IncrementalDetokenizer,
    response_id: str,
    created: int,
    model: str,
    input_tokens: int,
) -> AsyncIterator[str]:
    """Emit the Responses semantic SSE events, not chat chunks.

    The minimal text-generation lifecycle: ``response.created`` once, a
    ``response.output_text.delta`` per decodable text chunk (carrying ``delta``), then a
    terminal ``response.completed`` whose payload includes the assembled ``response`` object.
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
        delta_text = detok.feed(item.token_id)
        if delta_text:
            text_parts.append(delta_text)
            yield _sse_event(
                "response.output_text.delta",
                {"response_id": response_id, "delta": delta_text},
            )
    tail = detok.finalize()
    if tail:
        text_parts.append(tail)
        yield _sse_event("response.output_text.delta", {"response_id": response_id, "delta": tail})

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
    detok: IncrementalDetokenizer,
    completion_id: str,
    created: int,
    model: str,
) -> AsyncIterator[str]:
    """Emit OpenAI SSE chunks: per-token deltas, a finish chunk, then ``[DONE]``."""
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
        delta_text = detok.feed(item.token_id)
        if item.finish_reason is not None:
            finish_reason = item.finish_reason
        if delta_text:
            yield _sse(_delta_chunk(request_kind, completion_id, created, model, delta_text))

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


def _apply_chat_template(tokenizer: object, messages: list[ChatMessage]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        add_generation_prompt=True,
        tokenize=True,
    )
    if not rendered:
        raise HTTPException(400, "chat template produced zero tokens")
    return list(rendered)


def _reject_unsupported(request: ChatCompletionRequest) -> None:
    if request.tools is not None or request.functions is not None:
        raise HTTPException(422, "tools / function-calling are not supported")


def _check_sampling(request: SamplingParams, sampler: Sampler) -> None:
    """Reject anything we cannot honor, and any sampling that disagrees with the engine.

    The engine runs one fixed sampler for the whole server. We accept ``temperature`` and
    ``top_p`` only when they match it, rather than silently decoding differently from what
    the caller asked.
    """
    if request.n != 1:
        raise HTTPException(422, "'n' > 1 is not supported; this server returns a single choice")
    if request.logprobs not in (None, False, 0):
        raise HTTPException(422, "'logprobs' is not supported")
    if request.stop is not None:
        raise HTTPException(422, _STOP_UNSUPPORTED)
    _check_fixed_sampling(request.temperature, request.top_p, sampler)


def _check_fixed_sampling(temperature: float, top_p: float, sampler: Sampler) -> None:
    """The engine runs one fixed sampler; reject sampling params that disagree with it.

    We accept ``temperature``/``top_p`` only when they match the server's configured sampler,
    rather than silently decoding differently from what the caller asked for.
    """
    if not math.isclose(temperature, sampler.temperature):
        raise HTTPException(
            400,
            f"this server decodes at temperature={sampler.temperature}; "
            f"requested temperature={temperature} cannot be honored",
        )
    if not math.isclose(top_p, sampler.top_p):
        raise HTTPException(
            400,
            f"this server decodes at top_p={sampler.top_p}; "
            f"requested top_p={top_p} cannot be honored",
        )


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
    if not rendered:
        raise HTTPException(400, "chat template produced zero tokens")
    return list(rendered)


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


def _check_responses_sampling(request: ResponsesRequest, sampler: Sampler) -> None:
    if request.stop is not None:
        raise HTTPException(422, _STOP_UNSUPPORTED)
    _check_fixed_sampling(request.temperature, request.top_p, sampler)
