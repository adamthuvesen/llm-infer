"""Stop-aware token draining and SSE helpers for the HTTP server."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

from llm_infer.serving.server.async_engine import TokenStreamItem
from llm_infer.serving.server.metrics import ServerMetrics
from llm_infer.serving.server.protocol import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionDelta,
    CompletionChoice,
    CompletionChunk,
    Response,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
    Usage,
)
from llm_infer.serving.server.stop import StopSequenceDetokenizer


@dataclass(frozen=True)
class DrainResult:
    text: str
    token_count: int
    finish_reason: str


@dataclass(frozen=True)
class TextStreamItem:
    text: str
    token_count: int
    finish_reason: str | None = None
    terminal: bool = False


async def drain_token_stream(
    *,
    token_stream: AsyncIterator[TokenStreamItem],
    detok: StopSequenceDetokenizer,
    metrics: ServerMetrics | None = None,
    arrival: float | None = None,
) -> DrainResult:
    """Drain one engine stream into text, preserving stop truncation semantics."""
    text_parts: list[str] = []
    finish_reason = "length"
    token_count = 0
    async for item in iter_text_stream(
        token_stream=token_stream,
        detok=detok,
        metrics=metrics,
        arrival=arrival,
    ):
        token_count = item.token_count
        if item.finish_reason is not None:
            finish_reason = item.finish_reason
        if item.terminal:
            break
        text_parts.append(item.text)
    return DrainResult(
        text="".join(text_parts),
        token_count=token_count,
        finish_reason=finish_reason,
    )


async def iter_text_stream(
    *,
    token_stream: AsyncIterator[TokenStreamItem],
    detok: StopSequenceDetokenizer,
    metrics: ServerMetrics | None = None,
    arrival: float | None = None,
) -> AsyncIterator[TextStreamItem]:
    """Yield text deltas plus one terminal marker for a stop-aware stream."""
    token_count = 0
    finish_reason: str | None = None
    async for item in token_stream:
        token_count += 1
        fed = detok.feed(item.token_id)
        if item.finish_reason is not None:
            finish_reason = item.finish_reason
        if fed.text:
            yield TextStreamItem(fed.text, token_count, finish_reason)
        if fed.stopped:
            await finish_on_stop(token_stream, metrics, arrival)
            yield TextStreamItem("", token_count, "stop", terminal=True)
            return
    else:
        tail = detok.finalize()
        if tail:
            yield TextStreamItem(tail, token_count, finish_reason)
    yield TextStreamItem("", token_count, finish_reason, terminal=True)


async def stream_responses_sse(
    *,
    token_stream: AsyncIterator[TokenStreamItem],
    detok: StopSequenceDetokenizer,
    response_id: str,
    created: int,
    model: str,
    input_tokens: int,
    metrics: ServerMetrics | None = None,
    arrival: float | None = None,
) -> AsyncIterator[str]:
    """Emit the Responses semantic SSE events, not chat chunks."""
    created_response = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "model": model,
        "status": "in_progress",
    }
    yield sse_event("response.created", {"response": created_response})

    text_parts: list[str] = []
    output_tokens = 0
    async for item in iter_text_stream(
        token_stream=token_stream,
        detok=detok,
        metrics=metrics,
        arrival=arrival,
    ):
        output_tokens = item.token_count
        if item.terminal:
            break
        text_parts.append(item.text)
        yield sse_event(
            "response.output_text.delta",
            {"response_id": response_id, "delta": item.text},
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
    yield sse_event("response.completed", {"response": completed.model_dump()})


async def stream_openai_sse(
    *,
    request_kind: str,
    token_stream: AsyncIterator[TokenStreamItem],
    detok: StopSequenceDetokenizer,
    completion_id: str,
    created: int,
    model: str,
    prompt_tokens: int,
    include_usage: bool = False,
    metrics: ServerMetrics | None = None,
    arrival: float | None = None,
) -> AsyncIterator[str]:
    """Emit OpenAI SSE chunks: per-token deltas, a finish chunk, then ``[DONE]``."""
    if request_kind == "chat":
        first = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionDelta(role="assistant"))],
        )
        yield sse(first)

    finish_reason: str | None = None
    output_tokens = 0
    async for item in iter_text_stream(
        token_stream=token_stream,
        detok=detok,
        metrics=metrics,
        arrival=arrival,
    ):
        output_tokens = item.token_count
        if item.finish_reason is not None:
            finish_reason = item.finish_reason
        if item.terminal:
            break
        yield sse(delta_chunk(request_kind, completion_id, created, model, item.text))

    yield sse(finish_chunk(request_kind, completion_id, created, model, finish_reason or "length"))
    if include_usage:
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=output_tokens,
            total_tokens=prompt_tokens + output_tokens,
        )
        yield sse(usage_chunk(request_kind, completion_id, created, model, usage))
    yield "data: [DONE]\n\n"


def sse_event(event: str, payload: dict) -> str:
    """A named SSE event (``event:``/``data:`` pair) for semantic streams."""
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def delta_chunk(
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


def finish_chunk(
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


def usage_chunk(
    request_kind: str, completion_id: str, created: int, model: str, usage: Usage
) -> ChatCompletionChunk | CompletionChunk:
    if request_kind == "chat":
        return ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[],
            usage=usage,
        )
    return CompletionChunk(
        id=completion_id,
        created=created,
        model=model,
        choices=[],
        usage=usage,
    )


def sse(chunk: ChatCompletionChunk | CompletionChunk) -> str:
    # exclude_none keeps role-only and finish-only chunks faithful to OpenAI's stream shape.
    return f"data: {json.dumps(chunk.model_dump(exclude_none=True))}\n\n"


async def finish_on_stop(
    token_stream: AsyncIterator[TokenStreamItem],
    metrics: ServerMetrics | None = None,
    arrival: float | None = None,
) -> None:
    """Halt generation on a stop hit, then record it as a completed request."""
    await token_stream.aclose()
    if metrics is not None:
        metrics.requests_completed_total.inc(finish_reason="stop")
        if arrival is not None:
            metrics.request_latency_seconds.observe(time.perf_counter() - arrival)
