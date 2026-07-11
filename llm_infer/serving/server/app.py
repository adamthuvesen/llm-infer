"""OpenAI-shaped HTTP routes over an injected engine and tokenizer — transport only."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from llm_infer.model.interface import TokenizerLike
from llm_infer.serving.sampler import SamplingParams
from llm_infer.serving.server.async_engine import AsyncInferenceEngine, TokenStreamItem
from llm_infer.serving.server.metrics import ServerMetrics
from llm_infer.serving.server.protocol import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
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
from llm_infer.serving.server.streaming import (
    drain_token_stream,
    stream_openai_sse,
    stream_responses_sse,
)

# Bounds on the OpenAI `stop` field. Beyond these we 400 rather than do unbounded buffering work
# per token; OpenAI itself caps stop at 4 sequences, which is plenty for a from-scratch engine.
_MAX_STOP_SEQUENCES = 4
_MAX_STOP_LENGTH = 256


@dataclass(frozen=True)
class _StartedGeneration:
    request_id: str
    created: int
    arrival: float
    token_stream: AsyncIterator[TokenStreamItem]
    detok: StopSequenceDetokenizer


def create_app(
    *,
    async_engine: AsyncInferenceEngine,
    tokenizer: TokenizerLike,
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
            include_stream_usage=_include_stream_usage(request),
            model=model,
            sampling=_sampling_params(request),
            stop=_stop_sequences(request.stop),
            prefix_group_id=request.llm_infer_prefix_group_id,
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
            include_stream_usage=_include_stream_usage(request),
            model=model,
            sampling=_sampling_params(request),
            stop=_stop_sequences(request.stop),
            prefix_group_id=request.llm_infer_prefix_group_id,
        )

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest):
        _reject_responses_unsupported(request)
        model = _resolve_model(request.model, model_id)
        prompt_ids = _responses_prompt_ids(tokenizer, request)
        started = _start_generation(
            async_engine=async_engine,
            tokenizer=tokenizer,
            eos_token_ids=eos_token_ids,
            prompt_ids=prompt_ids,
            max_new_tokens=request.max_output_tokens,
            sampling=_sampling_params(request),
            stop=_stop_sequences(request.stop),
            prefix_group_id=request.llm_infer_prefix_group_id,
        )
        response_id = f"resp-{started.request_id}"
        if request.stream:
            sse = stream_responses_sse(
                token_stream=started.token_stream,
                detok=started.detok,
                response_id=response_id,
                created=started.created,
                model=model,
                input_tokens=len(prompt_ids),
                metrics=metrics,
                arrival=started.arrival,
            )
            return StreamingResponse(sse, media_type="text/event-stream")
        return await _collect_response(
            token_stream=started.token_stream,
            detok=started.detok,
            response_id=response_id,
            created=started.created,
            model=model,
            input_tokens=len(prompt_ids),
            metrics=metrics,
            arrival=started.arrival,
        )

    async def _serve(
        *,
        request_kind: str,
        prompt_ids: list[int],
        max_new_tokens: int,
        stream: bool,
        include_stream_usage: bool,
        model: str,
        sampling: SamplingParams,
        stop: list[str],
        prefix_group_id: str | None,
    ):
        started = _start_generation(
            async_engine=async_engine,
            tokenizer=tokenizer,
            eos_token_ids=eos_token_ids,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            stop=stop,
            prefix_group_id=prefix_group_id,
        )
        completion_id = f"cmpl-{started.request_id}"
        if stream:
            sse = stream_openai_sse(
                request_kind=request_kind,
                token_stream=started.token_stream,
                detok=started.detok,
                completion_id=completion_id,
                created=started.created,
                model=model,
                prompt_tokens=len(prompt_ids),
                include_usage=include_stream_usage,
                metrics=metrics,
                arrival=started.arrival,
            )
            return StreamingResponse(sse, media_type="text/event-stream")
        return await _collect(
            request_kind=request_kind,
            token_stream=started.token_stream,
            detok=started.detok,
            completion_id=completion_id,
            created=started.created,
            model=model,
            prompt_tokens=len(prompt_ids),
            metrics=metrics,
            arrival=started.arrival,
        )

    return app


def _start_generation(
    *,
    async_engine: AsyncInferenceEngine,
    tokenizer: TokenizerLike,
    eos_token_ids: frozenset[int],
    prompt_ids: list[int],
    max_new_tokens: int,
    sampling: SamplingParams,
    stop: list[str],
    prefix_group_id: str | None,
) -> _StartedGeneration:
    _assert_capacity(async_engine, prompt_ids, max_new_tokens)
    arrival = time.perf_counter()
    request_id = async_engine.next_request_id()
    created = int(time.time())
    return _StartedGeneration(
        request_id=request_id,
        created=created,
        arrival=arrival,
        token_stream=async_engine.stream(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            sampling=sampling,
            prefix_group_id=prefix_group_id,
        ),
        detok=StopSequenceDetokenizer(tokenizer, stop),
    )


async def _collect(
    *,
    request_kind: str,
    token_stream: AsyncIterator[TokenStreamItem],
    detok: StopSequenceDetokenizer,
    completion_id: str,
    created: int,
    model: str,
    prompt_tokens: int,
    metrics: ServerMetrics | None = None,
    arrival: float | None = None,
):
    """Drain the token stream and assemble a single non-streaming response.

    A stop string truncates the text before the stop, sets ``finish_reason="stop"``, and aborts
    the engine request so no compute is wasted past the stop.
    """
    drained = await drain_token_stream(
        token_stream=token_stream,
        detok=detok,
        metrics=metrics,
        arrival=arrival,
    )
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=drained.token_count,
        total_tokens=prompt_tokens + drained.token_count,
    )
    if request_kind == "chat":
        return ChatCompletionResponse(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatCompletionChoice(
                    message=ChatCompletionMessage(content=drained.text),
                    finish_reason=drained.finish_reason,
                )
            ],
            usage=usage,
        )
    return CompletionResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[CompletionChoice(text=drained.text, finish_reason=drained.finish_reason)],
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
    metrics: ServerMetrics | None = None,
    arrival: float | None = None,
):
    """Drain the stream into one Responses ``response`` object (a single assistant message)."""
    drained = await drain_token_stream(
        token_stream=token_stream,
        detok=detok,
        metrics=metrics,
        arrival=arrival,
    )
    return Response(
        id=response_id,
        created_at=created,
        model=model,
        output=[
            ResponseOutputMessage(content=[ResponseOutputText(text=drained.text)]),
        ],
        output_text=drained.text,
        usage=ResponseUsage(
            input_tokens=input_tokens,
            output_tokens=drained.token_count,
            total_tokens=input_tokens + drained.token_count,
        ),
    )


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


def _apply_chat_template(tokenizer: TokenizerLike, messages: list[ChatMessage]) -> list[int]:
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


def _include_stream_usage(request: SamplingRequestBody) -> bool:
    return bool(request.stream_options and request.stream_options.include_usage)


def _responses_prompt_ids(tokenizer: TokenizerLike, request: ResponsesRequest) -> list[int]:
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
