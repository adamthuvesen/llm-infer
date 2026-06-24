"""OpenAI-compatible request/response schemas — the wire contract, nothing more.

The request models set ``extra="forbid"`` so an unknown field (``tools``, ``functions``,
``logit_bias``, …) is a loud 422, never a silent drop. Fields this engine cannot honor
(``n > 1``, ``logprobs``, ``stop`` beyond what we support) are validated explicitly with a
clear message. We model only what we actually serve; we do not pretend to accept more.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class SamplingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = Field(default=128, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = 1
    logprobs: bool | int | None = None


class ChatCompletionRequest(SamplingParams):
    messages: list[ChatMessage] = Field(min_length=1)
    tools: object | None = None
    functions: object | None = None


class CompletionRequest(SamplingParams):
    prompt: str | list[str]


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None


class ChatCompletionDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: str | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "llm-infer"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


# --- Responses API (stateless, text-generation subset) -----------------------------------


class ResponseInputItem(BaseModel):
    """One structured input item: a role + text content, mirroring a chat message."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "developer"]
    content: str
    type: Literal["message"] = "message"


class ResponsesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | list[ResponseInputItem]
    instructions: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = Field(default=128, ge=1)
    stop: str | list[str] | None = None
    stream: bool = False
    n: int = 1
    # Stateful / hosted-tool features a from-scratch engine cannot honor — rejected explicitly.
    tools: object | None = None
    previous_response_id: str | None = None
    store: bool | None = None
    background: bool | None = None


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str


class ResponseOutputMessage(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ResponseOutputText]


class ResponseUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class Response(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: Literal["completed"] = "completed"
    output: list[ResponseOutputMessage]
    output_text: str
    usage: ResponseUsage
