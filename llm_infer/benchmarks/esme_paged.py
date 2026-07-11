"""Shared Esme benchmark workloads, timing, and reference-output helpers."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from llm_infer.model.decode import greedy_decode
from llm_infer.model.interface import CausalLMBackend


@dataclass(frozen=True)
class EsmeBenchRequest:
    request_id: str
    prompt: str
    prompt_ids: tuple[int, ...]


@dataclass(frozen=True)
class SystemTiming:
    """One system's measured result: tokens, wall-clock, and whether it matched the reference."""

    system: str
    mode: str
    matches_reference: bool
    median_seconds: float
    total_output_tokens: int
    outputs: dict[str, list[int]]

    @property
    def tokens_per_second(self) -> float | None:
        """tok/s only when the system matched the reference and produced positive wall-clock."""
        if not self.matches_reference or self.median_seconds <= 0:
            return None
        return self.total_output_tokens / self.median_seconds

    @property
    def raw_tokens_per_second(self) -> float | None:
        """Measured tok/s regardless of whether the row is fit for a public claim."""
        if self.median_seconds <= 0:
            return None
        return self.total_output_tokens / self.median_seconds


DEFAULT_PROMPTS = (
    "Write a tiny Python function that doubles an integer.",
    "Explain KV caching in one short sentence.",
    "Give one SQL query that counts rows in a table named events.",
    "Name two practical checks before trusting a benchmark.",
)

# The headline serving shape cycles a wider pool of chat-length prompts so a 64-request
# batch is not sixteen copies of four questions. Lengths vary from one-liners to
# paragraph-sized asks — the mix a small chat service actually sees.
HEADLINE_PROMPTS = (
    *DEFAULT_PROMPTS,
    "Summarize what a paged KV cache does and why serving systems use one.",
    "I have a CSV with columns user_id, plan, and mrr. Walk me through finding the "
    "plan with the highest total mrr using pandas.",
    "Draft a short, friendly message telling my team the deploy is delayed until "
    "tomorrow morning because the migration needs another review pass.",
    "What is the difference between throughput and latency in a serving benchmark, "
    "and when should I care about each one?",
)


def build_requests(
    tokenizer: object, num_requests: int, prompts: tuple[str, ...] = DEFAULT_PROMPTS
) -> list[EsmeBenchRequest]:
    """Render ``num_requests`` chat prompts to token ids, cycling the prompt pool."""
    if num_requests < 1:
        raise ValueError(f"num_requests must be >= 1; got {num_requests}")
    requests: list[EsmeBenchRequest] = []
    for index in range(num_requests):
        content = prompts[index % len(prompts)]
        tokenized = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if not isinstance(tokenized, list) or not tokenized:
            raise ValueError(f"Esme tokenizer returned invalid prompt ids for request {index}")
        requests.append(
            EsmeBenchRequest(
                request_id=f"esme-{index:03d}",
                prompt=content,
                prompt_ids=tuple(int(token_id) for token_id in tokenized),
            )
        )
    return requests


def requests_at_context_length(
    requests: list[EsmeBenchRequest], context_length: int
) -> list[EsmeBenchRequest]:
    """Repeat each valid tokenized prompt to an exact synthetic cached-context length."""
    if context_length < 1:
        raise ValueError(f"context_length must be >= 1; got {context_length}")
    resized: list[EsmeBenchRequest] = []
    for request in requests:
        prompt_ids = request.prompt_ids
        repeats = (context_length + len(prompt_ids) - 1) // len(prompt_ids)
        exact_ids = (prompt_ids * repeats)[:context_length]
        resized.append(
            EsmeBenchRequest(
                request_id=request.request_id,
                prompt=f"{request.prompt} [synthetic context: {context_length} tokens]",
                prompt_ids=exact_ids,
            )
        )
    return resized


def single_request_prompt_coverage(
    tokenizer: object,
    context_lengths: tuple[int, ...],
    prompts: tuple[str, ...] = HEADLINE_PROMPTS,
) -> list[tuple[int, EsmeBenchRequest]]:
    """Build every prompt as its own request at every synthetic context length."""
    if not context_lengths:
        raise ValueError("context_lengths must not be empty")
    base_requests = build_requests(tokenizer, len(prompts), prompts)
    return [
        (context_length, request)
        for context_length in context_lengths
        for request in requests_at_context_length(base_requests, context_length)
    ]


def reference_outputs(
    model: CausalLMBackend,
    requests: list[EsmeBenchRequest],
    *,
    max_new_tokens: int,
    eos_token_ids: frozenset[int],
) -> dict[str, list[int]]:
    """fp32-class reference: direct ``PretrainBundleModel.logits()`` greedy decode per request.

    Benchmark workloads cycle a small prompt pool, so large batches repeat prompts; greedy
    oracle outputs are identical per prompt and are computed once per unique prompt.
    """
    by_prompt: dict[tuple[int, ...], list[int]] = {}
    for req in requests:
        if req.prompt_ids not in by_prompt:
            by_prompt[req.prompt_ids] = greedy_decode(
                model,
                list(req.prompt_ids),
                max_new_tokens=max_new_tokens,
                eos_token_ids=set(eos_token_ids),
            )
    return {req.request_id: list(by_prompt[req.prompt_ids]) for req in requests}


def _time(
    decode_once: Callable[[], dict[str, list[int]]], *, warmup: int, iters: int, sync: bool
) -> tuple[float, dict[str, list[int]]]:
    """Run ``decode_once`` ``warmup`` times un-measured, then ``iters`` measured; median wall."""

    def maybe_sync() -> None:
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(warmup):
        decode_once()
        maybe_sync()
    per_iter: list[float] = []
    outputs: dict[str, list[int]] = {}
    for _ in range(iters):
        maybe_sync()
        start = time.perf_counter()
        outputs = decode_once()
        maybe_sync()
        per_iter.append(time.perf_counter() - start)
    return statistics.median(per_iter), outputs
