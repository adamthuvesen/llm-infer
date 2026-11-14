"""Shared Esme benchmark workloads, timing, and reference-output helpers."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from llm_infer.model.decode import greedy_decode
from llm_infer.model.interface import CausalLMBackend, TokenizerLike


@dataclass(frozen=True)
class EsmeBenchRequest:
    request_id: str
    prompt: str
    prompt_ids: tuple[int, ...]


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
    tokenizer: TokenizerLike, num_requests: int, prompts: tuple[str, ...] = DEFAULT_PROMPTS
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


class ReferenceOutputCache:
    """Greedy outputs cached for one oracle, EOS policy, and token budget."""

    def __init__(
        self,
        model: CausalLMBackend,
        *,
        max_new_tokens: int,
        eos_token_ids: frozenset[int],
    ) -> None:
        self._model = model
        self._max_new_tokens = max_new_tokens
        self._eos_token_ids = eos_token_ids
        self._by_prompt: dict[tuple[int, ...], tuple[int, ...]] = {}

    def __call__(self, requests: list[EsmeBenchRequest]) -> dict[str, list[int]]:
        for request in requests:
            if request.prompt_ids not in self._by_prompt:
                self._by_prompt[request.prompt_ids] = tuple(
                    greedy_decode(
                        self._model,
                        list(request.prompt_ids),
                        max_new_tokens=self._max_new_tokens,
                        eos_token_ids=set(self._eos_token_ids),
                    )
                )
        return {
            request.request_id: list(self._by_prompt[request.prompt_ids]) for request in requests
        }


def reference_outputs(
    model: CausalLMBackend,
    requests: list[EsmeBenchRequest],
    *,
    max_new_tokens: int,
    eos_token_ids: frozenset[int],
) -> dict[str, list[int]]:
    """Compute each unique prompt once against the full-recompute greedy oracle."""
    return ReferenceOutputCache(model, max_new_tokens=max_new_tokens, eos_token_ids=eos_token_ids)(
        requests
    )


def measure_repeated(
    decode_once: Callable[[], dict[str, list[int]]],
    *,
    warmup: int,
    iters: int,
    synchronize: Callable[[], None],
) -> tuple[list[float], dict[str, list[int]]]:
    """Keep every duration, synchronizing before and after each measured decode."""
    if warmup < 0 or iters < 1:
        raise ValueError(f"warmup must be >= 0 and iters >= 1; got {warmup=} and {iters=}")
    for _ in range(warmup):
        decode_once()
        synchronize()
    per_iter: list[float] = []
    outputs: dict[str, list[int]] = {}
    for _ in range(iters):
        synchronize()
        start = time.perf_counter()
        outputs = decode_once()
        synchronize()
        per_iter.append(time.perf_counter() - start)
    return per_iter, outputs


def time_call(
    decode_once: Callable[[], dict[str, list[int]]], *, warmup: int, iters: int, sync: bool
) -> tuple[float, dict[str, list[int]]]:
    """Run warmups, then return the median measured duration and final outputs."""

    def maybe_sync() -> None:
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()

    per_iter, outputs = measure_repeated(
        decode_once, warmup=warmup, iters=iters, synchronize=maybe_sync
    )
    return statistics.median(per_iter), outputs
