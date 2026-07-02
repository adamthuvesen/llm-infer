"""Esme paged-KV vs full-recompute comparison, shared by the Modal harness and CPU runs.

Esme serves through real paged K/V (``PretrainBundleModel`` writes/reads pages through the
same engine prefill/decode path as Qwen). This module times two systems on one workload:

* ``llm_infer_paged`` — the engine: all requests in one paged cache, every running request
  advanced in one fused batched decode (``decode_many``) per step.
* ``full_recompute`` — per-request ``greedy_decode``, one ``logits()``
  full forward over the whole growing sequence at every step (no cache).

Both must reproduce the *same* fp32 reference — direct ``PretrainBundleModel.logits()`` greedy
decode — before any tok/s is reported. Per the repo rule (match before measuring), a system that
diverges reports no throughput, only its measured token count and wall-clock. The pieces here are
pure (no Modal, no GPU assumption) so the same comparison runs on CPU for a relative check when a
GPU bench is deferred.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from llm_infer.benchmarks.report import normalize_at_eos, total_output_tokens
from llm_infer.model.decode import greedy_decode
from llm_infer.model.interface import CausalLMBackend
from llm_infer.model.runtime import ModelRuntime
from llm_infer.serving import InferenceEngine, Request


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


def _matches_reference(
    outputs: dict[str, list[int]],
    reference: dict[str, list[int]],
    eos_token_ids: frozenset[int],
) -> bool:
    return all(
        normalize_at_eos(outputs[request_id], eos_token_ids)
        == normalize_at_eos(reference[request_id], eos_token_ids)
        for request_id in reference
    )


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


def compare_paged_vs_recompute(
    runtime: ModelRuntime,
    requests: list[EsmeBenchRequest],
    *,
    max_new_tokens: int,
    block_size: int,
    num_blocks: int,
    warmup: int,
    iters: int,
    device: str = "cpu",
) -> list[SystemTiming]:
    """Time the paged engine and the full-recompute baseline, each gated on the reference.

    Both systems are checked against ``reference_outputs`` before timing is reported; a diverging
    system keeps its wall-clock and token count but reports no tok/s. Returns one
    :class:`SystemTiming` per system (paged first, then full recompute).
    """
    eos = runtime.eos_token_ids
    reference = reference_outputs(
        runtime.model, requests, max_new_tokens=max_new_tokens, eos_token_ids=eos
    )
    sync = device == "cuda"

    def paged_once() -> dict[str, list[int]]:
        engine = InferenceEngine(
            runtime.model,
            block_size=block_size,
            num_blocks=num_blocks,
            device=device,
            capabilities=runtime.capabilities,
        )
        for req in requests:
            engine.add_request(Request(req.request_id, list(req.prompt_ids), max_new_tokens, eos))
        return engine.run()

    def recompute_once() -> dict[str, list[int]]:
        return {
            req.request_id: greedy_decode(
                runtime.model,
                list(req.prompt_ids),
                max_new_tokens=max_new_tokens,
                eos_token_ids=set(eos),
            )
            for req in requests
        }

    timings: list[SystemTiming] = []
    for system, mode, decode_once in (
        ("llm_infer_paged", "paged KV + batched decode", paged_once),
        ("full_recompute", "per-request full recompute", recompute_once),
    ):
        median_s, outputs = _time(decode_once, warmup=warmup, iters=iters, sync=sync)
        timings.append(
            SystemTiming(
                system=system,
                mode=mode,
                matches_reference=_matches_reference(outputs, reference, eos),
                median_seconds=median_s,
                total_output_tokens=total_output_tokens(outputs, eos),
                outputs=outputs,
            )
        )
    return timings
