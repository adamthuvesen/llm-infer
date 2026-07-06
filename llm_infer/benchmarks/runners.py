"""The llm-infer benchmark runner and warmup/measurement timing wrapper.

The Esme benchmark builds its HF and vLLM closures in
``llm_infer.benchmarks.esme_three_way``. This module keeps the shared timing wrapper and
the in-engine runner used by benchmarks and CPU smoke tests.

Timing: ``warmup`` un-measured iterations (CUDA graphs / allocator / autotune settle),
then ``iters`` measured iterations with a CUDA sync at each boundary; greedy decoding is
deterministic, so every iteration produces identical tokens and only wall-clock varies.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from llm_infer.benchmarks.workload import Workload
from llm_infer.model.interface import CausalLMBackend
from llm_infer.profiling import TimingProfiler
from llm_infer.serving import GREEDY, InferenceEngine, Request

BLOCK_SIZE = 128


@dataclass
class RunResult:
    """One system's benchmark output: tokens generated and the measured per-iter seconds."""

    system: str
    outputs: dict[str, list[int]]
    per_iter_seconds: list[float]
    config: dict[str, object] = field(default_factory=dict)
    profiles: list[dict[str, object]] = field(default_factory=list)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_system(
    system: str,
    decode_once: Callable[[], dict[str, list[int]]],
    *,
    warmup: int,
    iters: int,
    config: dict[str, object] | None = None,
) -> RunResult:
    """Run ``decode_once`` ``warmup`` times un-measured, then ``iters`` times measured.

    Returns the tokens from the last iteration (identical across iterations under greedy)
    and the measured wall-clock of each timed iteration.
    """
    for _ in range(warmup):
        decode_once()
        _sync()

    per_iter: list[float] = []
    outputs: dict[str, list[int]] = {}
    for _ in range(iters):
        _sync()
        start = time.perf_counter()
        outputs = decode_once()
        _sync()
        per_iter.append(time.perf_counter() - start)

    return RunResult(system=system, outputs=outputs, per_iter_seconds=per_iter, config=config or {})


def run_llm_infer(
    model: CausalLMBackend,
    workload: Workload,
    *,
    num_blocks: int,
    warmup: int,
    iters: int,
    device: str = "cuda",
    collect_profile: bool = False,
    enable_prefix_caching: bool = False,
) -> RunResult:
    """This engine, all requests in one paged cache under the batching loop."""
    profiles: list[dict[str, object]] = []
    baseline_prefill_tokens = sum(workload.prompt_lengths)
    shared_prefill_tokens = sum(
        len(prompt) for prompts in _prompt_groups(workload).values() for prompt in prompts
    )

    def decode_once(profiler: TimingProfiler | None = None) -> dict[str, list[int]]:
        engine = InferenceEngine(
            model,
            block_size=BLOCK_SIZE,
            num_blocks=num_blocks,
            device=device,
            profiler=profiler,
        )
        for req in workload.requests:
            engine.add_request(
                Request(
                    req.request_id,
                    list(req.prompt_ids),
                    workload.max_new_tokens,
                    workload.eos_token_ids,
                    prefix_group_id=req.case_id if enable_prefix_caching else None,
                    sampling=GREEDY,
                )
            )
        outputs = engine.run()
        return outputs

    result = time_system(
        "llm_infer",
        lambda: decode_once(),
        warmup=warmup,
        iters=iters,
        config={
            "backend": type(model.backend).__name__,
            "dtype": str(model.dtype),
            "block_size": BLOCK_SIZE,
            "num_blocks": num_blocks,
            # All running requests advance in one fused batched decode (decode_many) per step.
            "batched_forward": True,
            "prefix_caching": enable_prefix_caching,
            "prefill_token_ops": {
                "per_sibling_baseline": baseline_prefill_tokens,
                "shared_prefix": shared_prefill_tokens
                if enable_prefix_caching
                else baseline_prefill_tokens,
                "reduction": baseline_prefill_tokens
                - (shared_prefill_tokens if enable_prefix_caching else baseline_prefill_tokens),
            },
            "sampling": {"mode": "greedy", "temperature": 0.0},
            "profile": collect_profile,
        },
    )
    if collect_profile:
        _sync()
        profiler = TimingProfiler(device)
        decode_once(profiler)
        _sync()
        profiles.append(profiler.summary().as_dict())
    result.profiles = profiles
    return result


def _prompt_groups(workload: Workload) -> dict[str, set[tuple[int, ...]]]:
    """Group explicitly by workload case id; mismatched ids/prompts stay visible."""
    groups: dict[str, set[tuple[int, ...]]] = {}
    for req in workload.requests:
        groups.setdefault(req.case_id, set()).add(req.prompt_ids)
    return groups
