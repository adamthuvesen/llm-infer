"""The four system runners and the warmup/measurement timing wrapper.

Each runner decodes the same :class:`~llm_infer.benchmarks.workload.Workload` greedily and
returns the generated continuation ids plus per-iteration wall-clock. The systems:

* ``hf_sequential`` — **the naive baseline, defined out loud**: HuggingFace
  ``model.generate()`` called once per request, one at a time. HF's own cached generate
  (default SDPA attention), but with **no cross-request batching** — the realistic thing a
  person writes first. Cross-request batching is exactly the systems contribution
  llm-infer and vLLM add, so this is the honest floor, not a strawman (it is not the slow
  full-recompute path; it uses HF's optimized generate).
* ``hf_batched`` — a stronger HF reference: a single left-padded batched
  ``model.generate()``. Included so "llm-infer beats naive HF" cannot be read as beating a
  deliberately weak baseline.
* ``llm_infer`` — this engine on the flash-attn backend (bf16, CUDA), all requests admitted
  to one paged cache and driven by the continuous-batching loop. Every running request
  advances in one **fused batched decode** (``decode_many``) per step — one matmul/kernel
  call over the whole running batch, not one per request — so the throughput win over naive
  per-request HF generate is the batched forward plus the fused paged kernel.
* ``vllm`` — vLLM offline ``LLM.generate`` with prefix caching off and flags pinned. The
  ceiling, never the thing we beat.

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
from llm_infer.model.qwen import QwenModel
from llm_infer.profiling import TimingProfiler
from llm_infer.serving import GREEDY, InferenceEngine, Request, SamplingParams

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


def _sampling_config(sampling: object) -> dict[str, object]:
    """Render a system's decoding config for the pinned record (greedy vs sampled)."""
    if sampling is None:
        return {"mode": "greedy", "temperature": 0.0}
    return {
        "mode": "sampling",
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "seed": sampling.seed,
    }


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
    model: QwenModel,
    workload: Workload,
    *,
    num_blocks: int,
    warmup: int,
    iters: int,
    device: str = "cuda",
    collect_profile: bool = False,
    enable_prefix_caching: bool = False,
) -> RunResult:
    """This engine, flash backend, all requests in one paged cache under the batching loop.

    Greedy by default; under ``workload.sampling`` each request carries its OWN SamplingParams
    with a seed derived from the pinned base seed plus the request's index, so the ``G``
    completions of a prompt are independent draws (a real GRPO group needs diverse rollouts, not
    ``G`` identical ones), while every iteration still reproduces the same tokens (the same
    derived seeds) — the median wall-clock measures equal work, not RNG drift. A shared seed
    would seed every request's generator identically and collapse the group to one completion.
    """
    sampling = workload.sampling
    profiles: list[dict[str, object]] = []
    baseline_prefill_tokens = sum(workload.prompt_lengths)
    shared_prefill_tokens = sum(
        len(prompt) for prompts in _prompt_groups(workload).values() for prompt in prompts
    )

    def request_sampling(index: int) -> SamplingParams:
        """This request's sampling: greedy when the workload is greedy, else its own seed.

        The seed is ``base_seed + index`` so each request in the expanded ``prompts × G`` list
        draws independently yet reproducibly. ``request.generator()`` seeds from the request's
        own params, so the per-request seed (not a shared engine default) is what actually drives
        each draw — that is the bug this closes.
        """
        if sampling is None:
            return GREEDY
        return SamplingParams(
            temperature=sampling.temperature, top_p=sampling.top_p, seed=sampling.seed + index
        )

    def decode_once(profiler: TimingProfiler | None = None) -> dict[str, list[int]]:
        engine = InferenceEngine(
            model,
            block_size=BLOCK_SIZE,
            num_blocks=num_blocks,
            device=device,
            profiler=profiler,
        )
        for index, req in enumerate(workload.requests):
            engine.add_request(
                Request(
                    req.request_id,
                    list(req.prompt_ids),
                    workload.max_new_tokens,
                    workload.eos_token_ids,
                    prefix_group_id=req.case_id if enable_prefix_caching else None,
                    sampling=request_sampling(index),
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
            "sampling": _sampling_config(sampling),
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


def run_hf_sequential(
    hf_model: object,
    workload: Workload,
    *,
    warmup: int,
    iters: int,
    device: str = "cuda",
) -> RunResult:
    """Naive baseline (the floor): HF ``generate()`` once per request, sequentially.

    Greedy by default; under ``workload.sampling`` it samples with the pinned temperature/
    top-p and re-seeds (``set_seed``) at the start of every ``decode_once`` so each measured
    iteration reproduces identical tokens.
    """
    eos = sorted(workload.eos_token_ids)
    sampling = workload.sampling
    gen_kwargs: dict[str, object] = {
        "max_new_tokens": workload.max_new_tokens,
        "num_beams": 1,
        "eos_token_id": eos,
        "pad_token_id": eos[0],
    }
    if sampling is None:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(do_sample=True, temperature=sampling.temperature, top_p=sampling.top_p)

    def decode_once() -> dict[str, list[int]]:
        if sampling is not None:
            from transformers import set_seed

            set_seed(sampling.seed)
        outputs: dict[str, list[int]] = {}
        for req in workload.requests:
            input_ids = torch.tensor([req.prompt_ids], device=device)
            gen = hf_model.generate(input_ids, **gen_kwargs)
            outputs[req.request_id] = gen[0, input_ids.shape[1] :].tolist()
        return outputs

    return time_system(
        "hf_sequential",
        decode_once,
        warmup=warmup,
        iters=iters,
        config={
            "method": "per-request model.generate()",
            "attn_implementation": getattr(hf_model.config, "_attn_implementation", "unknown"),
            "dtype": str(next(hf_model.parameters()).dtype),
            "sampling": _sampling_config(sampling),
        },
    )


def run_hf_batched(
    hf_model: object,
    workload: Workload,
    *,
    warmup: int,
    iters: int,
    device: str = "cuda",
) -> RunResult:
    """Stronger HF reference: one left-padded batched ``generate()`` over all requests."""
    eos = sorted(workload.eos_token_ids)
    pad_id = eos[0]
    max_len = max(workload.prompt_lengths)
    # Left-pad so every prompt's last real token sits at the same column — generation
    # continues from there for every row, and an attention mask hides the pad.
    input_rows, mask_rows = [], []
    for req in workload.requests:
        pad = max_len - len(req.prompt_ids)
        input_rows.append([pad_id] * pad + list(req.prompt_ids))
        mask_rows.append([0] * pad + [1] * len(req.prompt_ids))
    input_ids = torch.tensor(input_rows, device=device)
    attention_mask = torch.tensor(mask_rows, device=device)
    ids_by_index = [req.request_id for req in workload.requests]

    def decode_once() -> dict[str, list[int]]:
        gen = hf_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=workload.max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos,
            pad_token_id=pad_id,
        )
        continuations = gen[:, input_ids.shape[1] :]
        return {ids_by_index[i]: continuations[i].tolist() for i in range(len(ids_by_index))}

    return time_system(
        "hf_batched",
        decode_once,
        warmup=warmup,
        iters=iters,
        config={
            "method": "single left-padded batched model.generate()",
            "attn_implementation": getattr(hf_model.config, "_attn_implementation", "unknown"),
            "dtype": str(next(hf_model.parameters()).dtype),
        },
    )


def run_vllm(
    workload: Workload,
    *,
    warmup: int,
    iters: int,
    gpu_memory_utilization: float = 0.90,
    max_num_seqs: int = 256,
) -> RunResult:
    """vLLM offline generate, bf16, prefix caching off, flags pinned and recorded.

    Greedy on the base weights for the Phase D benchmark; the rollout passes the merged
    grpo-s0 path as ``workload.model_id`` plus ``workload.sampling``. Either way vLLM runs
    bf16 — the model's native dtype. The ``vllm`` import is local so the rest of the benchmark
    loads on the flash image (which has no vLLM); this runner only runs on the vLLM image.
    """
    import vllm
    from vllm import LLM, SamplingParams

    sampling = workload.sampling
    max_model_len = max(workload.prompt_lengths) + workload.max_new_tokens
    llm = LLM(
        model=workload.model_id,
        revision=workload.model_revision,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=False,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
    )
    # Greedy (Phase D) → temperature 0. Rollout → the pinned temperature/top-p, seeded for
    # a reproducible per-iteration token set. n=1 because the G=4 replication is already in
    # the workload (32 independent completions), so every system decodes identical request set.
    if sampling is None:
        params = SamplingParams(
            temperature=0.0,
            max_tokens=workload.max_new_tokens,
            n=1,
            stop_token_ids=sorted(workload.eos_token_ids),
            ignore_eos=False,
        )
    else:
        params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            seed=sampling.seed,
            max_tokens=workload.max_new_tokens,
            n=1,
            stop_token_ids=sorted(workload.eos_token_ids),
            ignore_eos=False,
        )
    prompts = [{"prompt_token_ids": list(req.prompt_ids)} for req in workload.requests]
    ids_by_index = [req.request_id for req in workload.requests]

    def decode_once() -> dict[str, list[int]]:
        results = llm.generate(prompts, params, use_tqdm=False)
        # Coerce to plain Python ints: vLLM token_ids can be tensor/array scalars, which pickle
        # with a torch ref and fail to deserialize in the (torch-less) local `modal run` env.
        return {
            ids_by_index[i]: [int(t) for t in results[i].outputs[0].token_ids]
            for i in range(len(results))
        }

    result = time_system(
        "vllm",
        decode_once,
        warmup=warmup,
        iters=iters,
        config={
            "version": vllm.__version__,
            "enable_prefix_caching": False,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            # Native sampler (VLLM_USE_FLASHINFER_SAMPLER=0, image env) — flashinfer's sampler
            # JIT-needs nvcc; under temperature/top-p vLLM samples on this native path.
            "sampler": "native-torch",
            "attention_backend": "FLASH_ATTN (vLLM auto-selected, precompiled)",
            "sampling": _sampling_config(sampling),
            "model": workload.model_id,
        },
    )
    return result
