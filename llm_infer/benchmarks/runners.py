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
  to one paged cache and driven by the continuous-batching loop. NB: the v1 loop advances
  each running request with its *own* forward inside a step — it does not yet fuse the batch
  into one matmul, so its throughput edge over naive HF comes from the fused kernel + paged
  cache + a tight loop, not from batched matmuls. The table reports that honestly.
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
from llm_infer.serving import InferenceEngine, Request

BLOCK_SIZE = 128


@dataclass
class RunResult:
    """One system's benchmark output: tokens generated and the measured per-iter seconds."""

    system: str
    outputs: dict[str, list[int]]
    per_iter_seconds: list[float]
    config: dict[str, object] = field(default_factory=dict)


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
    model: QwenModel,
    workload: Workload,
    *,
    num_blocks: int,
    warmup: int,
    iters: int,
    device: str = "cuda",
) -> RunResult:
    """This engine, flash backend, all requests in one paged cache under the batching loop."""

    def decode_once() -> dict[str, list[int]]:
        engine = InferenceEngine(
            model, block_size=BLOCK_SIZE, num_blocks=num_blocks, device=device
        )
        for req in workload.requests:
            engine.add_request(
                Request(
                    req.request_id,
                    list(req.prompt_ids),
                    workload.max_new_tokens,
                    workload.eos_token_ids,
                )
            )
        return engine.run()

    return time_system(
        "llm_infer",
        decode_once,
        warmup=warmup,
        iters=iters,
        config={
            "backend": type(model.backend).__name__,
            "dtype": str(model.dtype),
            "block_size": BLOCK_SIZE,
            "num_blocks": num_blocks,
            # v1: per-request forward inside a step, not a fused batch matmul (see docstring).
            "batched_forward": False,
        },
    )


def run_hf_sequential(
    hf_model: object,
    workload: Workload,
    *,
    warmup: int,
    iters: int,
    device: str = "cuda",
) -> RunResult:
    """Naive baseline: HF ``generate()`` once per request, sequentially."""
    eos = sorted(workload.eos_token_ids)

    def decode_once() -> dict[str, list[int]]:
        outputs: dict[str, list[int]] = {}
        for req in workload.requests:
            input_ids = torch.tensor([req.prompt_ids], device=device)
            gen = hf_model.generate(
                input_ids,
                max_new_tokens=workload.max_new_tokens,
                do_sample=False,
                num_beams=1,
                eos_token_id=eos,
                pad_token_id=eos[0],
            )
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
    """vLLM offline generate, greedy, prefix caching off, flags pinned and recorded.

    The ``vllm`` import is local so the rest of the benchmark loads on the flash image
    (which has no vLLM); this runner only ever executes on the dedicated vLLM image.
    """
    import vllm
    from vllm import LLM, SamplingParams

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
    params = SamplingParams(
        temperature=0.0,
        max_tokens=workload.max_new_tokens,
        n=1,
        stop_token_ids=sorted(workload.eos_token_ids),
        ignore_eos=False,
    )
    prompts = [{"prompt_token_ids": list(req.prompt_ids)} for req in workload.requests]
    ids_by_index = [req.request_id for req in workload.requests]

    def decode_once() -> dict[str, list[int]]:
        results = llm.generate(prompts, params, use_tqdm=False)
        # vLLM may reorder; map each result back by its input index.
        return {ids_by_index[i]: list(results[i].outputs[0].token_ids) for i in range(len(results))}

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
            # JIT-needs nvcc; greedy decoding (argmax) is sampler-backend-independent anyway.
            "sampler": "native-torch",
            "attention_backend": "FLASH_ATTN (vLLM auto-selected, precompiled)",
        },
    )
    return result
