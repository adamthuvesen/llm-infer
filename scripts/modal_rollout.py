"""Modal A100 rollout-timing comparison: llm-infer vs vLLM as llm-rlvr-sql GRPO rollout backends.

Phase E's differentiator. Extends the Phase D harness (``llm_infer/benchmarks/`` runners +
report, the same pinned-config Modal pattern) to the *sampled* rollout workload: one frozen
llm-rlvr-sql GRPO rollout batch (anchor ``grpo-s0``) — 8 Spider-dev prompts × G=4 = 32
completions, ``max_completion_length=1024``, ``temperature=1.0``, ``top_p=1.0``, pinned seed —
served from the merged grpo-s0 bf16 weights (``scripts/merge_adapter.py`` wrote them to the
``llm-infer-merged`` volume).

Three rows, the same three systems as Phase D but timed under sampling:

* ``vllm`` — llm-rlvr-sql's *current* rollout backend, the ceiling (never the thing we beat);
* ``llm_infer`` — this engine, flash backend, fused batched decode, seeded sampler;
* ``hf_sequential`` — the naive floor (per-request ``generate``, sampling).

Unlike Phase D there is **no** cross-system token-equivalence gate: under sampling the engines
use different RNG, so identical tokens are neither expected nor honest to require. The metrics
(all free derivations of one run) are rollout-batch wall-clock · rollout tok/s · $/1k rollouts.

    modal run scripts/modal_rollout.py --command smoke    # cheap wiring + bf16 sanity (2 comp)
    modal run scripts/modal_rollout.py --command rollout  # the frozen batch (32 comp, 1024 tok)
    modal run scripts/modal_rollout.py --command rollout --profile  # diagnostic extra run
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/llm-infer"
HF_CACHE = "/hf-cache"
MERGED_MOUNT = "/merged"
MERGED_MODEL_PATH = f"{MERGED_MOUNT}/grpo-s0"  # written by scripts/merge_adapter.py
# Same devel CUDA image as modal_oracle / modal_benchmark, so the flash-attn layer is cached.
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"

app = modal.App("llm-infer-rollout")

_IGNORE = [
    "**/.git",
    "**/.venv",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
    "bench-results/**",
]

# Flash image: byte-identical prefix to modal_benchmark's so the expensive flash-attn build is
# a cache hit; only the source-copy tail relinks.
flash_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install("git", "build-essential")
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install("torch>=2.2", "transformers>=4.43", "numpy>=1.26", "pytest>=8.0")
    .pip_install("wheel", "packaging", "setuptools", "ninja")
    .pip_install("flash-attn>=2.5", extra_options="--no-build-isolation")
    .env({"HF_HOME": HF_CACHE})
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
    .workdir(REMOTE_ROOT)
    .run_commands("pip install --no-deps -e .")
)

# vLLM image: identical to modal_benchmark's (vLLM ships its own torch/CUDA; spawn + no
# flashinfer JIT). The exact resolved vLLM version is recorded into the result at runtime.
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
    .env(
        {
            "HF_HOME": HF_CACHE,
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
    .workdir(REMOTE_ROOT)
    .run_commands("pip install --no-deps -e .")
)

hf_cache = modal.Volume.from_name("llm-infer-hf-cache", create_if_missing=True)
merged = modal.Volume.from_name("llm-infer-merged")  # must exist — written by merge_adapter.py


def _sample_texts(tokenizer, outputs: dict[str, list[int]], eos: set[int], k: int = 2) -> dict:
    """Decode the first ``k`` completions to text — the bf16-sanity eyeball for the smoke."""
    texts = {}
    for rid, ids in list(outputs.items())[:k]:
        trimmed = []
        for tok in ids:
            trimmed.append(tok)
            if tok in eos:
                break
        texts[rid] = tokenizer.decode(trimmed, skip_special_tokens=True)
    return texts


@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    volumes={HF_CACHE: hf_cache, MERGED_MOUNT: merged},
    timeout=90 * 60,
)
def rollout_vllm(
    num_prompts: int,
    num_generations: int,
    max_completion: int,
    warmup: int,
    iters: int,
    with_sample_text: bool,
) -> str:
    """The vLLM ceiling, serving the merged grpo-s0 bf16 weights under the rollout sampling."""
    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.runners import run_vllm
    from llm_infer.benchmarks.workload import build_rollout_workload

    hf_cache.commit()
    workload = build_rollout_workload(
        MERGED_MODEL_PATH,
        served_model_revision=None,
        num_prompts=num_prompts,
        num_generations=num_generations,
        max_completion_length=max_completion,
    )
    result = run_vllm(workload, warmup=warmup, iters=iters)

    sample_texts = {}
    if with_sample_text:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
        sample_texts = _sample_texts(tokenizer, result.outputs, set(workload.eos_token_ids))
    hf_cache.commit()
    return json.dumps(
        {
            "outputs": {k: [int(x) for x in v] for k, v in result.outputs.items()},
            "per_iter_seconds": [float(s) for s in result.per_iter_seconds],
            "config": result.config,
            "sample_texts": sample_texts,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=flash_image,
    gpu="A100-80GB",
    volumes={HF_CACHE: hf_cache, MERGED_MOUNT: merged},
    timeout=90 * 60,
)
def rollout_engine_and_hf(
    num_prompts: int,
    num_generations: int,
    max_completion: int,
    warmup: int,
    iters: int,
    with_sample_text: bool,
    profile: bool,
) -> str:
    """This engine (flash, bf16, seeded sampler) and the naive HF floor, on the merged weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.runners import BLOCK_SIZE, run_hf_sequential, run_llm_infer
    from llm_infer.benchmarks.workload import build_rollout_workload
    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.model.qwen import QwenModel

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    hf_cache.commit()
    workload = build_rollout_workload(
        MERGED_MODEL_PATH,
        served_model_revision=None,
        num_prompts=num_prompts,
        num_generations=num_generations,
        max_completion_length=max_completion,
    )
    eos = set(workload.eos_token_ids)

    # Size the paged pool to admit every completion at once (full-batch continuous batching).
    needed = sum(
        math.ceil((length + max_completion) / BLOCK_SIZE) for length in workload.prompt_lengths
    )
    num_blocks = needed + 8

    engine_model = QwenModel.load(
        dtype=torch.bfloat16,
        backend=FlashAttnPagedAttention(),
        device="cuda",
        model_id=MERGED_MODEL_PATH,
        revision=None,
    )
    infer = run_llm_infer(
        engine_model,
        workload,
        num_blocks=num_blocks,
        warmup=warmup,
        iters=iters,
        collect_profile=profile,
        enable_prefix_caching=True,
    )

    hf_model = (
        AutoModelForCausalLM.from_pretrained(MERGED_MODEL_PATH, dtype=torch.bfloat16)
        .eval()
        .to("cuda")
    )
    hf_seq = run_hf_sequential(hf_model, workload, warmup=warmup, iters=iters)
    hf_cache.commit()

    sample_texts = {"llm_infer": {}, "hf_sequential": {}}
    if with_sample_text:
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
        sample_texts["llm_infer"] = _sample_texts(tokenizer, infer.outputs, eos)
        sample_texts["hf_sequential"] = _sample_texts(tokenizer, hf_seq.outputs, eos)

    def _section(run) -> dict:
        return {
            "outputs": {k: [int(x) for x in v] for k, v in run.outputs.items()},
            "per_iter_seconds": [float(s) for s in run.per_iter_seconds],
            "config": run.config,
            "profiles": run.profiles,
        }

    return json.dumps(
        {
            "llm_infer": _section(infer),
            "hf_sequential": _section(hf_seq),
            "num_blocks": num_blocks,
            "sample_texts": sample_texts,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(
    command: str = "rollout",
    num_prompts: int = 0,
    num_generations: int = 0,
    max_completion: int = 0,
    warmup: int = -1,
    iters: int = 0,
    profile: bool = False,
) -> None:
    """Orchestrate both GPU functions, assemble the pinned rollout record, print the table.

    smoke:   cheap wiring + bf16 sanity (2 prompts × 1 gen, 64 tokens, 1 iter, decoded samples).
    rollout: the frozen GRPO batch (8 prompts × 4 gens, ≤1024 tokens, 1 warmup + 2 measured).
    """
    from llm_infer.benchmarks import (
        A100_80GB_USD_PER_HOUR,
        assemble_rollout_markdown,
        rollout_rows,
    )
    from llm_infer.benchmarks.workload import ROLLOUT_FIXTURE, build_rollout_workload

    if command == "smoke":
        defaults = {
            "num_prompts": 2,
            "num_generations": 1,
            "max_completion": 64,
            "warmup": 0,
            "iters": 1,
        }
    elif command == "rollout":
        defaults = {
            "num_prompts": 8,
            "num_generations": 4,
            "max_completion": 1024,
            "warmup": 1,
            "iters": 2,
        }
    else:
        raise ValueError(f"command must be 'smoke' or 'rollout', got {command!r}")
    n_prompts = num_prompts or defaults["num_prompts"]
    g = num_generations or defaults["num_generations"]
    max_comp = max_completion or defaults["max_completion"]
    w = defaults["warmup"] if warmup < 0 else warmup
    it = iters or defaults["iters"]
    with_sample_text = command == "smoke"

    fixture = json.loads(ROLLOUT_FIXTURE.read_text(encoding="utf-8"))
    workload = build_rollout_workload(
        MERGED_MODEL_PATH,
        num_prompts=n_prompts,
        num_generations=g,
        max_completion_length=max_comp,
    )
    num_completions = len(workload.requests)
    print(
        f"[rollout] {command}: {n_prompts} prompts × G={g} = {num_completions} completions, "
        f"≤{max_comp} tok, temp={workload.sampling.temperature} top_p={workload.sampling.top_p} "
        f"seed={workload.sampling.seed}, warmup={w}, iters={it}"
    )
    print("[rollout] running vLLM ceiling (own image, A100) ...")
    vllm_res = json.loads(rollout_vllm.remote(n_prompts, g, max_comp, w, it, with_sample_text))
    print("[rollout] running llm-infer + HF floor (flash image, A100) ...")
    main_res = json.loads(
        rollout_engine_and_hf.remote(n_prompts, g, max_comp, w, it, with_sample_text, profile)
    )

    rows_in = [
        {
            "system": "hf_sequential",
            "outputs": main_res["hf_sequential"]["outputs"],
            "per_iter_seconds": main_res["hf_sequential"]["per_iter_seconds"],
        },
        {
            "system": "llm_infer",
            "outputs": main_res["llm_infer"]["outputs"],
            "per_iter_seconds": main_res["llm_infer"]["per_iter_seconds"],
        },
        {
            "system": "vllm",
            "outputs": vllm_res["outputs"],
            "per_iter_seconds": vllm_res["per_iter_seconds"],
        },
    ]
    rows = rollout_rows(
        rows_in, workload.eos_token_ids, num_completions, baseline_system="hf_sequential"
    )

    config = {
        "command": command,
        "anchor": "grpo-s0",
        "gpu": {"llm_infer_and_hf": main_res["gpu"], "vllm": vllm_res["gpu"]},
        "versions": {"llm_infer_and_hf": main_res["versions"], "vllm": vllm_res["versions"]},
        "usd_per_hour": A100_80GB_USD_PER_HOUR,
        "served_model": {
            "merged_path": MERGED_MODEL_PATH,
            "base_id": fixture["model"]["id"],
            "base_revision": fixture["model"]["revision"],
            "adapter": "llm-rlvr-sql grpo/grpo-s0 (rank 32), merged bf16",
            "dtype": "bfloat16",
        },
        "workload": {
            "completions": num_completions,
            "num_prompts": n_prompts,
            "num_generations": g,
            "max_completion_length": max_comp,
            "temperature": workload.sampling.temperature,
            "top_p": workload.sampling.top_p,
            "seed": workload.sampling.seed,
            "warmup": w,
            "iters": it,
            "dataset": fixture["dataset"],
            "selection": fixture["selection"],
            "source": workload.source,
            "eos_token_ids": sorted(workload.eos_token_ids),
        },
        "system_config": {
            "llm_infer": {**main_res["llm_infer"]["config"], "num_blocks": main_res["num_blocks"]},
            "hf_sequential": main_res["hf_sequential"]["config"],
            "vllm": vllm_res["config"],
        },
        "profile": {
            "requested": profile,
            "diagnostic_only": True,
            "note": "Profiles are collected in an extra llm-infer run outside headline timing.",
        },
        "profiles": {"llm_infer": main_res["llm_infer"].get("profiles", [])},
        "repro_command": (
            f"modal run scripts/modal_rollout.py --command {command}"
            f"{' --profile' if profile else ''}"
        ),
    }

    print("\n" + assemble_rollout_markdown(rows, config) + "\n")
    if with_sample_text:
        print("[bf16-sanity] sample completions (first request per system):")
        for system, texts in (
            ("vllm", vllm_res.get("sample_texts", {})),
            ("llm_infer", main_res.get("sample_texts", {}).get("llm_infer", {})),
            ("hf_sequential", main_res.get("sample_texts", {}).get("hf_sequential", {})),
        ):
            first = next(iter(texts.items()), None)
            if first:
                snippet = first[1][:240].replace("\n", " ⏎ ")
                print(f"  {system} [{first[0]}]: {snippet}")

    record = {"rows": rows, "config": config}
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"rollout-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[rollout] wrote {out_path}")
