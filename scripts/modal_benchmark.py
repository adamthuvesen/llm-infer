"""Modal A100 three-way benchmark: naive HF vs llm-infer vs vLLM, config fully pinned.

Phase D evidence. Two GPU functions on the project's target A100-80GB:

* ``bench_engine_and_hf`` (flash-attn image, shared with the oracle) runs the two naive
  HF baselines and this engine's flash backend, then adjudicates cross-system token
  equivalence with the fp32 reference under the oracle's tie policy.
* ``bench_vllm`` (its own image — vLLM ships its own torch/CUDA) runs the vLLM ceiling.

The local entrypoint calls vLLM first, hands its tokens to the HF/engine function for
adjudication, then assembles a single pinned record (GPU + clocks, every library version,
vLLM flags, workload, repro command) and prints the three-way table. Only systems whose
greedy tokens match the reference (exactly or at a traced numerical tie) get a tok/s
number — validate before you brag.

    modal run scripts/modal_benchmark.py --command smoke    # cheap wiring check (N=2, 8 tok)
    modal run scripts/modal_benchmark.py --command bench    # full run (N=32, 128 tok)
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
# devel CUDA image matching torch's CUDA major so flash-attn compiles — copied verbatim
# from scripts/modal_oracle.py so it shares that baked, cached flash-attn layer (the 40-min
# build runs at most once across both scripts).
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"

app = modal.App("llm-infer-benchmark")

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

# Flash image: identical prefix to modal_oracle's, so the expensive flash-attn layer is a
# cache hit; only the source-copy + editable-install tail relinks for new benchmark code.
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

# vLLM image: vLLM pulls its own torch + CUDA runtime, so it must not share the flash env.
# Two env pins keep it running on a toolkit-less (no nvcc) base:
#   * VLLM_WORKER_MULTIPROC_METHOD=spawn — the V1 engine core *spawns* (not forks) its
#     subprocess; a forked child cannot re-initialize CUDA, which aborts the engine.
#   * VLLM_USE_FLASHINFER_SAMPLER=0 — use the native PyTorch top-k/top-p sampler instead of
#     FlashInfer's, which JIT-compiles a CUDA kernel (needs nvcc) at startup. We decode
#     greedy (temp 0 = argmax), so the sampler backend cannot change tokens, and throughput
#     stays on vLLM's optimized FLASH_ATTN attention path (precompiled, no JIT).
# The exact resolved vLLM version is recorded into the result at runtime (reproducible).
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


@app.function(image=vllm_image, gpu="A100-80GB", volumes={HF_CACHE: hf_cache}, timeout=60 * 60)
def bench_vllm(num_requests: int, max_new_tokens: int, warmup: int, iters: int) -> dict:
    """Run the vLLM ceiling on the A100; return its tokens, timing, flags, and environment.

    Deliberately does NOT touch ``torch.cuda`` in this parent process — initializing CUDA
    here and then letting vLLM fork its engine core is what triggers the re-init abort. vLLM
    owns CUDA in its own (spawned) worker; a missing GPU surfaces loudly from ``LLM(...)``.
    """
    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.runners import run_vllm
    from llm_infer.benchmarks.workload import build_workload

    hf_cache.commit()
    workload = build_workload(num_requests, max_new_tokens)
    result = run_vllm(workload, warmup=warmup, iters=iters)
    hf_cache.commit()
    # Return a JSON string, not a dict: the local `modal run` entrypoint runs in modal's own
    # (torch-less) env, so any torch/numpy scalar in a pickled result fails to deserialize
    # there. Coercing to JSON-native types here (the GPU env, which has torch) makes the
    # boundary bulletproof regardless of what dtype vLLM hands back.
    return json.dumps(
        {
            "outputs": {k: [int(x) for x in v] for k, v in result.outputs.items()},
            "per_iter_seconds": [float(s) for s in result.per_iter_seconds],
            "config": result.config,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(image=flash_image, gpu="A100-80GB", volumes={HF_CACHE: hf_cache}, timeout=60 * 60)
def bench_engine_and_hf(
    num_requests: int,
    max_new_tokens: int,
    warmup: int,
    iters: int,
    vllm_outputs: dict,
) -> dict:
    """Run naive-HF (sequential + batched) and this engine, then adjudicate all equivalence.

    The reference is ``hf_sequential`` (HF's own greedy generate). Every other system —
    hf_batched, llm_infer, and the passed-in vLLM tokens — is checked against it under the
    oracle's tie policy: exact match, or a first divergence the fp32 reference proves is a
    genuine numerical tie. A non-tie divergence marks that system non-equivalent (no tok/s).
    """
    import sys

    sys.path.insert(0, REMOTE_ROOT)  # make the tests.* tie-policy importable (see generate_goldens)
    import torch
    from transformers import AutoModelForCausalLM

    from llm_infer.benchmarks import gpu_snapshot, library_versions, normalize_at_eos
    from llm_infer.benchmarks.runners import (
        BLOCK_SIZE,
        run_hf_batched,
        run_hf_sequential,
        run_llm_infer,
    )
    from llm_infer.benchmarks.workload import build_workload
    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.model.qwen import QwenModel
    from tests.correctness.tie_tolerance import DEFAULT_TIE_TOLERANCE, compare_under_tie_tolerance

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    hf_cache.commit()
    workload = build_workload(num_requests, max_new_tokens)
    eos = workload.eos_token_ids
    prompts_by_id = {r.request_id: list(r.prompt_ids) for r in workload.requests}

    # Size the paged pool to admit every request at once (full-batch continuous batching).
    needed = sum(
        math.ceil((length + max_new_tokens) / BLOCK_SIZE) for length in workload.prompt_lengths
    )
    num_blocks = needed + 8

    hf_model = (
        AutoModelForCausalLM.from_pretrained(
            workload.model_id, revision=workload.model_revision, dtype=torch.bfloat16
        )
        .eval()
        .to("cuda")
    )
    hf_cache.commit()
    hf_seq = run_hf_sequential(hf_model, workload, warmup=warmup, iters=iters)
    hf_bat = run_hf_batched(hf_model, workload, warmup=warmup, iters=iters)

    engine_model = QwenModel.load(
        dtype=torch.bfloat16, backend=FlashAttnPagedAttention(), device="cuda"
    )
    infer = run_llm_infer(engine_model, workload, num_blocks=num_blocks, warmup=warmup, iters=iters)

    reference = {rid: normalize_at_eos(ids, eos) for rid, ids in hf_seq.outputs.items()}

    # Lazily load the fp32 reference (heavy) only if some system actually diverges.
    _fp32: list[QwenModel] = []

    def fp32_reference() -> QwenModel:
        if not _fp32:
            _fp32.append(QwenModel.load(dtype=torch.float32, device="cuda"))
        return _fp32[0]

    def adjudicate(outputs: dict[str, list[int]]) -> dict:
        ties, failures, length_mismatch = [], [], []
        for rid, raw in outputs.items():
            fast = normalize_at_eos(raw, eos)
            gold = reference[rid]
            if fast != gold:
                res = compare_under_tie_tolerance(
                    fp32_reference(),
                    prompts_by_id[rid],
                    fast,
                    gold,
                    tolerance=DEFAULT_TIE_TOLERANCE,
                )
                if not res.ok:
                    failures.append({"request": rid, "failure": res.failure})
                elif res.divergence is not None:
                    d = res.divergence
                    ties.append(
                        {
                            "request": rid,
                            "step": d.step,
                            "fast_token": d.fast_token,
                            "golden_token": d.golden_token,
                            "reference_gap": d.reference_gap,
                        }
                    )
            if len(fast) != len(gold):
                length_mismatch.append(
                    {"request": rid, "fast_len": len(fast), "ref_len": len(gold)}
                )
        return {
            "equivalent": not failures,
            "ties": ties,
            "failures": failures,
            "length_mismatch": length_mismatch,
        }

    equivalence = {
        "hf_batched": adjudicate(hf_bat.outputs),
        "llm_infer": adjudicate(infer.outputs),
        "vllm": adjudicate(vllm_outputs),
    }
    hf_cache.commit()

    def _section(run) -> dict:
        return {
            "outputs": {k: [int(x) for x in v] for k, v in run.outputs.items()},
            "per_iter_seconds": [float(s) for s in run.per_iter_seconds],
            "config": run.config,
        }

    # JSON string, not a dict — see bench_vllm: the local entrypoint env has no torch, so the
    # crossing payload must be JSON-native (coerced here, in the GPU env that has torch).
    return json.dumps(
        {
            "hf_sequential": _section(hf_seq),
            "hf_batched": _section(hf_bat),
            "llm_infer": _section(infer),
            "equivalence": equivalence,
            "num_blocks": num_blocks,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(
    command: str = "bench",
    num_requests: int = 0,
    max_new_tokens: int = 0,
    warmup: int = -1,
    iters: int = 0,
) -> None:
    """Orchestrate both GPU functions, assemble the pinned record, print the table, write JSON.

    smoke: cheap end-to-end wiring check (N=2, 8 new tokens, 1 iter, no warmup).
    bench: the real run (N=32, 128 new tokens, 1 warmup + 3 measured iters).
    Explicit flags override the per-command defaults.
    """
    from llm_infer.benchmarks import assemble_markdown, throughput_rows
    from llm_infer.benchmarks.workload import build_workload

    if command == "smoke":
        defaults = {"num_requests": 2, "max_new_tokens": 8, "warmup": 0, "iters": 1}
    elif command == "bench":
        defaults = {"num_requests": 32, "max_new_tokens": 128, "warmup": 1, "iters": 3}
    else:
        raise ValueError(f"command must be 'smoke' or 'bench', got {command!r}")
    n = num_requests or defaults["num_requests"]
    m = max_new_tokens or defaults["max_new_tokens"]
    w = defaults["warmup"] if warmup < 0 else warmup
    it = iters or defaults["iters"]

    workload = build_workload(n, m)
    print(f"[benchmark] {command}: {n} requests x {m} new tokens, warmup={w}, iters={it}")
    print("[benchmark] running vLLM (own image, A100) ...")
    vllm_res = json.loads(bench_vllm.remote(n, m, w, it))
    print("[benchmark] running naive HF + llm-infer + equivalence (flash image, A100) ...")
    main_res = json.loads(bench_engine_and_hf.remote(n, m, w, it, vllm_res["outputs"]))

    eq = main_res["equivalence"]
    rows_in = [
        {
            "system": "hf_sequential",
            "outputs": main_res["hf_sequential"]["outputs"],
            "per_iter_seconds": main_res["hf_sequential"]["per_iter_seconds"],
            "equivalent": True,  # the reference is equivalent to itself by definition
        },
        {
            "system": "hf_batched",
            "outputs": main_res["hf_batched"]["outputs"],
            "per_iter_seconds": main_res["hf_batched"]["per_iter_seconds"],
            "equivalent": eq["hf_batched"]["equivalent"],
        },
        {
            "system": "llm_infer",
            "outputs": main_res["llm_infer"]["outputs"],
            "per_iter_seconds": main_res["llm_infer"]["per_iter_seconds"],
            "equivalent": eq["llm_infer"]["equivalent"],
        },
        {
            "system": "vllm",
            "outputs": vllm_res["outputs"],
            "per_iter_seconds": vllm_res["per_iter_seconds"],
            "equivalent": eq["vllm"]["equivalent"],
        },
    ]
    rows = throughput_rows(rows_in, workload.eos_token_ids, baseline_system="hf_sequential")

    config = {
        "command": command,
        "gpu": {"hf_and_engine": main_res["gpu"], "vllm": vllm_res["gpu"]},
        "versions": {"hf_and_engine": main_res["versions"], "vllm": vllm_res["versions"]},
        "workload": {
            "num_requests": n,
            "max_new_tokens": m,
            "warmup": w,
            "iters": it,
            "model_id": workload.model_id,
            "model_revision": workload.model_revision,
            "eos_token_ids": sorted(workload.eos_token_ids),
            "prompt_lengths": list(workload.prompt_lengths),
            "source": workload.source,
            "greedy": True,
        },
        "system_config": {
            "hf_sequential": main_res["hf_sequential"]["config"],
            "hf_batched": main_res["hf_batched"]["config"],
            "llm_infer": {**main_res["llm_infer"]["config"], "num_blocks": main_res["num_blocks"]},
            "vllm": vllm_res["config"],
        },
        "repro_command": f"modal run scripts/modal_benchmark.py --command {command}",
    }

    print("\n" + assemble_markdown(rows, config) + "\n")
    for system, detail in eq.items():
        if detail["failures"]:
            print(f"[equivalence] {system}: NON-EQUIVALENT — {detail['failures']}")
        elif detail["ties"]:
            print(
                f"[equivalence] {system}: equivalent with "
                f"{len(detail['ties'])} traced tie(s): {detail['ties']}"
            )
        else:
            print(f"[equivalence] {system}: exact token match vs naive HF")

    record = {"rows": rows, "config": config, "equivalence": eq}
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {out_path}")
