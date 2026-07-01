"""Historical Qwen Modal A100 benchmark: naive HF vs llm-infer vs vLLM, config pinned.

Archived public-baseline evidence for Qwen2.5-Coder. Current benchmark work starts with Esme
(``scripts/modal_esme_three_way.py``); keep this harness for reproducibility and regression
coverage. Two GPU functions on the project's target A100-80GB:

* ``bench_engine_and_hf`` (flash-attn image, shared with the reference check) runs the two naive
  HF baselines and this engine's flash backend, then adjudicates cross-system token
  equivalence with the fp32 reference under the reference check's tie policy.
* ``bench_vllm`` (its own image — vLLM ships its own torch/CUDA) runs the vLLM ceiling.

The local entrypoint calls vLLM first, hands its tokens to the HF/engine function for
adjudication, then assembles a single pinned record (GPU + clocks, every library version,
vLLM flags, workload, repro command) and prints the three-way table. Only systems whose
greedy tokens match the reference (exactly or at a traced numerical tie) get a tok/s
number — match before measuring speed.

    modal run scripts/modal_benchmark.py --command smoke    # cheap wiring check (N=2, 8 tok)
    modal run scripts/modal_benchmark.py --command bench    # full run (N=32, 128 tok)
    modal run scripts/modal_benchmark.py --command bench --profile  # diagnostic extra run
"""

from __future__ import annotations

import json
import math
import time

import modal

from scripts.modal_flash_image import FLASH_IMAGE, IGNORE, REMOTE_ROOT, REPO_ROOT

HF_CACHE = "/hf-cache"

app = modal.App("llm-infer-benchmark")

# Flash image: the one shared flash-attn image (scripts/modal_flash_image.py; flash-attn from a
# prebuilt wheel), with HF_HOME chained on for the model cache. The .env is after the flash-attn
# layer, so it never invalidates it — every GPU harness shares one image.
flash_image = FLASH_IMAGE.env({"HF_HOME": HF_CACHE})

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
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=IGNORE)
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
    profile: bool,
) -> dict:
    """Run naive-HF (sequential + batched) and this engine, then adjudicate all equivalence.

    The reference is the fp32 full-recompute engine output for each unique prompt.
    HF sequential, HF batched, llm_infer, and the passed-in vLLM tokens are compared
    against it under the reference check's tie policy: exact match, or a first
    divergence the fp32 reference proves is a genuine numerical tie. A non-tie
    divergence marks that system non-equivalent (no tok/s).
    """
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
    from llm_infer.serving import InferenceEngine, Request
    from llm_infer.validation.tie_tolerance import compare_under_tie_tolerance

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
    infer = run_llm_infer(
        engine_model,
        workload,
        num_blocks=num_blocks,
        warmup=warmup,
        iters=iters,
        collect_profile=profile,
    )

    # The equivalence reference is the fp32 full-recompute output, NOT bf16 HF
    # generate. generate() itself diverges from reference at real margins (the documented step-32
    # case), so using it as the reference wrongly fails any backend that is *more* faithful to
    # fp32. Compute reference by running each UNIQUE prompt through the engine on the fp32 model —
    # cached fp32 decode == fp32 full-recompute (paged-cache), the same tokens at O(n) not O(n^2).
    fp32_model = QwenModel.load(dtype=torch.float32, device="cuda")
    reference_by_prompt: dict[tuple, list[int]] = {}
    for req in workload.requests:
        if req.prompt_ids not in reference_by_prompt:
            eng = InferenceEngine(
                fp32_model, block_size=BLOCK_SIZE, num_blocks=num_blocks, device="cuda"
            )
            eng.add_request(Request("reference", list(req.prompt_ids), max_new_tokens, eos))
            reference_by_prompt[req.prompt_ids] = eng.run()["reference"]
    reference = {
        r.request_id: normalize_at_eos(reference_by_prompt[r.prompt_ids], eos)
        for r in workload.requests
    }

    # bf16 noise at these logit magnitudes (~20-25) is ~0.05-0.1, so a genuine
    # bf16 tie can flip
    # within ~0.1; the fp32-sized 1e-3 tolerance wrongly calls those flips
    # "real". A first divergence
    # whose fp32 top-2 gap exceeds this is a real reduction-order divergence (e.g. HF generate's
    # ~0.4 step-32 effect) — reported transparently, not laundered as a tie.
    bf16_tolerance = 0.1

    def comparison_vs_reference(outputs: dict[str, list[int]]) -> dict:
        exact, ties, divergences = 0, [], []
        missing = sorted(set(reference) - set(outputs))
        extra = sorted(set(outputs) - set(reference))
        if missing:
            divergences.append(
                {"request": missing[0], "detail": f"missing outputs for {missing[:3]}"}
            )
        if extra:
            divergences.append(
                {"request": extra[0], "detail": f"unexpected outputs for {extra[:3]}"}
            )
        for rid in reference:
            if rid not in outputs:
                continue
            raw = outputs[rid]
            fast = normalize_at_eos(raw, eos)
            gold = reference[rid]
            if fast == gold:
                exact += 1
                continue
            res = compare_under_tie_tolerance(
                fp32_model, prompts_by_id[rid], fast, gold, tolerance=bf16_tolerance
            )
            if res.ok and res.divergence is not None:
                d = res.divergence
                ties.append({"request": rid, "step": d.step, "gap": d.reference_gap})
            elif not res.ok:
                divergences.append({"request": rid, "detail": res.failure})
        return {
            "exact": exact,
            "tie": len(ties),
            "nontie": len(divergences),
            "total": len(reference),
            # Matches fp32 reference except at genuine bf16 ties.
            "all_ties_or_exact": not divergences,
            "ties_sample": ties[:3],
            "divergences_sample": divergences[:3],
        }

    agreement = {
        "hf_sequential": comparison_vs_reference(hf_seq.outputs),
        "hf_batched": comparison_vs_reference(hf_bat.outputs),
        "llm_infer": comparison_vs_reference(infer.outputs),
        "vllm": comparison_vs_reference(vllm_outputs),
    }
    hf_cache.commit()

    def _section(run) -> dict:
        return {
            "outputs": {k: [int(x) for x in v] for k, v in run.outputs.items()},
            "per_iter_seconds": [float(s) for s in run.per_iter_seconds],
            "config": run.config,
            "profiles": run.profiles,
        }

    # JSON string, not a dict — see bench_vllm: the local entrypoint env has no torch, so the
    # crossing payload must be JSON-native (coerced here, in the GPU env that has torch).
    return json.dumps(
        {
            "hf_sequential": _section(hf_seq),
            "hf_batched": _section(hf_bat),
            "llm_infer": _section(infer),
            "comparison_vs_fp32_reference": agreement,
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
    profile: bool = False,
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
    effective_num_requests = num_requests or defaults["num_requests"]
    effective_max_new_tokens = max_new_tokens or defaults["max_new_tokens"]
    effective_warmup = defaults["warmup"] if warmup < 0 else warmup
    effective_iters = iters or defaults["iters"]

    workload = build_workload(effective_num_requests, effective_max_new_tokens)
    print(
        f"[benchmark] {command}: {effective_num_requests} requests x "
        f"{effective_max_new_tokens} new tokens, warmup={effective_warmup}, iters={effective_iters}"
    )
    print("[benchmark] running vLLM (own image, A100) ...")
    vllm_res = json.loads(
        bench_vllm.remote(
            effective_num_requests, effective_max_new_tokens, effective_warmup, effective_iters
        )
    )
    print("[benchmark] running naive HF + llm-infer + fp32-reference agreement ...")
    main_res = json.loads(
        bench_engine_and_hf.remote(
            effective_num_requests,
            effective_max_new_tokens,
            effective_warmup,
            effective_iters,
            vllm_res["outputs"],
            profile,
        )
    )

    agree = main_res["comparison_vs_fp32_reference"]
    rows_in = [
        {
            "system": "hf_sequential",
            "outputs": main_res["hf_sequential"]["outputs"],
            "per_iter_seconds": main_res["hf_sequential"]["per_iter_seconds"],
            "matches_reference": agree["hf_sequential"]["all_ties_or_exact"],
        },
        {
            "system": "hf_batched",
            "outputs": main_res["hf_batched"]["outputs"],
            "per_iter_seconds": main_res["hf_batched"]["per_iter_seconds"],
            "matches_reference": agree["hf_batched"]["all_ties_or_exact"],
        },
        {
            "system": "llm_infer",
            "outputs": main_res["llm_infer"]["outputs"],
            "per_iter_seconds": main_res["llm_infer"]["per_iter_seconds"],
            "matches_reference": agree["llm_infer"]["all_ties_or_exact"],
        },
        {
            "system": "vllm",
            "outputs": vllm_res["outputs"],
            "per_iter_seconds": vllm_res["per_iter_seconds"],
            "matches_reference": agree["vllm"]["all_ties_or_exact"],
        },
    ]
    rows = throughput_rows(rows_in, workload.eos_token_ids, baseline_system="hf_sequential")

    config = {
        "command": command,
        "gpu": {"hf_and_engine": main_res["gpu"], "vllm": vllm_res["gpu"]},
        "versions": {"hf_and_engine": main_res["versions"], "vllm": vllm_res["versions"]},
        "workload": {
            "num_requests": effective_num_requests,
            "max_new_tokens": effective_max_new_tokens,
            "warmup": effective_warmup,
            "iters": effective_iters,
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
        "profile": {
            "requested": profile,
            "diagnostic_only": True,
            "note": "Profiles are collected in an extra llm-infer run outside headline timing.",
        },
        "profiles": {"llm_infer": main_res["llm_infer"].get("profiles", [])},
        "repro_command": (
            f"modal run scripts/modal_benchmark.py --command {command}"
            f"{' --profile' if profile else ''}"
        ),
    }

    print("\n" + assemble_markdown(rows, config) + "\n")
    for system, prof in agree.items():
        line = (
            f"[fp32-reference] {system}: exact={prof['exact']}/{prof['total']} "
            f"tie={prof['tie']} non-tie={prof['nontie']}"
        )
        if prof["nontie"]:
            line += f" | sample: {prof['divergences_sample'][:1]}"
        print(line)

    record = {"rows": rows, "config": config, "comparison_vs_fp32_reference": agree}
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {out_path}")
