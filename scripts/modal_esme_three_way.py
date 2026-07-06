"""Modal A100 three-way benchmark for Esme-214M-Chat: naive HF vs llm_infer vs vLLM.

The primary Esme external-baseline harness. Esme has no public HF/vLLM checkpoint, so
this harness converts the ``llm_pretrain_dense_v1`` bundle to a ``Qwen3ForCausalLM`` checkpoint
(``scripts/convert_esme_to_hf.py``). The local ``modal run`` client has no torch, so it only
uploads the raw bundle files to the repo-scoped Modal volume ``llm-infer-esme-bundles``; the
torch-importing conversion runs **remotely** in :func:`convert_bundle_to_hf` on ``FLASH_IMAGE`` and
writes the HF checkpoint back to the volume. Then it runs:

* ``hf_sequential`` — naive HF ``Qwen3ForCausalLM.generate()`` per request (the floor).
* ``llm_infer`` — this engine's Esme paged-KV path on the bundle, bf16 on the **flash-attn**
  backend (the fast path). flash is correct: bf16 flash ==
  bf16 torch_naive exactly; its only divergence from the fp32 oracle is whole-model bf16 rounding,
  counted as a genuine tie by the tie-tolerant agreement below.
* ``vllm`` — vLLM offline generate on the converted checkpoint (the ceiling).

Agreement uses the audited tie-tolerant rule against the fp32 oracle
(``compare_under_tie_tolerance``, tolerance 0.1): a bf16 system whose only divergences are
genuine numerical ties counts as agreement and reports tok/s; a non-tie divergence reports no tok/s
(match before measuring speed), with the exact/tie/non-tie profile printed per system. vLLM runs in
its own image (it ships its own torch/CUDA); the HF + llm_infer leg runs
in the shared flash CUDA image. Both functions are A100-80GB.

    modal run scripts/modal_esme_three_way.py --command smoke     # N=2, 8 tokens, 1 iter
    modal run scripts/modal_esme_three_way.py --command bench     # N=8, 64 tokens, 1+3 iters
    modal run scripts/modal_esme_three_way.py --command headline  # N=64, 256 tokens, 1+3 iters

``smoke``/``bench`` cycle the four short DEFAULT_PROMPTS (the cheap oracle-gated regression
workload); ``headline`` is the published serving shape — 64 chat requests over the wider
HEADLINE_PROMPTS pool, up to 256 new tokens each.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import modal

from scripts.modal_esme_bundle import (
    ESME_BUNDLE_MOUNT,
    REMOTE_BUNDLE_PATH,
    VOLUME_NAME,
    local_bundle_path,
    stage_bundle,
)
from scripts.modal_flash_image import FLASH_IMAGE, IGNORE, REMOTE_ROOT, REPO_ROOT

ESME_HF_DIR = "esme-214m-chat-hf"
REMOTE_HF_PATH = f"{ESME_BUNDLE_MOUNT}/{ESME_HF_DIR}"
HF_CHECKPOINT_FILES = ("config.json", "model.safetensors", "tokenizer.json")
BLOCK_SIZE = 128

app = modal.App("llm-infer-esme-three-way")

# HF + llm_infer image: the one shared flash-attn image (scripts/modal_flash_image.py), because the
# llm_infer row runs on FlashAttnPagedAttention. flash-attn installs from a prebuilt wheel there
# (no source compile); importing the shared definition keeps every GPU harness on one cached build.
esme_image = FLASH_IMAGE

# vLLM image: vLLM pulls its own torch + CUDA, so it must not share the flash env. Same two env
# pins for the vLLM ceiling run: spawn the engine core, native sampler, no nvcc JIT.
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm")
    .env(
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=IGNORE)
    .workdir(REMOTE_ROOT)
    .run_commands("pip install --no-deps -e .")
)

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(image=esme_image, volumes={ESME_BUNDLE_MOUNT: esme_bundles}, timeout=30 * 60)
def convert_bundle_to_hf() -> str:
    """Convert the staged raw bundle to an HF Qwen3 checkpoint on the volume (remote, has torch).

    Runs on ``FLASH_IMAGE`` (torch present), reads the raw bundle the local entrypoint staged, and
    writes the HF checkpoint back to the volume so the HF and vLLM legs can load it. CPU-only — the
    conversion is a key remap, no GPU. ``commit()`` makes the written files visible to later
    functions reading the same volume.
    """
    from scripts.convert_esme_to_hf import convert

    config = convert(Path(REMOTE_BUNDLE_PATH), Path(REMOTE_HF_PATH), max_position_embeddings=1024)
    esme_bundles.commit()
    written = [name for name in HF_CHECKPOINT_FILES if (Path(REMOTE_HF_PATH) / name).is_file()]
    if sorted(written) != sorted(HF_CHECKPOINT_FILES):
        raise AssertionError(f"converted checkpoint missing files: wrote {written}")
    return (
        f"converted HF Qwen3 checkpoint -> {REMOTE_HF_PATH} "
        f"(layers={config['num_hidden_layers']}, tied={config['tie_word_embeddings']})"
    )


@app.function(
    image=vllm_image,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def bench_vllm(
    num_requests: int, max_new_tokens: int, warmup: int, iters: int, headline: bool
) -> str:
    """Run the vLLM ceiling on the converted Qwen3 checkpoint; return tokens, timing, flags."""
    import torch

    from llm_infer.benchmarks.esme_paged import (
        DEFAULT_PROMPTS,
        HEADLINE_PROMPTS,
        _time,
        build_requests,
    )
    from llm_infer.benchmarks.esme_three_way import run_vllm_esme
    from llm_infer.model.runtime import load_model_runtime

    esme_bundles.reload()  # pick up the HF checkpoint convert_bundle_to_hf committed to the volume
    # Esme tokenizer + EOS from the bundle on CPU; vLLM owns CUDA in its own spawned worker, so
    # this parent never initializes torch.cuda (that re-init is what aborts a forked engine core).
    runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cpu"
    )
    prompts = HEADLINE_PROMPTS if headline else DEFAULT_PROMPTS
    requests = build_requests(runtime.tokenizer, num_requests, prompts)
    decode_once, vllm_config = run_vllm_esme(
        Path(REMOTE_HF_PATH),
        requests,
        max_new_tokens=max_new_tokens,
        eos_token_ids=runtime.eos_token_ids,
    )
    median_s, outputs = _time(decode_once, warmup=warmup, iters=iters, sync=False)
    return json.dumps(
        {
            "outputs": {k: [int(x) for x in v] for k, v in outputs.items()},
            "median_seconds": float(median_s),
            "config": vllm_config,
        }
    )


@app.function(
    image=esme_image,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def bench_hf_and_engine(
    num_requests: int,
    max_new_tokens: int,
    warmup: int,
    iters: int,
    vllm_outputs: dict,
    headline: bool,
) -> str:
    """Run naive HF + llm_infer (flash) on the A100; tie-tolerant agreement vs the fp32 oracle."""
    import torch

    from llm_infer.benchmarks.esme_paged import (
        DEFAULT_PROMPTS,
        HEADLINE_PROMPTS,
        SystemTiming,
        build_requests,
        reference_outputs,
    )
    from llm_infer.benchmarks.esme_three_way import run_three_way, tie_tolerant_agreement
    from llm_infer.benchmarks.report import normalize_at_eos, total_output_tokens
    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.model.decode_graph import DEFAULT_CAPTURE_SIZES, enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()  # pick up the HF checkpoint convert_bundle_to_hf committed to the volume
    # Reference oracle: fp32 PretrainBundleModel.logits() (torch_naive). The llm_infer row runs bf16
    # on the flash-attn backend — the fast path. flash is
    # correct: bf16 flash == bf16 torch_naive exactly; its only divergence from the fp32 oracle is
    # whole-model bf16 rounding (esme-001 step 22, fp32 gap 0.0119), which the tie-tolerant
    # agreement (the audited tie rule, tolerance 0.1) counts as a genuine tie, not a bug. See
    # scripts/modal_esme_flash_divergence_probe.py and docs/benchmark.md.
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    flash_runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
        attention_backend=FlashAttnPagedAttention(),
    )
    if not flash_runtime.capabilities.flash_attention:
        raise AssertionError("Esme llm_infer row must run on the flash-attn backend")
    # The llm_infer row runs what serving runs: decode-window CUDA graphs, captured up front
    # so no capture cost lands inside a timed iteration. The oracle stays eager fp32.
    capture_s = enable_decode_graphs_if_cuda(flash_runtime.model)
    print(f"[esme-3way] decode graphs: captured {DEFAULT_CAPTURE_SIZES} in {capture_s:.1f} s")
    prompts = HEADLINE_PROMPTS if headline else DEFAULT_PROMPTS
    requests = build_requests(oracle_runtime.tokenizer, num_requests, prompts)
    needed = sum(math.ceil((len(req.prompt_ids) + max_new_tokens) / BLOCK_SIZE) for req in requests)
    num_blocks = needed + max(4, len(requests))

    timings, agreements = run_three_way(
        oracle_runtime,
        Path(REMOTE_HF_PATH),
        requests,
        max_new_tokens=max_new_tokens,
        block_size=BLOCK_SIZE,
        num_blocks=num_blocks,
        warmup=warmup,
        iters=iters,
        device="cuda",
        include_vllm=False,
        llm_infer_runtime=flash_runtime,
    )

    # Gate the passed-in vLLM tokens with the SAME tie-tolerant rule against the fp32 oracle.
    eos = oracle_runtime.eos_token_ids
    reference = reference_outputs(
        oracle_runtime.model, requests, max_new_tokens=max_new_tokens, eos_token_ids=eos
    )
    vllm_norm = {k: normalize_at_eos(v, eos) for k, v in vllm_outputs.items()}
    vllm_agreement = tie_tolerant_agreement(
        oracle_runtime.model, requests, vllm_norm, reference, eos
    )
    agreements["vllm"] = vllm_agreement
    vllm_timing = SystemTiming(
        system="vllm",
        mode="vLLM offline generate (ceiling)",
        matches_reference=vllm_agreement.all_ties_or_exact,
        median_seconds=0.0,  # filled from the vLLM function's own measured wall-clock by the caller
        total_output_tokens=total_output_tokens(vllm_outputs, eos),
        outputs=vllm_norm,
    )

    def _row(t: SystemTiming) -> dict:
        agreement = agreements[t.system]
        return {
            "system": t.system,
            "mode": t.mode,
            "matches_reference": t.matches_reference,
            "agreement": {
                "exact": agreement.exact,
                "tie": agreement.tie,
                "nontie": agreement.nontie,
                "total": agreement.total,
                "ties_sample": agreement.ties_sample,
                "divergences_sample": agreement.divergences_sample,
            },
            "median_seconds": t.median_seconds,
            "total_output_tokens": t.total_output_tokens,
            "tokens_per_second": t.tokens_per_second,
        }

    return json.dumps(
        {
            "rows": [_row(t) for t in timings] + [_row(vllm_timing)],
            "num_blocks": num_blocks,
            "vllm_matches_reference": vllm_agreement.all_ties_or_exact,
            "vllm_total_output_tokens": total_output_tokens(vllm_outputs, eos),
            "decode_graphs": {
                "capture_sizes": list(DEFAULT_CAPTURE_SIZES),
                "capture_s": capture_s,
            },
        }
    )


def _print_table(rows: list[dict]) -> None:
    header = (
        "| model | system | agrees fp32 oracle | median s | output tok | tok/s |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for row in rows:
        tps = row["tokens_per_second"]
        tps_cell = f"{tps:.1f}" if tps is not None else "—"
        ag = row["agreement"]
        # Agreement profile: exact / genuine-tie / non-tie divergence counts.
        if ag["nontie"]:
            agree_cell = f"diverges ({ag['nontie']}/{ag['total']} non-tie)"
        elif ag["tie"]:
            agree_cell = f"yes ({ag['exact']} exact, {ag['tie']} tie)"
        else:
            agree_cell = f"yes ({ag['exact']}/{ag['total']} exact)"
        lines.append(
            f"| Esme-214M-Chat | {row['system']} | {agree_cell} | "
            f"{row['median_seconds']:.3f} | {row['total_output_tokens']} | {tps_cell} |"
        )
    print("\n" + "\n".join(lines) + "\n")
    for row in rows:
        ag = row["agreement"]
        line = (
            f"[agreement] {row['system']}: exact={ag['exact']}/{ag['total']} "
            f"tie={ag['tie']} non-tie={ag['nontie']}"
        )
        if ag["nontie"]:
            line += f" | sample: {ag['divergences_sample'][:1]}"
        elif ag["tie"]:
            line += f" | tie sample: {ag['ties_sample'][:1]}"
        print(line)


@app.local_entrypoint()
def main(command: str = "bench", bundle_path: str = "") -> None:
    """Stage the raw bundle, convert it to HF remotely, run all three systems, write the record.

    The local ``modal run`` client has no torch, so it only uploads the raw bundle files (file IO).
    The bundle->HF conversion (which imports torch) runs in the remote :func:`convert_bundle_to_hf`
    on ``FLASH_IMAGE``; the HF + vLLM legs then read the converted checkpoint from the volume.
    """
    if command == "smoke":
        defaults = {"num_requests": 2, "max_new_tokens": 8, "warmup": 0, "iters": 1}
    elif command == "bench":
        defaults = {"num_requests": 8, "max_new_tokens": 64, "warmup": 1, "iters": 3}
    elif command == "headline":
        defaults = {"num_requests": 64, "max_new_tokens": 256, "warmup": 1, "iters": 3}
    else:
        raise ValueError(f"command must be 'smoke', 'bench', or 'headline', got {command!r}")
    headline = command == "headline"
    effective_num_requests = defaults["num_requests"]
    effective_max_new_tokens = defaults["max_new_tokens"]
    effective_warmup = defaults["warmup"]
    effective_iters = defaults["iters"]

    local_bundle = local_bundle_path(bundle_path)
    stage_bundle(esme_bundles, local_bundle, label="esme-3way")
    print("[esme-3way] converting bundle -> HF Qwen3 checkpoint (remote, FLASH_IMAGE) ...")
    print(f"[esme-3way] {convert_bundle_to_hf.remote()}")

    print(
        f"[esme-3way] {command}: Esme-214M-Chat naive HF vs llm_infer vs vLLM, "
        f"{effective_num_requests}x{effective_max_new_tokens}"
    )
    print("[esme-3way] running vLLM (own image, A100) ...")
    vllm_res = json.loads(
        bench_vllm.remote(
            effective_num_requests,
            effective_max_new_tokens,
            effective_warmup,
            effective_iters,
            headline,
        )
    )
    print("[esme-3way] running naive HF + llm_infer + oracle gate ...")
    main_res = json.loads(
        bench_hf_and_engine.remote(
            effective_num_requests,
            effective_max_new_tokens,
            effective_warmup,
            effective_iters,
            vllm_res["outputs"],
            headline,
        )
    )

    rows = main_res["rows"]
    for row in rows:
        if row["system"] == "vllm":
            row["median_seconds"] = vllm_res["median_seconds"]
            tokens = main_res["vllm_total_output_tokens"]
            row["tokens_per_second"] = (
                tokens / vllm_res["median_seconds"]
                if row["matches_reference"] and vllm_res["median_seconds"] > 0
                else None
            )

    _print_table(rows)

    record = {
        "rows": rows,
        "config": {
            "command": command,
            "model": "Esme-214M-Chat",
            "reference": "direct PretrainBundleModel.logits() greedy decode (bundle oracle)",
            "workload": {
                "num_requests": effective_num_requests,
                "max_new_tokens": effective_max_new_tokens,
                "warmup": effective_warmup,
                "iters": effective_iters,
                "prompt_pool": "headline" if headline else "default",
            },
            "vllm": vllm_res["config"],
            "num_blocks": main_res["num_blocks"],
            "decode_graphs": main_res["decode_graphs"],
            "repro_command": f"modal run scripts/modal_esme_three_way.py --command {command}",
        },
    }
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-3way-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-3way] wrote {out_path}")
