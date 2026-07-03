"""Modal A100 benchmark for Esme-214M-Chat through the llm-infer Esme backend.

This is intentionally not a vLLM/HF comparison. Esme is loaded from an
``llm_pretrain_dense_v1`` export bundle and serves through real paged KV:
the benchmark compares the paged-KV engine path against full recompute
on one workload. Both systems must match direct
``PretrainBundleModel.logits()`` greedy decode before any tokens/s number is
printed — a system that diverges reports no throughput, only its measured token
count and wall-clock.

The local entrypoint stages the serving bundle files into the repo-scoped Modal
volume ``llm-infer-esme-bundles`` under ``/esme-214m-chat``. Set
``--bundle-path`` or ``ESME_BUNDLE_PATH``; otherwise it tries the standard sibling
checkout export at ``../esme-posttrain/exports/esme-214m-chat``.

    modal run scripts/modal_esme_benchmark.py --command smoke
    modal run scripts/modal_esme_benchmark.py --command bench
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
    REQUIRED_BUNDLE_FILES,
    VOLUME_NAME,
    local_bundle_path,
    stage_bundle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/llm-infer"
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"

BLOCK_SIZE = 128

app = modal.App("llm-infer-esme-benchmark")

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

esme_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .pip_install("torch>=2.2", "transformers>=4.43", "numpy>=1.26")
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
    .workdir(REMOTE_ROOT)
    .run_commands("pip install --no-deps -e .")
)

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _run_comparison(
    bundle_path: Path,
    *,
    device: str,
    dtype_name: str,
    num_requests: int,
    max_new_tokens: int,
    warmup: int,
    iters: int,
    include_samples: bool,
) -> dict:
    """Build the workload, run the paged-vs-recompute comparison, and assemble the record.

    Importable on CPU or GPU — the GPU specifics (CUDA assert, dtype, snapshot) are passed in,
    so the same comparison drives both the Modal A100 run and a local CPU relative check.
    """
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions, normalize_at_eos
    from llm_infer.benchmarks.esme_paged import build_requests, compare_paged_vs_recompute
    from llm_infer.model.runtime import load_model_runtime

    dtype = getattr(torch, dtype_name)
    runtime = load_model_runtime("esme", bundle_path=bundle_path, dtype=dtype, device=device)
    manifest = runtime.metadata["manifest"]
    if runtime.backend_id != "esme":
        raise AssertionError(f"expected esme backend, got {runtime.backend_id!r}")
    if runtime.model_id != "esme-214m-chat":
        raise AssertionError(f"expected esme-214m-chat, got {runtime.model_id!r}")
    if runtime.eos_token_ids != frozenset({2}):
        raise AssertionError(f"expected EOS [2], got {sorted(runtime.eos_token_ids)}")
    if not runtime.capabilities.paged_kv:
        raise AssertionError("Esme runtime must advertise paged_kv for the paged-vs-recompute run")

    requests = build_requests(runtime.tokenizer, num_requests)
    needed_blocks = sum(
        math.ceil((len(req.prompt_ids) + max_new_tokens) / BLOCK_SIZE) for req in requests
    )
    num_blocks = needed_blocks + max(4, len(requests))

    timings = compare_paged_vs_recompute(
        runtime,
        requests,
        max_new_tokens=max_new_tokens,
        block_size=BLOCK_SIZE,
        num_blocks=num_blocks,
        warmup=warmup,
        iters=iters,
        device=device,
    )
    diverged = [t.system for t in timings if not t.matches_reference]
    if diverged:
        raise AssertionError(f"systems diverged from direct bundle logits: {diverged}")

    paged = next(t for t in timings if t.system == "llm_infer_paged")
    recompute = next(t for t in timings if t.system == "full_recompute")
    speedup = (
        paged.tokens_per_second / recompute.tokens_per_second
        if paged.tokens_per_second and recompute.tokens_per_second
        else None
    )

    samples: dict[str, str] = {}
    if include_samples:
        for request_id, token_ids in list(paged.outputs.items())[:2]:
            trimmed = normalize_at_eos(token_ids, runtime.eos_token_ids)
            samples[request_id] = runtime.tokenizer.decode(trimmed, skip_special_tokens=True)

    rows = [
        {
            "model": "Esme-214M-Chat",
            "system": t.system,
            "mode": t.mode,
            "reference": "PretrainBundleModel.logits() greedy_decode",
            "matches_reference": t.matches_reference,
            "median_seconds": t.median_seconds,
            "total_output_tokens": t.total_output_tokens,
            "tokens_per_second": t.tokens_per_second,
            "iters": iters,
        }
        for t in timings
    ]
    return {
        "rows": rows,
        "paged_vs_recompute_speedup": speedup,
        "reference_check": {
            "exact": len(requests),
            "total": len(requests),
            "mismatches": [],
            "reference": "direct PretrainBundleModel.logits() greedy decode",
        },
        "config": {
            "backend": runtime.backend_id,
            "device": device,
            "served_model": {
                "name": "Esme-214M-Chat",
                "model_id": runtime.model_id,
                "bundle_format": runtime.metadata["format"],
                "model_family": manifest.get("model_family"),
                "dtype": str(runtime.model.dtype),
            },
            "capabilities": {
                "paged_kv": runtime.capabilities.paged_kv,
                "prefix_caching": runtime.capabilities.prefix_caching,
                "speculative": runtime.capabilities.speculative,
                "flash_attention": runtime.capabilities.flash_attention,
            },
            "workload": {
                "num_requests": num_requests,
                "max_new_tokens": max_new_tokens,
                "prompt_lengths": [len(req.prompt_ids) for req in requests],
                "warmup": warmup,
                "iters": iters,
                "eos_token_ids": sorted(runtime.eos_token_ids),
                "chat_template": "user\\n...\\nassistant\\n",
            },
            "bundle": {
                "volume": VOLUME_NAME,
                "remote_path": str(bundle_path),
                "staged_files": list(REQUIRED_BUNDLE_FILES),
            },
            "system_config": {
                "block_size": BLOCK_SIZE,
                "num_blocks": num_blocks,
                "paged": "real paged KV writes/reads, fused batched decode",
                "recompute": "per-request greedy full recompute (PretrainBundleModel.logits)",
            },
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        },
        "samples": samples,
    }


@app.function(
    image=esme_image,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def bench_esme(
    num_requests: int,
    max_new_tokens: int,
    warmup: int,
    iters: int,
    include_samples: bool,
) -> str:
    """Run the Esme paged-KV vs full-recompute comparison on the A100, gated on the reference."""
    import torch

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    record = _run_comparison(
        Path(REMOTE_BUNDLE_PATH),
        device="cuda",
        dtype_name="bfloat16",
        num_requests=num_requests,
        max_new_tokens=max_new_tokens,
        warmup=warmup,
        iters=iters,
        include_samples=include_samples,
    )
    return json.dumps(record)


def _print_table(record: dict) -> None:
    header = (
        "| model | system | mode | reference check | median s | output tok | tok/s |\n"
        "| --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for row in record["rows"]:
        tps = row["tokens_per_second"]
        tps_cell = f"{tps:.1f}" if tps is not None else "—"
        lines.append(
            f"| {row['model']} | {row['system']} | {row['mode']} | "
            f"{'yes' if row['matches_reference'] else 'no'} | "
            f"{row['median_seconds']:.3f} | {row['total_output_tokens']} | "
            f"{tps_cell} |"
        )
    print("\n" + "\n".join(lines) + "\n")
    speedup = record.get("paged_vs_recompute_speedup")
    if speedup is not None:
        print(f"[esme] paged KV vs full recompute: {speedup:.2f}x")


@app.local_entrypoint()
def main(
    command: str = "bench",
    bundle_path: str = "",
    num_requests: int = 0,
    max_new_tokens: int = 0,
    warmup: int = -1,
    iters: int = 0,
) -> None:
    """Stage Esme, run the selected benchmark command, print/write the pinned record."""
    if command == "smoke":
        defaults = {"num_requests": 2, "max_new_tokens": 8, "warmup": 0, "iters": 1}
    elif command == "bench":
        defaults = {"num_requests": 8, "max_new_tokens": 64, "warmup": 1, "iters": 3}
    else:
        raise ValueError(f"command must be 'smoke' or 'bench', got {command!r}")

    effective_num_requests = num_requests or defaults["num_requests"]
    effective_max_new_tokens = max_new_tokens or defaults["max_new_tokens"]
    effective_warmup = defaults["warmup"] if warmup < 0 else warmup
    effective_iters = iters or defaults["iters"]
    local_bundle = local_bundle_path(bundle_path)
    stage_bundle(esme_bundles, local_bundle, label="esme")

    print(
        f"[esme] {command}: Esme-214M-Chat paged-KV vs full recompute, "
        f"{effective_num_requests} requests x {effective_max_new_tokens} new tokens, "
        f"warmup={effective_warmup}, iters={effective_iters}"
    )
    record = json.loads(
        bench_esme.remote(
            effective_num_requests,
            effective_max_new_tokens,
            effective_warmup,
            effective_iters,
            command == "smoke",
        )
    )
    record["config"]["command"] = command
    record["config"]["bundle"]["local_source"] = str(local_bundle)
    record["config"]["repro_command"] = (
        f"modal run scripts/modal_esme_benchmark.py --command {command}"
    )

    _print_table(record)
    if record.get("samples"):
        print("[esme] smoke samples:")
        for request_id, text in record["samples"].items():
            snippet = text[:200].replace("\n", " ")
            print(f"  {request_id}: {snippet}")

    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme] wrote {out_path}")
