"""Modal A100 harness: validate Esme bundle serving against direct bundle logits.

Esme loads from ``llm_pretrain_dense_v1`` export bundles and serves through the engine's paged
KV path. This check keeps that contract explicit: the engine output must match direct
``PretrainBundleModel.logits()`` full-recompute greedy decode before any Esme benchmark number is
useful.

The local entrypoint stages the serving bundle files into the repo-scoped Modal volume
``llm-infer-esme-bundles`` under ``/esme-214m-chat``. Set ``--bundle-path`` or
``ESME_BUNDLE_PATH``; otherwise it tries the standard sibling checkout export at
``../esme-posttrain/exports/esme-214m-chat``.

    modal run scripts/modal_esme_reference_check.py --command smoke
    modal run scripts/modal_esme_reference_check.py --command check
"""

from __future__ import annotations

from pathlib import Path

import modal

from scripts import modal_stall_guard as stall_guard
from scripts.modal_approval import require_spend_approval
from scripts.modal_esme_bundle import (
    ESME_BUNDLE_MOUNT,
    REMOTE_BUNDLE_PATH,
    VOLUME_NAME,
    check_requests,
    check_size,
    engine_outputs,
    local_bundle_path,
    stage_bundle,
)
from scripts.modal_image_pins import UV_VERSION, locked

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/llm-infer"
LOCAL_PYTHON = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"

app = modal.App("llm-infer-esme-reference-check")

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

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python=LOCAL_PYTHON)
    # The locked torch, as in pyproject.toml and the flash image; --no-deps below cannot correct it.
    .uv_pip_install(
        f"uv=={UV_VERSION}", *locked("torch", "transformers", "numpy"), uv_version=UV_VERSION
    )
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
    .workdir(REMOTE_ROOT)
    .run_commands("uv pip install --system --no-deps -e .")
)

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=30 * 60,
)
def check_esme(num_requests: int, max_new_tokens: int, approved: bool = False) -> str:
    """Fail if engine generation diverges from direct bundle-logits greedy decode."""
    require_spend_approval(approved, __file__)
    import torch

    from llm_infer.benchmarks import normalize_at_eos
    from llm_infer.model.decode import greedy_decode
    from llm_infer.model.runtime import load_model_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    stall_guard.arm()
    runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    if runtime.backend_id != "esme":
        raise AssertionError(f"expected esme backend, got {runtime.backend_id!r}")
    if runtime.model_id != "esme-214m-chat":
        raise AssertionError(f"expected esme-214m-chat, got {runtime.model_id!r}")
    if runtime.eos_token_ids != frozenset({2}):
        raise AssertionError(f"expected EOS [2], got {sorted(runtime.eos_token_ids)}")

    requests = check_requests(runtime, num_requests)
    reference = {
        request_id: greedy_decode(
            runtime.model,
            list(prompt_ids),
            max_new_tokens=max_new_tokens,
            eos_token_ids=set(runtime.eos_token_ids),
        )
        for request_id, prompt_ids in requests
    }
    stall_guard.progress()
    outputs = engine_outputs(runtime, requests, max_new_tokens)
    missing = sorted(set(reference) - set(outputs))
    extra = sorted(set(outputs) - set(reference))
    mismatches = [{"request": request_id, "detail": "missing output"} for request_id in missing]
    mismatches.extend(
        {"request": request_id, "detail": "unexpected output"} for request_id in extra
    )
    for request_id in reference:
        if request_id not in outputs:
            continue
        if normalize_at_eos(outputs[request_id], runtime.eos_token_ids) != normalize_at_eos(
            reference[request_id], runtime.eos_token_ids
        ):
            mismatches.append({"request": request_id, "detail": "tokens diverged"})
    if mismatches:
        raise AssertionError(f"Esme engine diverged from direct bundle logits: {mismatches[:3]}")
    return (
        f"Esme reference check PASSED: {len(requests)}/{len(requests)} exact against "
        "direct PretrainBundleModel.logits() greedy decode"
    )


@app.local_entrypoint()
def main(command: str = "smoke", bundle_path: str = "", approved: bool = False) -> None:
    """Run the cheap smoke or fuller Esme reference check on the A100.

    smoke:   modal run scripts/modal_esme_reference_check.py --command smoke
    check:   modal run scripts/modal_esme_reference_check.py --command check
    """
    require_spend_approval(approved, __file__, command=command, bundle_path=bundle_path)
    num_requests, max_new_tokens = check_size(command)
    stage_bundle(esme_bundles, local_bundle_path(bundle_path), label="esme-reference")
    print(check_esme.remote(num_requests, max_new_tokens, approved=True))
