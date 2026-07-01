"""Historical Qwen Modal A100 harness: validate flash-attn against the reference output.

The primary Esme flash gate is ``scripts/modal_esme_flash_reference_check.py``. This
Qwen harness remains for archived public-baseline reproduction. flash-attn needs a CUDA build,
which this project's dev Mac does not have, so the
flash backend is exercised on Modal's A100-80GB (the project's target GPU). The
``torch_naive`` reference and its exact CPU reference check stay the local check;
this only runs
the GPU-only flash path.

    modal run scripts/modal_reference_check.py --command smoke  # cheap import + decode
    modal run scripts/modal_reference_check.py --command check  # full flash check

The cheap ``smoke`` runs first by design (a few decode steps on one case) to confirm
auth, the flash-attn import, the weight load on the A100, and the engine loop on GPU
before the (still short) full reference check spends anything more. The HF model
cache lives on
a Modal Volume so a model download happens at most once across runs.

Run with the ``modal`` CLI (``uv tool install modal``), not as a project dependency.
"""

from __future__ import annotations

import modal

from scripts.modal_flash_image import FLASH_IMAGE, REMOTE_ROOT

APP_NAME = "llm-infer-reference-check"
HF_CACHE = "/hf-cache"

app = modal.App(APP_NAME)

# The one shared flash-attn image (scripts/modal_flash_image.py; flash-attn from a prebuilt wheel),
# with HF_HOME chained on for the model cache. The .env is after the flash-attn layer, so it never
# invalidates it — the Qwen and Esme gates share one image.
image = FLASH_IMAGE.env({"HF_HOME": HF_CACHE})

# Persist the HF model cache so the 3B weights download at most once across runs.
hf_cache = modal.Volume.from_name("llm-infer-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", volumes={HF_CACHE: hf_cache}, timeout=30 * 60)
def smoke() -> str:
    """Cheap pre-spend check: flash-attn imports, weights load on the A100, engine decodes.

    Decodes only a handful of tokens on one tiny golden case — enough to prove the GPU
    path is wired end to end before the full reference check runs.
    """
    import json

    import torch

    from llm_infer.fixtures import QWEN_COT_GOLDEN
    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.model.qwen import QwenModel
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    print(f"device: {torch.cuda.get_device_name(0)}; flash-attn imported OK")

    fixture = json.loads(QWEN_COT_GOLDEN.read_text(encoding="utf-8"))
    case = fixture["cases"][0]
    eos = frozenset(fixture["decoding"]["eos_token_ids"])

    model = QwenModel.load(dtype=torch.bfloat16, backend=FlashAttnPagedAttention(), device="cuda")
    hf_cache.commit()  # persist any freshly downloaded weights
    print("weights loaded on A100 (bf16, flash backend)")

    engine = InferenceEngine(model, block_size=128, num_blocks=8, device="cuda")
    engine.add_request(Request("smoke", list(case["prompt_ids"]), 5, eos))
    out = engine.run()["smoke"]
    golden_prefix = case["continuation_ids"][:5]
    print(f"5-token decode: flash={out} golden_prefix={golden_prefix}")
    return f"smoke OK: {case['case_id']} flash={out} golden={golden_prefix}"


@app.function(image=image, gpu="A100-80GB", volumes={HF_CACHE: hf_cache}, timeout=30 * 60)
def check() -> str:
    """Full flash-attn reference check on the A100; fail on any mismatch.

    Runs ``tests/correctness/test_flash_attn_paged.py`` (the flash path under the
    documented tie rule). Exits non-zero on any failure so ``modal run`` surfaces it.
    """
    import subprocess

    import torch

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    hf_cache.commit()
    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "tests/correctness/test_flash_attn_paged.py",
            "-v",
            "-s",
            "-rA",
        ],
        cwd=REMOTE_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    print(result.stderr)
    hf_cache.commit()
    if result.returncode != 0:
        raise RuntimeError(
            f"flash reference check FAILED (exit {result.returncode}) — see output above"
        )
    # Surface the last lines (pass summary + any traced ties printed by the test).
    tail = "\n".join(result.stdout.strip().splitlines()[-25:])
    return f"flash reference check PASSED on A100\n{tail}"


@app.local_entrypoint()
def main(command: str = "smoke") -> None:
    """Run the cheap smoke or the full flash reference check on the A100.

    smoke:   modal run scripts/modal_reference_check.py --command smoke
    check:   modal run scripts/modal_reference_check.py --command check
    """
    if command == "smoke":
        print(smoke.remote())
    elif command == "check":
        print(check.remote())
    else:
        raise ValueError(f"command must be 'smoke' or 'check', got {command!r}")
