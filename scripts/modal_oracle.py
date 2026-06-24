"""Modal A100 harness: validate the flash-attn backend against the golden oracle on GPU.

flash-attn needs a CUDA build, which this project's dev Mac does not have, so the
flash backend is exercised on Modal's A100-80GB (the project's target GPU). The
``torch_naive`` reference and its exact CPU oracle stay the local gate; this only runs
the GPU-only flash path.

    modal run scripts/modal_oracle.py --command smoke    # cheap: auth + import + 1 short decode
    modal run scripts/modal_oracle.py --command oracle    # full: flash vs golden under tie bar

The cheap ``smoke`` runs first by design (a few decode steps on one case) to confirm
auth, the flash-attn import, the weight load on the A100, and the engine loop on GPU
before the (still short) full oracle spends anything more. The HF model cache lives on
a Modal Volume so a model download happens at most once across runs.

Run with the ``modal`` CLI (``uv tool install modal``), not as a project dependency.
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "llm-infer-oracle"
# devel (not runtime) image: flash-attn compiles CUDA kernels at install, needs nvcc.
# The toolkit major must match the CUDA the installed torch was built against, or
# flash-attn's build aborts on a version mismatch — torch 2.12 ships CUDA 13.0, so the
# nvcc here is 13.0 too.
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"
REMOTE_ROOT = "/root/llm-infer"
HF_CACHE = "/hf-cache"

app = modal.App(APP_NAME)

_IGNORE = [
    "**/.git",
    "**/.venv",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
]

# torch first (flash-attn builds against the installed torch), then flash-attn with
# build isolation off so it sees that torch, then the rest of the engine's deps. The
# heavy layer is keyed only on these pins, so editing engine source relinks fast.
# flash-attn's setup.py compiles CUDA kernels at install: with --no-build-isolation it
# uses the current env, so wheel/packaging/setuptools/ninja must already be present
# (ninja parallelizes the nvcc build — without it the compile is far slower).
# build-essential gives nvcc a real host C++ compiler (g++); the CUDA 13 devel image
# otherwise detects only a clang stub (clang++ 0.0.0) and flash-attn's build aborts.
# CC/CXX pin nvcc to gcc/g++ explicitly so it never reaches for the stub clang.
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .apt_install("git", "build-essential")
    .env({"CC": "gcc", "CXX": "g++"})
    .pip_install("torch>=2.2", "transformers>=4.43", "numpy>=1.26", "pytest>=8.0")
    .pip_install("wheel", "packaging", "setuptools", "ninja")
    .pip_install("flash-attn>=2.5", extra_options="--no-build-isolation")
    .env({"HF_HOME": HF_CACHE})
    .add_local_dir(Path(__file__).parent.parent, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
    .workdir(REMOTE_ROOT)
    .run_commands("pip install --no-deps -e .")
)

# Persist the HF model cache so the 3B weights download at most once across runs.
hf_cache = modal.Volume.from_name("llm-infer-hf-cache", create_if_missing=True)


@app.function(image=image, gpu="A100-80GB", volumes={HF_CACHE: hf_cache}, timeout=30 * 60)
def smoke() -> str:
    """Cheap pre-spend check: flash-attn imports, weights load on the A100, engine decodes.

    Decodes only a handful of tokens on one tiny golden case — enough to prove the GPU
    path is wired end to end before the full oracle runs.
    """
    import json

    import torch

    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.model.qwen import QwenModel
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    print(f"device: {torch.cuda.get_device_name(0)}; flash-attn imported OK")

    fixture = json.loads(
        (
            Path(REMOTE_ROOT) / "tests/correctness/goldens/qwen2_5_coder_3b_instruct_cot.json"
        ).read_text(encoding="utf-8")
    )
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
def oracle() -> str:
    """Full flash-attn oracle on the A100: pytest the tie-tolerance suite, fail on any miss.

    Runs ``tests/correctness/test_flash_attn_paged.py`` (the flash path under the
    documented tie bar). Exits non-zero on any failure so ``modal run`` surfaces it.
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
        raise RuntimeError(f"flash oracle FAILED (exit {result.returncode}) — see output above")
    # Surface the last lines (pass summary + any traced ties printed by the test).
    tail = "\n".join(result.stdout.strip().splitlines()[-25:])
    return f"flash oracle PASSED on A100\n{tail}"


@app.local_entrypoint()
def main(command: str = "smoke") -> None:
    """Run the cheap smoke or the full flash oracle on the A100.

    smoke:   modal run scripts/modal_oracle.py --command smoke
    oracle:  modal run scripts/modal_oracle.py --command oracle
    """
    if command == "smoke":
        print(smoke.remote())
    elif command == "oracle":
        print(oracle.remote())
    else:
        raise ValueError(f"command must be 'smoke' or 'oracle', got {command!r}")
