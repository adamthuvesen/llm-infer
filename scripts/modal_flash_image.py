"""One shared Modal GPU image for every flash attention harness.

The GPU harnesses use the same image: a CUDA base with torch + flash-attn + FlashInfer.
Defining it once — instead of copy-pasting the recipe into each harness — means there is exactly
one image definition, baked once and cached, then reused by all of them. Any divergence between
harnesses (a different torch/transformers pin, a reordered layer) silently invalidates the cache.

flash-attn installs from a prebuilt wheel: torch 2.8.0 on CUDA 12.8 plus the matching
``flash-attn==2.8.3.post1`` wheel by direct URL from the GitHub release. FlashInfer is the
default Esme CUDA decode backend and can JIT a shape-specific paged decode op, so this shared
image uses CUDA ``-devel`` and carries ``nvcc``.

flash-attn ships two wheels per release — one per torch C++ ABI (``cxx11abiTRUE`` /
``cxx11abiFALSE``). Installing the wrong one imports but fails at the first kernel call. So the ABI
is **selected from torch's actual** ``torch.compiled_with_cxx11_abi()`` at build time (not guessed),
and a guard then imports flash-attn and re-checks torch/ABI agreement — a wrong pick fails the build
loudly instead of producing a silently broken image.

``transformers>=4.51`` (Qwen3 support, needed by the Esme HF/vLLM checkpoint) sits after the
flash-attn layer and satisfies the project's ``>=4.43`` lower bound too. The image also carries
the lightweight serving deps so the same blessed GPU image can run `python -m llm_infer.serve`
and the server startup smoke.
"""

from __future__ import annotations

from pathlib import Path

import modal

from scripts.modal_image_pins import UV_VERSION, VLLM_VERSION, locked, locked_version

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/llm-infer"
# Images mount the whole repo, so .python-version is present in the container too.
LOCAL_PYTHON = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()
_WHEEL_TAG = "cp" + LOCAL_PYTHON.replace(".", "")
# CUDA devel is intentional: FlashInfer's paged decode wrapper may JIT through nvcc at startup.
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
FLASHINFER_VERSION = locked_version("flashinfer-python")

# Pinned stack that ships a prebuilt flash-attn wheel (no source build):
#   torch 2.8.0 (cu128) + flash-attn 2.8.3.post1 (cu12torch2.8).
TORCH_VERSION = locked_version("torch")
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
FLASH_ATTN_VERSION = "2.8.3.post1"
FLASH_ATTN_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v{ver}/flash_attn-{ver}+cu12torch2.8cxx11abi{abi}-{tag}-{tag}-linux_x86_64.whl"
)

IGNORE = [
    "**/.git",
    "**/.venv",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
    "bench-results/**",
]

# Build commands run as Dockerfile RUN steps; Modal's Dockerfile parser does NOT accept bash
# heredocs, so multi-statement Python must use `python -c "a; b; c"`, never `python - <<EOF`.
#
# Step 1: pick the wheel whose ABI matches the installed torch and install it — ABI selection and
# install stay in ONE run-string so the shell var ``$ABI`` persists. The URL's ``{abi}`` is left as
# the literal shell ``${{ABI}}`` (doubled braces escape the f-string), resolved by the shell from
# torch's real ``compiled_with_cxx11_abi()``, never guessed.
_FLASH_WHEEL_URL_SHELL = FLASH_ATTN_WHEEL_URL.format(
    ver=FLASH_ATTN_VERSION, abi="${ABI}", tag=_WHEEL_TAG
)
_DETECT_ABI = (
    "python -c \"import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')\""
)
_INSTALL_FLASH_ATTN = (
    f'ABI=$({_DETECT_ABI}) && echo "flash-attn wheel ABI=$ABI" '
    f'&& uv pip install --system --no-cache "{_FLASH_WHEEL_URL_SHELL}"'
)
# Step 2: guard, as its own one-line command. Importing flash_attn's CUDA extension is the real ABI
# check — a wrong-ABI wheel imports the Python package but raises ``undefined symbol`` here, so a
# bad pick fails the build, not a GPU run. Version asserts catch a wrong wheel/torch outright.
_GUARD_FLASH_ATTN = (
    'python -c "import flash_attn, torch, flash_attn_2_cuda; '
    f"assert flash_attn.__version__ == '{FLASH_ATTN_VERSION}', flash_attn.__version__; "
    f"assert torch.__version__.startswith('{TORCH_VERSION}'), torch.__version__\""
)


def build_flash_image() -> modal.Image:
    """The shared A100 image for the default FlashInfer fast path.

    No source compile: flash-attn is a prebuilt wheel installed by URL, with an ABI guard. The
    ``add_local_dir`` + editable install are the cheap tail that relinks when engine source
    changes; the layers above are keyed only on the pinned versions, so they stay a cache hit
    across every harness and across ordinary source edits.
    """
    return _finish_image(
        _build_flash_base().uv_pip_install(
            f"flashinfer-python=={FLASHINFER_VERSION}", uv_version=UV_VERSION
        )
    )


def _build_flash_base(*, cuda_image: str = CUDA_IMAGE) -> modal.Image:
    return (
        modal.Image.from_registry(cuda_image, add_python=LOCAL_PYTHON)
        .apt_install("git", "build-essential")
        # uv itself is a pinned package, so later RUN steps can call `uv pip install`.
        .uv_pip_install(f"uv=={UV_VERSION}", uv_version=UV_VERSION)
        .uv_pip_install(
            f"torch=={TORCH_VERSION}", index_url=TORCH_CUDA_INDEX, uv_version=UV_VERSION
        )
        .run_commands(_INSTALL_FLASH_ATTN, _GUARD_FLASH_ATTN)
        .uv_pip_install(
            *locked("transformers", "numpy", "pytest", "fastapi", "uvicorn", "httpx"),
            uv_version=UV_VERSION,
        )
    )


def _finish_image(image: modal.Image) -> modal.Image:
    return (
        image.add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=IGNORE)
        .workdir(REMOTE_ROOT)
        .run_commands("uv pip install --system --no-deps -e .")
    )


# Module-level baseline instance so every existing harness imports the same image object.
FLASH_IMAGE = build_flash_image()

# vLLM brings its own torch and CUDA stack, so it cannot share the flash image or its container.
# The env pins spawn the engine core and use the native sampler, so no nvcc JIT runs.
VLLM_IMAGE = (
    modal.Image.debian_slim(python_version=LOCAL_PYTHON)
    .uv_pip_install(f"uv=={UV_VERSION}", f"vllm=={VLLM_VERSION}", uv_version=UV_VERSION)
    .env(
        {
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=IGNORE)
    .workdir(REMOTE_ROOT)
    .run_commands("uv pip install --system --no-deps -e .")
)
