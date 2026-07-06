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

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/llm-infer"
# CUDA devel is intentional: FlashInfer's paged decode wrapper may JIT through nvcc at startup.
CUDA_IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
FLASHINFER_VERSION = "0.6.14"

# Pinned stack that ships a prebuilt flash-attn wheel (no source build):
#   torch 2.8.0 (cu128) + flash-attn 2.8.3.post1 (cu12torch2.8).
TORCH_VERSION = "2.8.0"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
FLASH_ATTN_VERSION = "2.8.3.post1"
FLASH_ATTN_WHEEL_URL = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/"
    "v{ver}/flash_attn-{ver}+cu12torch2.8cxx11abi{abi}-cp311-cp311-linux_x86_64.whl"
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
_FLASH_WHEEL_URL_SHELL = FLASH_ATTN_WHEEL_URL.format(ver=FLASH_ATTN_VERSION, abi="${ABI}")
_DETECT_ABI = (
    "python -c \"import torch; print('TRUE' if torch.compiled_with_cxx11_abi() else 'FALSE')\""
)
_INSTALL_FLASH_ATTN = (
    f'ABI=$({_DETECT_ABI}) && echo "flash-attn wheel ABI=$ABI" '
    f'&& pip install --no-cache-dir "{_FLASH_WHEEL_URL_SHELL}"'
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
        _build_flash_base().pip_install(f"flashinfer-python=={FLASHINFER_VERSION}")
    )


def _build_flash_base(*, cuda_image: str = CUDA_IMAGE) -> modal.Image:
    return (
        modal.Image.from_registry(cuda_image, add_python="3.11")
        .apt_install("git", "build-essential")
        .pip_install(f"torch=={TORCH_VERSION}", extra_options=f"--index-url {TORCH_CUDA_INDEX}")
        .run_commands(_INSTALL_FLASH_ATTN, _GUARD_FLASH_ATTN)
        .pip_install(
            "transformers>=4.51",
            "numpy>=1.26",
            "pytest>=8.0",
            "fastapi>=0.110",
            "uvicorn>=0.29",
        )
    )


def _finish_image(image: modal.Image) -> modal.Image:
    return (
        image
        .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=IGNORE)
        .workdir(REMOTE_ROOT)
        .run_commands("pip install --no-deps -e .")
    )


# Module-level baseline instance so every existing harness imports the same image object.
FLASH_IMAGE = build_flash_image()
