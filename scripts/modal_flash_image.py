"""One shared Modal flash-attn image for every GPU harness.

Both serving tracks need the same GPU image: a CUDA base with torch + flash-attn + transformers.
Defining it once — instead of copy-pasting the recipe into each harness — means there is exactly
one image definition, baked once and cached, then reused by all of them. Any divergence between
harnesses (a different torch/transformers pin, a reordered layer) silently invalidates the cache.

flash-attn installs from a prebuilt wheel: torch 2.8.0 on CUDA 12.8 plus the matching
``flash-attn==2.8.3.post1`` wheel by direct URL from the GitHub release. Nothing CUDA compiles
at build time, so a CUDA runtime base is enough (no ``-devel``). ``build-essential`` ships a
host C toolchain anyway: torch.compile's Triton backend builds its kernel launcher stubs with
``cc`` at runtime, and without one Inductor fails with "Failed to find C compiler".

flash-attn ships two wheels per release — one per torch C++ ABI (``cxx11abiTRUE`` /
``cxx11abiFALSE``). Installing the wrong one imports but fails at the first kernel call. So the ABI
is **selected from torch's actual** ``torch.compiled_with_cxx11_abi()`` at build time (not guessed),
and a guard then imports flash-attn and re-checks torch/ABI agreement — a wrong pick fails the build
loudly instead of producing a silently broken image.

``transformers>=4.51`` (Qwen3 support, needed by the Esme HF/vLLM checkpoint) sits after the
flash-attn layer and satisfies the Qwen gates' old ``>=4.43`` too, so one pin serves both tracks.
"""

from __future__ import annotations

from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/llm-infer"
# CUDA 12.8 RUNTIME base — nothing compiles (flash-attn is a prebuilt wheel), so no devel toolkit
# is needed. 12.8 matches the torch 2.8.0 cu128 build below and the flash-attn cu12 wheel.
CUDA_IMAGE = "nvidia/cuda:12.8.1-runtime-ubuntu22.04"

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
    """The shared flash-attn A100 image: torch -> flash-attn wheel (ABI-matched) -> transformers.

    No source compile: flash-attn is a prebuilt wheel installed by URL, with an ABI guard. The
    ``add_local_dir`` + editable install are the cheap tail that relinks when engine source
    changes; the layers above are keyed only on the pinned versions, so they stay a cache hit
    across every harness and across ordinary source edits.
    """
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
        # build-essential: host cc for Triton's runtime launcher builds (torch.compile).
        .apt_install("git", "build-essential")
        # torch first — flash-attn's wheel is built against this exact torch + CUDA.
        .pip_install(f"torch=={TORCH_VERSION}", extra_options=f"--index-url {TORCH_CUDA_INDEX}")
        # flash-attn prebuilt wheel, ABI chosen from the installed torch, then guarded (two RUN
        # steps — no heredocs, which Modal's Dockerfile parser rejects).
        .run_commands(_INSTALL_FLASH_ATTN, _GUARD_FLASH_ATTN)
        # Everything that changes more often than torch comes AFTER the flash-attn layer.
        .pip_install("transformers>=4.51", "numpy>=1.26", "pytest>=8.0")
        .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=IGNORE)
        .workdir(REMOTE_ROOT)
        .run_commands("pip install --no-deps -e .")
    )


# A single module-level instance so every harness imports the *same* image object/definition.
FLASH_IMAGE = build_flash_image()
