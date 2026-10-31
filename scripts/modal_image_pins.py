"""Exact package versions for the Modal images.

Packages in uv.lock install at their locked version, so a GPU image runs the stack the CPU
tests ran. vLLM is not in uv.lock, because it brings its own torch and CUDA stack.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UV_VERSION = "0.13.5"
# The version in the committed assets/esme-vllm-same-host*.json records, so reruns stay comparable.
VLLM_VERSION = "0.24.0"


def locked_version(name: str) -> str:
    """Return the one version of `name` in uv.lock."""
    packages = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))["package"]
    versions = {package["version"] for package in packages if package["name"] == name}
    if len(versions) != 1:
        raise SystemExit(
            f"modal image pins: {name} must appear in uv.lock exactly once, found {sorted(versions)}. "
            "Fix: add the package to pyproject.toml and run `uv lock`."
        )
    return versions.pop()


def locked(*names: str) -> list[str]:
    """Return `name==version` for each package, at its version in uv.lock."""
    return [f"{name}=={locked_version(name)}" for name in names]
