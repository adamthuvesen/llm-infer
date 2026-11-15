"""Refuse a paid Modal run unless Adam approved it.

Every ``local_entrypoint`` and every Modal function takes ``approved: bool = False`` and calls
``require_spend_approval`` first. Entrypoints forward ``approved=True`` to the Modal functions, so
``modal run <script>::<function>`` still refuses inside the container. The deployable server has
no entrypoint, so ``require_serve_approval`` gates it on an environment variable at import time.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

LAUNCH_APPROVAL_FLAG = "--approved"
SERVE_APPROVAL_ENV = "LLM_INFER_SERVE_APPROVED"


def _refusal(command: str) -> SystemExit:
    return SystemExit(
        "Refusing to start: this run spends money on Modal GPUs.\n"
        "Agents: stop here and ask Adam to approve this exact command:\n"
        f"  {command}"
    )


def approved_command(script: str, args: dict[str, object]) -> str:
    # Entry scripts all live in scripts/; a Modal container may mount them outside the repo root.
    parts = ["modal", "run", f"scripts/{Path(script).name}"]
    for name, value in args.items():
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            parts.append(flag if value else "--no-" + flag.removeprefix("--"))
        else:
            parts += [flag, shlex.quote(str(value))]
    return " ".join([*parts, LAUNCH_APPROVAL_FLAG])


def require_spend_approval(approved: bool, script: str, /, **args: object) -> None:
    """Exit before any spend unless the run carries ``--approved``; pass every entrypoint arg."""
    if not approved:
        raise _refusal(approved_command(script, args))


def require_serve_approval(command: str) -> None:
    """Exit unless ``LLM_INFER_SERVE_APPROVED=1`` is set for ``modal serve`` or ``modal deploy``."""
    if os.environ.get(SERVE_APPROVAL_ENV) != "1":
        raise _refusal(f"{SERVE_APPROVAL_ENV}=1 {command}")
