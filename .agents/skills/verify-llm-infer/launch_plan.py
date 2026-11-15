"""Import every Modal script with the real modal package and try each paid path unapproved.

Run by `verify launch-plan` from the repo root. Prints one JSON object. Nothing dispatches:
local entrypoints and Modal functions, CPU-only ones included, must refuse, and the deployable
server must refuse to import.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

import modal

HELPERS = {"modal_approval", "modal_stall_guard", "modal_flash_image"}
MODAL_BIN = "/usr/local/bin/modal"


def outcome(call: Callable[[], object]) -> str:
    try:
        result = call()
        if inspect.isgenerator(result):
            next(result)  # a generator function's body, and so its gate, runs on first next()
    except SystemExit as exc:
        return str(exc)
    except Exception as exc:  # the drive reports a crash as the outcome text
        return f"CRASH {type(exc).__name__}: {exc}"
    return "RAN"


def placeholders(function: modal.Function[..., object, object]) -> dict[str, None]:
    # `modal run script::fn` makes the caller pass required arguments, so fill them.
    params = inspect.signature(function.get_raw_f()).parameters.values()
    return {p.name: None for p in params if p.default is inspect.Parameter.empty}


warnings.simplefilter("ignore")
plan: dict[str, Mapping[str, object]] = {}
for path in sorted(Path("scripts").glob("modal_*.py")):
    if path.stem in HELPERS:
        continue
    sys.argv = [MODAL_BIN, "serve", str(path)]
    imported = outcome(lambda path=path: importlib.import_module(f"scripts.{path.stem}"))
    if imported != "RAN":
        plan[path.stem] = {"import": imported}
        continue
    module = sys.modules[f"scripts.{path.stem}"]
    entry: dict[str, list[object]] = {}
    for name, obj in vars(module).items():
        sys.argv = [MODAL_BIN, "run", f"{path}::{name}"]
        if isinstance(obj, modal.app.LocalEntrypoint):
            entry[name] = ["entrypoint", None, outcome(obj)]
        elif isinstance(obj, modal.Function):
            result = outcome(lambda obj=obj: obj.local(**placeholders(obj)))
            entry[name] = ["function", obj.spec.gpus, result]
    plan[path.stem] = entry
print(json.dumps(plan))
