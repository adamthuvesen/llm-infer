"""Every Modal GPU function arms the stall guard, so a hung run fails before its timeout bills."""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
# A web server idles between requests by design; its scaledown_window bounds idle GPU time.
SERVER_DECORATORS = {"asgi_app", "wsgi_app", "web_server"}


def _decorator_name(decorator: ast.expr) -> str:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")


def _requests_gpu(decorator: ast.expr) -> bool:
    return isinstance(decorator, ast.Call) and any(kw.arg == "gpu" for kw in decorator.keywords)


def _arms_guard(function: ast.FunctionDef) -> bool:
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "arm"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "stall_guard"
        for node in ast.walk(function)
    )


def test_gpu_functions_arm_the_stall_guard() -> None:
    unguarded = []
    for path in sorted(SCRIPTS.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.FunctionDef):
                continue
            names = {_decorator_name(decorator) for decorator in node.decorator_list}
            gpu = any(_requests_gpu(decorator) for decorator in node.decorator_list)
            if gpu and not names & SERVER_DECORATORS and not _arms_guard(node):
                unguarded.append(f"scripts/{path.name}:{node.lineno} {node.name}")
    assert not unguarded, (
        "Modal GPU functions without a stall guard. Add `stall_guard.arm()` at the top of each "
        "(`from scripts import modal_stall_guard as stall_guard`) and `stall_guard.progress()` "
        "after each row or batch:\n  " + "\n  ".join(unguarded)
    )
