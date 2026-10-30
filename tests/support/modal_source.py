"""Read the Modal scripts' source so tests can check their spend guards without importing them."""

from __future__ import annotations

import ast
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
# A web server idles between requests by design; its scaledown_window bounds idle GPU time.
SERVER_DECORATORS = {"asgi_app", "wsgi_app", "web_server"}
MODAL_CALLS = {"remote", "remote_gen", "spawn", "map", "starmap", "spawn_map", "for_each"}


def decorator_name(decorator: ast.expr) -> str:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")


def modal_functions(tree: ast.Module, kind: str) -> list[ast.FunctionDef]:
    found = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        names = {decorator_name(d) for d in node.decorator_list}
        gpu = any(
            isinstance(d, ast.Call) and any(kw.arg == "gpu" for kw in d.keywords)
            for d in node.decorator_list
        )
        server = bool(names & SERVER_DECORATORS)
        if (
            (kind == "gpu" and gpu and not server)
            or (kind == "server" and gpu and server)
            or (kind == "entrypoint" and "local_entrypoint" in names)
        ):
            found.append(node)
    return found


def modules() -> list[tuple[str, ast.Module]]:
    return [
        (f"scripts/{path.name}", ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(SCRIPTS.glob("*.py"))
    ]


def calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name
    ]


def modal_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in MODAL_CALLS
        and isinstance(call.func.value, ast.Name)
    ]


def first_statement(function: ast.FunctionDef) -> ast.stmt:
    return function.body[1] if ast.get_docstring(function) else function.body[0]
