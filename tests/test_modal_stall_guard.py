"""Modal spend guards: every paid path refuses without approval, and every GPU function arms the
stall guard so a hung run fails before its timeout bills."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.modal_approval import require_serve_approval, require_spend_approval

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
# A web server idles between requests by design; its scaledown_window bounds idle GPU time.
SERVER_DECORATORS = {"asgi_app", "wsgi_app", "web_server"}
MODAL_CALLS = {"remote", "remote_gen", "spawn", "map", "starmap", "spawn_map", "for_each"}
GATE_FIX = (
    "End the signature with `approved: bool = False`, make "
    "`require_spend_approval(approved, __file__, ...)` from `scripts/modal_approval.py` the first "
    "statement, and pass `approved=True` from the entrypoint"
)


def _decorator_name(decorator: ast.expr) -> str:
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    return target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")


def _modal_functions(tree: ast.Module, kind: str) -> list[ast.FunctionDef]:
    found = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        names = {_decorator_name(d) for d in node.decorator_list}
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


def _modules() -> list[tuple[str, ast.Module]]:
    return [
        (f"scripts/{path.name}", ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(SCRIPTS.glob("*.py"))
    ]


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name
    ]


def _modal_calls(node: ast.AST) -> list[ast.Call]:
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr in MODAL_CALLS
        and isinstance(call.func.value, ast.Name)
    ]


def _first_statement(function: ast.FunctionDef) -> ast.stmt:
    return function.body[1] if ast.get_docstring(function) else function.body[0]


def test_gpu_functions_arm_the_stall_guard() -> None:
    unguarded = [
        f"{name}:{fn.lineno} {fn.name}"
        for name, tree in _modules()
        for fn in _modal_functions(tree, "gpu")
        if not any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "arm"
            and ast.unparse(call.func.value) == "stall_guard"
            for call in ast.walk(fn)
            if isinstance(call, ast.Call)
        )
    ]
    assert not unguarded, (
        "Modal GPU functions without a stall guard. Add `stall_guard.arm()` at the top of each "
        "(`from scripts import modal_stall_guard as stall_guard`) and `stall_guard.progress()` "
        "after each row or batch:\n  " + "\n  ".join(unguarded)
    )


def test_gpu_functions_refuse_direct_calls_without_approval() -> None:
    ungated = []
    for name, tree in _modules():
        for fn in _modal_functions(tree, "gpu"):
            first = _first_statement(fn)
            gated = [arg.arg for arg in fn.args.args][-1:] == ["approved"] and _calls(
                first, "require_spend_approval"
            )
            if not gated:
                ungated.append(f"{name}:{fn.lineno} {fn.name}")
    assert not ungated, f"GPU functions callable without approval. {GATE_FIX}:\n  " + "\n  ".join(
        ungated
    )


def test_local_entrypoints_gate_before_any_modal_call() -> None:
    ungated = []
    for name, tree in _modules():
        for fn in _modal_functions(tree, "entrypoint"):
            params = [arg.arg for arg in fn.args.args]
            gates = [
                call
                for call in _calls(fn, "require_spend_approval")
                if {kw.arg for kw in call.keywords} == set(params[:-1])
            ]
            late = [call for call in _modal_calls(fn) if gates and call.lineno < gates[0].lineno]
            if params[-1:] != ["approved"] or not gates or late:
                ungated.append(f"{name}:{fn.lineno} {fn.name}")
    assert not ungated, (
        "Modal entrypoints must call `require_spend_approval(approved, __file__, <every other arg "
        "as name=name>)` before any `.remote`/`.spawn`/`.map`, with `approved: bool = False` "
        "last:\n  " + "\n  ".join(ungated)
    )


def test_gpu_function_calls_forward_approval() -> None:
    missing = []
    for name, tree in _modules():
        gpu_names = {fn.name for fn in _modal_functions(tree, "gpu")}
        for call in _modal_calls(tree):
            target = ast.unparse(call.func.value)
            forwarded = any(
                kw.arg == "approved" and ast.unparse(kw.value) == "True" for kw in call.keywords
            )
            if target in gpu_names and not forwarded:
                missing.append(f"{name}:{call.lineno} {target}")
    assert not missing, f"GPU calls without `approved=True`. {GATE_FIX}:\n  " + "\n  ".join(missing)


def test_server_apps_refuse_to_start_without_approval() -> None:
    ungated = []
    for name, tree in _modules():
        if not _modal_functions(tree, "server"):
            continue
        apps = [n.lineno for n in tree.body if "modal.App(" in ast.unparse(n)]
        gates = [n.lineno for n in tree.body if _calls(n, "require_serve_approval")]
        if not gates or not apps or gates[0] > apps[0]:
            ungated.append(name)
    assert not ungated, (
        "Deployable Modal servers must call `require_serve_approval(...)` under "
        "`if modal.is_local():` before `modal.App(...)`: " + ", ".join(ungated)
    )


def test_spend_gate_refuses_with_the_exact_approved_command() -> None:
    script = str(SCRIPTS / "modal_esme_serving_eval.py")
    with pytest.raises(SystemExit) as refusal:
        require_spend_approval(False, script, batch_sizes="1,8", grouped=False, grouped_ab=True)
    message = str(refusal.value)
    assert "spends money" in message
    assert "ask Adam" in message
    assert message.endswith(
        "modal run scripts/modal_esme_serving_eval.py --batch-sizes 1,8 --no-grouped "
        "--grouped-ab --approved"
    )
    require_spend_approval(True, script, batch_sizes="1,8")


def test_serve_gate_needs_the_approval_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    command = "modal deploy scripts/modal_esme_serve.py"
    monkeypatch.delenv("LLM_INFER_SERVE_APPROVED", raising=False)
    with pytest.raises(SystemExit, match=f"LLM_INFER_SERVE_APPROVED=1 {command}"):
        require_serve_approval(command)
    monkeypatch.setenv("LLM_INFER_SERVE_APPROVED", "1")
    require_serve_approval(command)
