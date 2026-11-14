"""Every paid Modal path refuses to start until Adam approves it."""

from __future__ import annotations

import ast

import pytest

from scripts.modal_approval import require_serve_approval, require_spend_approval
from tests.support.modal_source import (
    SCRIPTS,
    assigns_approved,
    decorator_name,
    defaults_to_false,
    first_statement,
    gate_call,
    modal_calls,
    modal_functions,
    modules,
)

GATE_FIX = (
    "End the signature with `approved: bool = False`, make "
    "`require_spend_approval(approved, __file__, ...)` from `scripts/modal_approval.py` the first "
    "statement, and pass `approved=True` from the entrypoint"
)


Modules = list[tuple[str, ast.Module]]


def ungated_modal_functions(mods: Modules) -> list[str]:
    ungated = []
    for name, tree in mods:
        for fn in modal_functions(tree, "paid"):
            first = first_statement(fn)
            gated = defaults_to_false(fn) and gate_call(first) and not assigns_approved(fn)
            if not gated:
                ungated.append(f"{name}:{fn.lineno} {fn.name}")
    return ungated


def calls_outside_raise(statement: ast.stmt) -> bool:
    """Whether a statement calls anything other than an exception it raises."""
    raised = {
        id(node) for r in ast.walk(statement) if isinstance(r, ast.Raise) for node in ast.walk(r)
    }
    return any(isinstance(n, ast.Call) and id(n) not in raised for n in ast.walk(statement))


def ungated_entrypoints(mods: Modules) -> list[str]:
    ungated = []
    for name, tree in mods:
        for fn in modal_functions(tree, "entrypoint"):
            params = [arg.arg for arg in fn.args.args]
            gate_at = next(
                (
                    index
                    for index, statement in enumerate(fn.body)
                    # An entrypoint may gate only its paid branch, so the gate may sit in a block.
                    if any(
                        {kw.arg for kw in call.keywords} == set(params[:-1])
                        for node in ast.walk(statement)
                        if isinstance(node, ast.stmt) and (call := gate_call(node))
                    )
                ),
                None,
            )
            early = gate_at is not None and any(map(calls_outside_raise, fn.body[:gate_at]))
            if not defaults_to_false(fn) or gate_at is None or early or assigns_approved(fn):
                ungated.append(f"{name}:{fn.lineno} {fn.name}")
    return ungated


def unforwarded_modal_calls(mods: Modules) -> list[str]:
    missing = []
    for name, tree in mods:
        paid_names = {fn.name for fn in modal_functions(tree, "paid")}
        for call in modal_calls(tree):
            assert isinstance(call.func, ast.Attribute)
            target = ast.unparse(call.func.value)
            forwarded = any(
                kw.arg == "approved" and ast.unparse(kw.value) == "True" for kw in call.keywords
            )
            if target in paid_names and not forwarded:
                missing.append(f"{name}:{call.lineno} {target}")
    return missing


def approvals_outside_entrypoints(mods: Modules) -> list[str]:
    """`approved=True` anywhere but a local entrypoint, or a module function only it calls."""
    stray = []
    for name, tree in mods:
        entrypoints = modal_functions(tree, "entrypoint")
        called = {
            c.func.id
            for fn in entrypoints
            for c in ast.walk(fn)
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)
        }
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in called
            and not node.decorator_list
            and not any(
                isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == node.name
                for other in tree.body
                if other not in entrypoints and other is not node
                for c in ast.walk(other)
            )
        ]
        inside = {id(node) for fn in [*entrypoints, *helpers] for node in ast.walk(fn)}
        stray += [
            f"{name}:{call.lineno}"
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and id(call) not in inside
            and any(
                kw.arg == "approved" and ast.unparse(kw.value) != "False" for kw in call.keywords
            )
        ]
    return stray


def serve_gate(statement: ast.stmt) -> bool:
    """A bare `require_serve_approval(...)` at module level or directly under `if modal.is_local():`."""
    if isinstance(statement, ast.If) and ast.unparse(statement.test) == "modal.is_local()":
        return any(serve_gate(inner) for inner in statement.body)
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "require_serve_approval"
    )


def ungated_servers(mods: Modules) -> list[str]:
    ungated = []
    for name, tree in mods:
        if not modal_functions(tree, "server"):
            continue
        apps = [n.lineno for n in tree.body if "modal.App(" in ast.unparse(n)]
        gates = [n.lineno for n in tree.body if serve_gate(n)]
        if not gates or not apps or gates[0] > apps[0]:
            ungated.append(name)
    return ungated


def modal_classes(mods: Modules) -> list[str]:
    """Modal classes and sandboxes, which the gate, stall-guard, and launch-plan checks miss."""
    return [
        f"{name}:{node.lineno}"
        for name, tree in mods
        for node in ast.walk(tree)
        if (isinstance(node, ast.ClassDef) and "cls" in map(decorator_name, node.decorator_list))
        or (isinstance(node, ast.Attribute) and node.attr in {"Cls", "Sandbox"})
    ]


def test_modal_functions_refuse_direct_calls_without_approval() -> None:
    ungated = ungated_modal_functions(modules())
    assert not ungated, (
        f"Modal functions callable without approval (CPU time bills too). {GATE_FIX}:\n  "
        + "\n  ".join(ungated)
    )


def test_paid_work_uses_modal_functions_not_classes() -> None:
    found = modal_classes(modules())
    assert not found, (
        "Use `@app.function`, not `@app.cls`, `modal.Cls`, or `modal.Sandbox`: the spend gate, "
        "stall guard, and launch-plan checks cover functions only:\n  " + "\n  ".join(found)
    )


def test_only_a_gated_entrypoint_forwards_approval() -> None:
    stray = approvals_outside_entrypoints(modules())
    assert not stray, (
        "Pass `approved=True` only from a `@app.local_entrypoint()` after its gate, never from "
        "`__main__`, `app.run()`, a helper, or another script:\n  " + "\n  ".join(stray)
    )


def test_local_entrypoints_gate_before_any_modal_call() -> None:
    ungated = ungated_entrypoints(modules())
    assert not ungated, (
        "Modal entrypoints must call `require_spend_approval(approved, __file__, <every other arg "
        "as name=name>)` before any other call (only argument checks that raise may come first), "
        "with `approved: bool = False` last:\n  " + "\n  ".join(ungated)
    )


def test_modal_function_calls_forward_approval() -> None:
    missing = unforwarded_modal_calls(modules())
    assert not missing, (
        f"Modal function calls without `approved=True`. {GATE_FIX}:\n  " + "\n  ".join(missing)
    )


def test_server_apps_refuse_to_start_without_approval() -> None:
    ungated = ungated_servers(modules())
    assert not ungated, (
        "Deployable Modal servers must call `require_serve_approval(...)` under "
        "`if modal.is_local():` before `modal.App(...)`: " + ", ".join(ungated)
    )


BAD_SCRIPT = """
import modal
app = modal.App("x")
require_serve_approval("modal serve x.py")

@app.function(gpu="A100-80GB")
def measure(n: int, approved: bool = False) -> None:
    Path("rows.jsonl").write_text("")
    require_spend_approval(approved, __file__, n=n)

@app.function(gpu="A100-80GB")
@modal.asgi_app()
def web() -> None: ...

@app.function(timeout=60)
async def convert() -> None:
    from scripts.convert_esme_to_hf import convert

@app.cls(gpu="A100-80GB")
class Bench:
    @modal.method()
    def run(self) -> None: ...

@app.local_entrypoint()
def main(n: int = 1, approved: bool = False) -> None:
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    Path("attempts.jsonl").write_text("")
    require_spend_approval(approved, __file__, n=n)
    measure.remote(n)
    convert.remote(approved=True)
"""


def test_each_scan_finds_its_known_violation() -> None:
    bad = [("scripts/bad.py", ast.parse(BAD_SCRIPT))]
    assert ungated_modal_functions(bad) == ["scripts/bad.py:7 measure", "scripts/bad.py:16 convert"]
    assert ungated_entrypoints(bad) == ["scripts/bad.py:25 main"]
    assert unforwarded_modal_calls(bad) == ["scripts/bad.py:30 measure"]
    assert ungated_servers(bad) == ["scripts/bad.py"]
    assert modal_classes(bad) == ["scripts/bad.py:20"]


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


OPEN_GATES = """
@app.function(gpu="A100-80GB")
def opened(approved: bool = True) -> None:
    require_spend_approval(approved, __file__)

@app.function()
def forced(approved: bool = False) -> None:
    require_spend_approval(True, __file__)

@app.local_entrypoint()
def main(approved: bool = True) -> None:
    require_spend_approval(approved, __file__)

@app.function(timeout=60)
@modal.asgi_app()
def web() -> None: ...

def _gate() -> None:
    require_serve_approval("modal serve x.py")

app = modal.App("x")

@app.function()
def branch(n: int, approved: bool = False) -> None:
    if n > 100:
        require_spend_approval(approved, __file__)

@app.local_entrypoint()
def reset(n: int = 1, approved: bool = False) -> None:
    approved = True
    require_spend_approval(approved, __file__, n=n)

with image.imports():
    @app.function()
    def nested(n: int) -> None: ...

if __name__ == "__main__":
    with app.run():
        branch.remote(1, approved=True)
    modal.Sandbox.create("python", app=app)
"""


def test_each_scan_refuses_a_gate_that_cannot_refuse() -> None:
    bad = [("scripts/open.py", ast.parse(OPEN_GATES))]
    assert ungated_modal_functions(bad) == [
        "scripts/open.py:3 opened",
        "scripts/open.py:7 forced",
        "scripts/open.py:24 branch",
        "scripts/open.py:35 nested",
    ]
    assert ungated_entrypoints(bad) == ["scripts/open.py:11 main", "scripts/open.py:29 reset"]
    assert ungated_servers(bad) == ["scripts/open.py"]
    assert approvals_outside_entrypoints(bad) == ["scripts/open.py:39"]
    assert modal_classes(bad) == ["scripts/open.py:40"]
