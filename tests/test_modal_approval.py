"""Every paid Modal path refuses to start until Adam approves it."""

from __future__ import annotations

import ast

import pytest

from scripts.modal_approval import require_serve_approval, require_spend_approval
from tests.support.modal_source import (
    SCRIPTS,
    calls,
    decorator_name,
    first_statement,
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
            gated = [arg.arg for arg in fn.args.args][-1:] == ["approved"] and calls(
                first, "require_spend_approval"
            )
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
                    if any(
                        {kw.arg for kw in call.keywords} == set(params[:-1])
                        for call in calls(statement, "require_spend_approval")
                    )
                ),
                None,
            )
            early = gate_at is not None and any(map(calls_outside_raise, fn.body[:gate_at]))
            if params[-1:] != ["approved"] or gate_at is None or early:
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


def ungated_servers(mods: Modules) -> list[str]:
    ungated = []
    for name, tree in mods:
        if not modal_functions(tree, "server"):
            continue
        apps = [n.lineno for n in tree.body if "modal.App(" in ast.unparse(n)]
        gates = [n.lineno for n in tree.body if calls(n, "require_serve_approval")]
        if not gates or not apps or gates[0] > apps[0]:
            ungated.append(name)
    return ungated


def modal_classes(mods: Modules) -> list[str]:
    """Modal classes, which the gate, stall-guard, and launch-plan checks do not cover."""
    return [
        f"{name}:{node.lineno}"
        for name, tree in mods
        for node in ast.walk(tree)
        if (isinstance(node, ast.ClassDef) and "cls" in map(decorator_name, node.decorator_list))
        or (isinstance(node, ast.Attribute) and node.attr == "Cls")
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
        "Use `@app.function`, not `@app.cls` or `modal.Cls`: the spend gate, stall guard, and "
        "launch-plan checks cover functions only:\n  " + "\n  ".join(found)
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
