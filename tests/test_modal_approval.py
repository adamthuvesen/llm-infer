"""Every paid Modal path refuses to start until Adam approves it."""

from __future__ import annotations

import ast

import pytest

from scripts.modal_approval import require_serve_approval, require_spend_approval
from tests.support.modal_source import (
    SCRIPTS,
    calls,
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


def test_gpu_functions_refuse_direct_calls_without_approval() -> None:
    ungated = []
    for name, tree in modules():
        for fn in modal_functions(tree, "gpu"):
            first = first_statement(fn)
            gated = [arg.arg for arg in fn.args.args][-1:] == ["approved"] and calls(
                first, "require_spend_approval"
            )
            if not gated:
                ungated.append(f"{name}:{fn.lineno} {fn.name}")
    assert not ungated, f"GPU functions callable without approval. {GATE_FIX}:\n  " + "\n  ".join(
        ungated
    )


def test_local_entrypoints_gate_before_any_modal_call() -> None:
    ungated = []
    for name, tree in modules():
        for fn in modal_functions(tree, "entrypoint"):
            params = [arg.arg for arg in fn.args.args]
            gates = [
                call
                for call in calls(fn, "require_spend_approval")
                if {kw.arg for kw in call.keywords} == set(params[:-1])
            ]
            late = [call for call in modal_calls(fn) if gates and call.lineno < gates[0].lineno]
            if params[-1:] != ["approved"] or not gates or late:
                ungated.append(f"{name}:{fn.lineno} {fn.name}")
    assert not ungated, (
        "Modal entrypoints must call `require_spend_approval(approved, __file__, <every other arg "
        "as name=name>)` before any `.remote`/`.spawn`/`.map`, with `approved: bool = False` "
        "last:\n  " + "\n  ".join(ungated)
    )


def test_gpu_function_calls_forward_approval() -> None:
    missing = []
    for name, tree in modules():
        gpu_names = {fn.name for fn in modal_functions(tree, "gpu")}
        for call in modal_calls(tree):
            assert isinstance(call.func, ast.Attribute)
            target = ast.unparse(call.func.value)
            forwarded = any(
                kw.arg == "approved" and ast.unparse(kw.value) == "True" for kw in call.keywords
            )
            if target in gpu_names and not forwarded:
                missing.append(f"{name}:{call.lineno} {target}")
    assert not missing, f"GPU calls without `approved=True`. {GATE_FIX}:\n  " + "\n  ".join(missing)


def test_server_apps_refuse_to_start_without_approval() -> None:
    ungated = []
    for name, tree in modules():
        if not modal_functions(tree, "server"):
            continue
        apps = [n.lineno for n in tree.body if "modal.App(" in ast.unparse(n)]
        gates = [n.lineno for n in tree.body if calls(n, "require_serve_approval")]
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
