"""Every Modal GPU function arms the stall guard, so a hung run fails before its timeout bills."""

from __future__ import annotations

import ast

from tests.support.modal_source import modal_functions, modules


def test_gpu_functions_arm_the_stall_guard() -> None:
    unguarded = [
        f"{name}:{fn.lineno} {fn.name}"
        for name, tree in modules()
        for fn in modal_functions(tree, "gpu")
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
