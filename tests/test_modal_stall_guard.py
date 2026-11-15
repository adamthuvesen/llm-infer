"""Every Modal GPU function arms the stall guard, so a hung run fails before its timeout bills."""

from __future__ import annotations

import ast

from tests.support.modal_source import modal_functions, modules


def unguarded_gpu_functions(mods: list[tuple[str, ast.Module]]) -> list[str]:
    return [
        f"{name}:{fn.lineno} {fn.name}"
        for name, tree in mods
        for fn in modal_functions(tree, "gpu")
        if not any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "arm"
            and ast.unparse(call.func.value) == "stall_guard"
            for call in ast.walk(fn)
            if isinstance(call, ast.Call)
        )
    ]


def test_gpu_functions_arm_the_stall_guard() -> None:
    unguarded = unguarded_gpu_functions(modules())
    assert not unguarded, (
        "Modal GPU functions without a stall guard. Add `stall_guard.arm()` at the top of each "
        "(`from scripts import modal_stall_guard as stall_guard`) and `stall_guard.progress()` "
        "after each row or batch:\n  " + "\n  ".join(unguarded)
    )


def test_the_scan_finds_a_gpu_function_without_the_guard() -> None:
    source = '@app.function(gpu="A100-80GB")\ndef measure() -> None:\n    other.arm()\n'
    assert unguarded_gpu_functions([("scripts/bad.py", ast.parse(source))]) == [
        "scripts/bad.py:2 measure"
    ]
