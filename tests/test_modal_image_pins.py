"""Modal images install exact versions through uv, never floating versions through pip.

A floating version makes each image build pull whatever is newest, so a paid GPU run cannot
be reproduced, and it can drift from the stack the CPU tests ran on.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from scripts import modal_image_pins

ROOT = Path(__file__).resolve().parents[1]
EXACT = re.compile(r"^[A-Za-z0-9_.\-\[\]]+==[^=<>~!*\s]+$")


def unpinned(arg: ast.expr) -> bool:
    """Whether a package argument is a string literal or f-string without an exact pin."""
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return not EXACT.match(arg.value)
    if isinstance(arg, ast.JoinedStr):
        parts = [part.value for part in arg.values if isinstance(part, ast.Constant)]
        return "==" not in "".join(part for part in parts if isinstance(part, str))
    return False


def install_problems(source: str, label: str) -> list[str]:
    problems: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "pip_install":
                problems.append(f"{label}:{node.lineno} uses pip_install; use uv_pip_install")
            if node.func.attr == "uv_pip_install":
                problems += [
                    f"{label}:{arg.lineno} a package is not pinned with =="
                    for arg in node.args
                    if unpinned(arg)
                ]
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.search(r"(?<!uv )\bpip install\b", node.value)
        ):
            problems.append(f"{label}:{node.lineno} runs pip; run `uv pip install` instead")
    return problems


def test_no_modal_image_installs_a_floating_version_or_uses_pip() -> None:
    problems: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        problems += install_problems(path.read_text(encoding="utf-8"), path.name)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "source",
    [
        'image.pip_install("vllm")',
        'image.uv_pip_install("vllm")',
        'image.uv_pip_install("transformers>=4.43")',
        'image.uv_pip_install(f"vllm>={V}")',
        'image.run_commands("pip install --no-deps -e .")',
        'image.run_commands(f"{PY} -m pip install x==1")',
    ],
)
def test_the_scan_refuses_each_floating_or_pip_form(source: str) -> None:
    assert install_problems(source, "x.py")


def test_the_scan_accepts_exact_uv_installs() -> None:
    source = 'image.uv_pip_install("uv==0.13.5", f"vllm=={V}", *locked("torch"))\nimage.run_commands("uv pip install --system --no-deps -e .")'
    assert install_problems(source, "x.py") == []


def test_locked_returns_the_uv_lock_version() -> None:
    assert modal_image_pins.locked("torch") == [
        f"torch=={modal_image_pins.locked_version('torch')}"
    ]


def test_a_package_missing_from_the_lock_stops_the_image_definition() -> None:
    with pytest.raises(SystemExit, match="not-a-real-package"):
        modal_image_pins.locked("not-a-real-package")
