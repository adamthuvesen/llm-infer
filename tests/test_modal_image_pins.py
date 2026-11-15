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
PIP = re.compile(r"(?<!uv )\bpip3?(?:\.\d+)? install\b")
OTHER_INSTALLERS = re.compile(
    r"\b(?:(?:uv tool|pipx|conda|mamba|micromamba|poetry) install|uv pip sync|uv add|uv run --with)\b"
)
ALLOWED_INSTALL_METHODS = {"uv_pip_install", "apt_install"}
# `uv pip install` flags whose next token is a path or an interpreter, not a package.
UV_FLAGS_WITH_VALUE = {
    "-e",
    "--editable",
    "--python",
    "-p",
    "-r",
    "--requirement",
    "-c",
    "--index-url",
    "--extra-index-url",
    "-f",
    "--find-links",
    "--target",
}


def unpinned(arg: ast.expr) -> bool:
    """Whether a package argument is anything but an exact pin or `*locked(...)`.

    A variable can hold a floating version the scan cannot read, so packages are written inline.
    """
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return not EXACT.match(arg.value)
    if isinstance(arg, ast.JoinedStr):
        parts = [part.value for part in arg.values if isinstance(part, ast.Constant)]
        return "==" not in "".join(part for part in parts if isinstance(part, str))
    return not (
        isinstance(arg, ast.Starred)
        and isinstance(arg.value, ast.Call)
        and isinstance(arg.value.func, ast.Name)
        and arg.value.func.id == "locked"
    )


def package_args(call: ast.Call) -> list[ast.expr]:
    """The call's package arguments, with list and tuple literals opened up."""
    args: list[ast.expr] = []
    for arg in call.args:
        args += arg.elts if isinstance(arg, ast.List | ast.Tuple) else [arg]
    return args


def unpinned_uv_installs(command: str) -> list[str]:
    """What a literal `uv pip install` command installs without an exact pin."""
    found: list[str] = []
    for install in re.split(r"\buv pip install\b", command)[1:]:
        tokens = re.split(r"\s+", install.split("&&")[0].strip())
        if {"-r", "--requirement"} & set(tokens) or any(
            t.startswith("--requirement=") for t in tokens
        ):
            found.append("a requirements file")
        editable = {"-e", "--editable", "."} & set(tokens) or any(
            t.startswith("--editable=") for t in tokens
        )
        if editable and "--no-deps" not in tokens:
            found.append("a project without --no-deps, whose dependencies float")
        skip_next = False
        for token in tokens:
            if skip_next or not token:
                skip_next = False
                continue
            if token in UV_FLAGS_WITH_VALUE:
                skip_next = True
            elif not token.startswith("-") and not EXACT.match(token.strip("\"'")):
                found.append(token)
    return found


def install_problems(source: str, label: str) -> list[str]:
    problems: list[str] = []
    tree = ast.parse(source)
    # An f-string's literal parts end mid-command; its interpolated packages are pinned in code.
    fragments = {
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for part in node.values
    }
    # Docstrings tell a person what to run locally; they build no image.
    docs = {id(node.value) for node in ast.walk(tree) if isinstance(node, ast.Expr)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method.startswith("pip_install") or (
                "install" in method and method not in ALLOWED_INSTALL_METHODS
            ):
                problems.append(f"{label}:{node.lineno} uses {method}; use uv_pip_install")
            if node.func.attr == "uv_pip_install":
                if any(kw.arg == "requirements" for kw in node.keywords):
                    problems.append(
                        f"{label}:{node.lineno} requirements= installs floating versions; "
                        "list each package inline"
                    )
                problems += [
                    f"{label}:{arg.lineno} a package has no exact version; write `name==version` or "
                    '`*locked("name")` from scripts/modal_image_pins.py'
                    for arg in package_args(node)
                    if unpinned(arg)
                ]
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docs:
                continue
            if PIP.search(node.value) or OTHER_INSTALLERS.search(node.value):
                problems.append(
                    f"{label}:{node.lineno} runs pip or another installer; run `uv pip install` instead"
                )
            if id(node) not in fragments:
                problems += [
                    f"{label}:{node.lineno} `uv pip install` installs {what} without an exact pin"
                    for what in unpinned_uv_installs(node.value)
                ]
    return problems


def test_no_modal_image_installs_a_floating_version_or_uses_pip() -> None:
    problems: list[str] = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        problems += install_problems(path.read_text(encoding="utf-8"), f"scripts/{path.name}")
    assert not problems, "Modal images must install exact pins with uv:\n  " + "\n  ".join(problems)


@pytest.mark.parametrize(
    "source",
    [
        'image.pip_install("vllm")',
        'image.uv_pip_install("vllm")',
        'image.uv_pip_install("transformers>=4.43")',
        'image.uv_pip_install(f"vllm>={V}")',
        'image.run_commands("pip install --no-deps -e .")',
        'image.run_commands(f"{PY} -m pip install x==1")',
        'image.run_commands("pip3 install vllm==1")',
        'image.uv_pip_install(["vllm", "transformers==4.43"])',
        'image.run_commands("uv pip install --system vllm")',
        'PKGS = ["vllm"]\nimage.uv_pip_install(*PKGS)',
        "image.uv_pip_install(PKG)",
        'image.pip_install_from_requirements("requirements.txt")',
        'image.pip_install_from_pyproject("pyproject.toml")',
        'image.uv_pip_install(requirements=["requirements.txt"])',
        'image.run_commands("uv pip install --system -r requirements.txt")',
        'image.run_commands("uv pip install --system -e .")',
        'image.run_commands("uv tool install vllm")',
        'image.run_commands("pipx install vllm")',
        'image.micromamba_install("vllm")',
        'image.poetry_install_from_file("pyproject.toml")',
        'image.run_commands("uv pip sync --system requirements.txt")',
        'image.run_commands("poetry install")',
        'image.run_commands("uv pip install --system --editable=.")',
    ],
)
def test_the_scan_refuses_each_floating_or_pip_form(source: str) -> None:
    assert install_problems(source, "x.py")


def test_the_scan_accepts_exact_uv_installs() -> None:
    source = 'image.uv_pip_install("uv==0.13.5", f"vllm=={V}", *locked("torch"))\nimage.run_commands("uv pip install --system --no-deps -e .")'
    assert install_problems(source, "x.py") == []


def test_the_scan_accepts_a_pinned_install_from_an_index() -> None:
    source = (
        'image.run_commands("uv pip install --system torch==2.9.0 --index-url https://x.test/whl")'
    )
    assert install_problems(source, "x.py") == []


def test_locked_returns_the_uv_lock_version() -> None:
    assert modal_image_pins.locked("torch") == [
        f"torch=={modal_image_pins.locked_version('torch')}"
    ]


def test_a_package_missing_from_the_lock_stops_the_image_definition() -> None:
    with pytest.raises(SystemExit, match="not-a-real-package"):
        modal_image_pins.locked("not-a-real-package")
