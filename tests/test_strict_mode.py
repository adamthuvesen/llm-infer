"""No file turns type checking or a lint ban off with a comment.

basedpyright honors a file-level `# pyright:` comment, ruff honors a `noqa` on a banned API, and
the visualizer's type check honors `@ts-nocheck`. No config setting blocks any of them.
"""

import ast
import re
from pathlib import Path

from tests.support.repo_files import candidate_files, is_python_file

# Codes in `pyproject.toml::tool.ruff.lint` that hold a ban rather than a style choice.
BANNED_RULES = ("TID251",)
PYTHON_SUPPRESSIONS = {
    "file-level `# pyright:` comment changes type checking for the file": re.compile(
        r"^\s*#\s*pyright:"
    ),
    "`# pyright: ignore[rule]` has no reason after it": re.compile(
        r"#\s*pyright:\s*ignore\[[^\]]*\]\s*$"
    ),
    "`noqa` silences a banned API": re.compile(
        rf"#\s*(?:ruff:\s*)?noqa:[^#\n]*\b(?:{'|'.join(BANNED_RULES)})\b", re.IGNORECASE
    ),
}
SCRIPT_SUPPRESSIONS = {
    "`@ts-nocheck` turns the visualizer type check off": re.compile(r"@ts-nocheck\b"),
    "a lint disable with no rule turns every lint off": re.compile(
        r"(?://|/\*)\s*(?:eslint|oxlint)-disable(?:-next-line|-line)?\s*(?:--.*|\*/.*)?$"
    ),
}


def suppressions(path: Path, text: str) -> list[str]:
    if path.suffix in {".js", ".mjs", ".ts"}:
        patterns = SCRIPT_SUPPRESSIONS
    elif is_python_file(path):
        patterns = PYTHON_SUPPRESSIONS
    else:
        return []
    return [
        f"{path.name}:{number} {problem}"
        for number, line in enumerate(text.splitlines(), start=1)
        for problem, pattern in patterns.items()
        if pattern.search(line)
    ]


def test_no_file_downgrades_type_checking_mode() -> None:
    found = [
        problem
        for path in candidate_files()
        for problem in suppressions(path, path.read_text(errors="replace"))
    ]
    assert not found, (
        "Fix the code instead, or use a line-level `# pyright: ignore[rule]` with a reason:\n  "
        + "\n  ".join(found)
    )


def test_the_scan_finds_each_suppression_and_spares_rule_ignores() -> None:
    python = Path("x.py")
    for line in (
        "# pyright: basic",
        "#pyright: strict=false",
        "# pyright: reportAssignmentType=false",
        "x = 1  # pyright: ignore[reportAssignmentType]",
        f"from typing import Any  # noqa: {BANNED_RULES[0]}",
        f"# ruff: noqa: E402, {BANNED_RULES[0]}",
    ):
        assert suppressions(python, line), line
    for line in (
        "// @ts-nocheck",
        "/* eslint-disable */",
        "// oxlint-disable",
        "x; // eslint-disable-line",
        "// eslint-disable -- why",
        "/* oxlint-disable -- why */",
    ):
        assert suppressions(Path("x.js"), line), line
    assert not suppressions(Path("x.js"), "// oxlint-disable-next-line no-debugger -- reason")
    assert not suppressions(python, "x = 1  # pyright: ignore[reportUnknownMemberType] -- reason")
    assert not suppressions(python, "import os  # noqa: E402")
    assert not suppressions(Path("x.md"), "# pyright: basic")


SKIP_MARKERS = {"pytest.skip", "pytest.mark.skip", "pytest.mark.xfail", "unittest.skip"}
POSITIONAL_REASON = {"pytest.skip", "unittest.skip"}


def dotted_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute):
        return f"{dotted_name(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    return ""


def has_skip_reason(name: str, call: ast.Call) -> bool:
    reason = next((keyword.value for keyword in call.keywords if keyword.arg == "reason"), None)
    if reason is None and name in POSITIONAL_REASON and call.args:
        reason = call.args[0]
    return (
        isinstance(reason, ast.Constant)
        and isinstance(reason.value, str)
        and bool(reason.value.strip())
    )


def skips_without_reason(path: Path, text: str) -> list[str]:
    tree = ast.parse(text)
    calls = {id(node.func): node for node in ast.walk(tree) if isinstance(node, ast.Call)}
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute | ast.Name):
            continue
        name = dotted_name(node)
        if name not in SKIP_MARKERS:
            continue
        call = calls.get(id(node))
        if call is None or not has_skip_reason(name, call):
            problems.append(f"{path.name}:{node.lineno} {name}")
    return problems


def test_every_skip_gives_a_reason() -> None:
    found = [
        problem
        for path in candidate_files()
        if is_python_file(path)
        for problem in skips_without_reason(path, path.read_text(errors="replace"))
    ]
    assert not found, (
        "Give each skip, xfail, or unittest skip a non-empty reason, as in "
        '`@pytest.mark.skip(reason="needs a GPU")`: ' + ", ".join(found)
    )


def test_the_skip_scan_flags_bare_and_empty_reasons_and_spares_reasoned_ones() -> None:
    python = Path("x.py")
    for text in (
        "@pytest.mark.skip\ndef test_a() -> None: ...",
        "pytest.skip('')",
        "pytest.mark.xfail()",
        "@unittest.skip(None)\ndef test_b() -> None: ...",
    ):
        assert skips_without_reason(python, text), text
    for text in (
        'pytest.skip("node is not installed")',
        '@pytest.mark.skip(reason="needs a GPU")\ndef test_a() -> None: ...',
        "pytest.mark.xfail(reason='known gap', strict=True)",
    ):
        assert not skips_without_reason(python, text), text
