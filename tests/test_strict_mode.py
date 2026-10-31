"""No file turns type checking or a lint ban off with a comment.

basedpyright honors a file-level `# pyright:` comment, ruff honors a `noqa` on a banned API, and
the visualizer's type check honors `@ts-nocheck`. No config setting blocks any of them.
"""

import re
from pathlib import Path

from tests.support.repo_files import candidate_files, is_python_file

# Codes in `pyproject.toml::tool.ruff.lint` that hold a ban rather than a style choice.
BANNED_RULES = ("TID251",)
PYTHON_SUPPRESSIONS = {
    "file-level `# pyright:` comment changes type checking for the file": re.compile(
        r"^\s*#\s*pyright:"
    ),
    "`noqa` silences a banned API": re.compile(
        rf"#\s*(?:ruff:\s*)?noqa:[^#\n]*\b(?:{'|'.join(BANNED_RULES)})\b", re.IGNORECASE
    ),
}
SCRIPT_SUPPRESSIONS = {
    "`@ts-nocheck` turns the visualizer type check off": re.compile(r"@ts-nocheck\b"),
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
        f"from typing import Any  # noqa: {BANNED_RULES[0]}",
        f"# ruff: noqa: E402, {BANNED_RULES[0]}",
    ):
        assert suppressions(python, line), line
    assert suppressions(Path("x.js"), "// @ts-nocheck")
    assert not suppressions(python, "x = 1  # pyright: ignore[reportUnknownMemberType] -- reason")
    assert not suppressions(python, "import os  # noqa: E402")
    assert not suppressions(Path("x.md"), "# pyright: basic")
