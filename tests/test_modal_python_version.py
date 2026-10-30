"""Modal images run the repository's Python unless a commented pin says otherwise.

An image that mounts or imports repo source must read ``.python-version``. Source that
uses newer syntax fails at container start when the image runs an older Python. A literal
version is allowed only with a ``# modal-python-pin: <reason>`` comment and no repo source.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_KEYWORDS = {"python_version", "add_python"}
PIN_MARKER = "# modal-python-pin:"
SOURCE_MARKERS = ("PYTHONPATH", '"src"', "repo_root_src")


def python_version_violations(source: str, label: str) -> list[str]:
    lines = source.splitlines()
    tree = ast.parse(source)
    statements = [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
    problems: list[str] = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        for keyword in call.keywords:
            if keyword.arg not in VERSION_KEYWORDS or not isinstance(keyword.value, ast.Constant):
                continue
            where = f"{label}:{keyword.value.lineno} {keyword.arg}={keyword.value.value!r}"
            first = max(call.lineno - 2, 0)
            nearby = lines[first : keyword.value.lineno]
            if not any(PIN_MARKER in line for line in nearby):
                problems.append(
                    f"{where} is a literal. Read .python-version, or add "
                    f"`{PIN_MARKER} <reason>` for an image that imports no repo source."
                )
                continue
            enclosing = min(
                (s for s in statements if s.lineno <= call.lineno <= (s.end_lineno or s.lineno)),
                key=lambda s: (s.end_lineno or s.lineno) - s.lineno,
            )
            segment = "\n".join(lines[enclosing.lineno - 1 : enclosing.end_lineno])
            if any(marker in segment for marker in SOURCE_MARKERS):
                problems.append(f"{where} is pinned but the image also uses repo source.")
    return problems


def modal_python_files() -> list[Path]:
    roots = [ROOT / "scripts", ROOT / "src", ROOT]
    found = {p for r in roots for p in (r.rglob("*.py") if r != ROOT else r.glob("*.py"))}
    return sorted(p for p in found if "modal" in p.read_text(encoding="utf-8"))


def test_images_use_the_repository_python_version() -> None:
    problems = [
        problem
        for path in modal_python_files()
        for problem in python_version_violations(
            path.read_text(encoding="utf-8"), str(path.relative_to(ROOT))
        )
    ]
    assert not problems, "\n".join(problems)


def test_the_check_flags_a_literal_version() -> None:
    planted = 'image = modal.Image.debian_slim(python_version="3.13").env({"PYTHONPATH": "/x"})\n'
    assert python_version_violations(planted, "planted.py")
    pinned = f'{PIN_MARKER} legacy tool\nimage = modal.Image.debian_slim(python_version="3.13")\n'
    assert not python_version_violations(pinned, "pinned.py")
