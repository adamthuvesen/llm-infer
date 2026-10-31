"""Every AGENTS.md Enforced Rules row cites `path::name`, and the cited file defines that name."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX = (
    "Cite `path::name`, where the file defines the name: a Python def or class, a JS function or "
    "test title, a Makefile target, or a dotted TOML key"
)


def _rows(agents: str) -> list[str]:
    heading = re.search(r"^## Enforced Rules$", agents, re.MULTILINE)
    assert heading, "AGENTS.md has no `## Enforced Rules` heading"
    section = agents[heading.end() :].split("\n## ")[0]
    rows = [line for line in section.splitlines() if line.startswith("| ")][2:]
    assert rows, "AGENTS.md `## Enforced Rules` has no table rows"
    return rows


def _defines(path: Path, name: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return any(
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and node.name == name
            for node in ast.walk(ast.parse(text))
        )
    if path.suffix in {".js", ".mjs"}:
        title = re.escape(name)
        return re.search(rf"function {title}\(|test\(\"{title}\"", text) is not None
    if path.name == "Makefile":
        return re.search(rf"^{re.escape(name)}:", text, re.MULTILINE) is not None
    if path.suffix == ".toml":
        node: object = tomllib.loads(text)
        for key in name.split("."):
            if not isinstance(node, dict) or key not in node:
                return False
            node = node[key]
        return True
    return False


def citation_problems(agents: str) -> list[str]:
    problems = []
    for row in _rows(agents):
        rule, enforced_by = row.split("|")[1:3]
        citations = re.findall(r"`([^`]+)`", enforced_by)
        if not citations:
            problems.append(f"{rule.strip()}: cites nothing")
        for citation in citations:
            path, _, name = citation.partition("::")
            if not name or not (ROOT / path).is_file() or not _defines(ROOT / path, name):
                problems.append(f"{rule.strip()}: `{citation}`")
    return problems


def test_enforced_rules_cite_defined_names() -> None:
    problems = citation_problems((ROOT / "AGENTS.md").read_text(encoding="utf-8"))
    assert not problems, f"AGENTS.md Enforced Rules citations fail. {FIX}:\n  " + "\n  ".join(
        problems
    )


def test_the_check_refuses_each_rotten_citation() -> None:
    table = (
        "## Enforced Rules\n\n| Rule | Enforced by |\n| --- | --- |\n"
        "| ok | `Makefile::check`, `pyproject.toml::tool.basedpyright` |\n"
        "| bare | none |\n| target | `Makefile::check-foo` |\n"
        "| def | `tests/test_strict_mode.py::no_such_test` |\n| key | `pyproject.toml::tool.nope` |\n"
    )
    assert citation_problems(table) == [
        "bare: cites nothing",
        "target: `Makefile::check-foo`",
        "def: `tests/test_strict_mode.py::no_such_test`",
        "key: `pyproject.toml::tool.nope`",
    ]
