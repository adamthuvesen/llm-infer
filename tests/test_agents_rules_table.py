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


def _table(agents: str) -> tuple[list[str], list[str]]:
    """The table's rows and the problems with its shape: header, separator, and trailing prose."""
    heading = re.search(r"^## Enforced Rules$", agents, re.MULTILINE)
    assert heading, "AGENTS.md has no `## Enforced Rules` heading"
    lines = agents[heading.end() :].split("\n## ")[0].splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("|")), len(lines))
    end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
    table, problems = lines[start:end], []
    if not table or not re.match(r"^\|\s*Rule\s*\|\s*Enforced by\s*\|", table[0]):
        problems.append("the table must start with `| Rule | Enforced by |`")
    if len(table) < 2 or not re.fullmatch(r"[|\s:-]+", table[1]):
        problems.append("the header needs a `| --- | --- |` separator")
    if any(line.strip() for line in lines[end:]):
        problems.append("prose after the table: move it before the table")
    rows = table[2:]
    assert rows, "AGENTS.md `## Enforced Rules` has no table rows"
    return rows, problems


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
    rows, problems = _table(agents)
    for row in rows:
        rule, enforced_by = [*row.split("|")[1:3], ""][:2]
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
        "|nospace| none |\n"
    )
    assert citation_problems(table) == [
        "bare: cites nothing",
        "target: `Makefile::check-foo`",
        "def: `tests/test_strict_mode.py::no_such_test`",
        "key: `pyproject.toml::tool.nope`",
        "nospace: cites nothing",
    ]


def test_the_check_refuses_a_wrong_header_and_trailing_prose() -> None:
    table = "## Enforced Rules\n\n| Thing | Where |\n| --- | --- |\n| ok | `Makefile::check` |\n\nLater.\n"
    assert citation_problems(table) == [
        "the table must start with `| Rule | Enforced by |`",
        "prose after the table: move it before the table",
    ]
