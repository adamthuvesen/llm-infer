"""Every citation in the AGENTS.md Enforced Rules table names a real file, config key, or target."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _cited_names() -> list[str]:
    section = (ROOT / "AGENTS.md").read_text(encoding="utf-8").split("## Enforced Rules")[1]
    rows = [line for line in section.split("\n## ")[0].splitlines() if line.startswith("| ")]
    return [name for row in rows[2:] for name in re.findall(r"`([^`]+)`", row.split("|")[2])]


def _is_real(name: str) -> bool:
    if name.startswith("make "):
        target = name.removeprefix("make ")
        makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
        return re.search(rf"^{re.escape(target)}:", makefile, re.MULTILINE) is not None
    if name.startswith("[tool."):
        return name in (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return (ROOT / name).exists()


def test_enforced_rules_cite_real_checks() -> None:
    cited = _cited_names()
    missing = [name for name in cited if not _is_real(name)]
    assert cited, "AGENTS.md has no `## Enforced Rules` table rows"
    assert not missing, (
        "AGENTS.md Enforced Rules cites names that do not exist. Cite a repo-root path, a "
        f"`[tool.*]` key in pyproject.toml, or a `make <target>`; fix or remove: {missing}"
    )
