"""No doc shows a runnable paid command with its approval already added.

An agent copies commands from docs. A doc that shows `--approved` hands it the spend Adam must
approve; the refusal prints the approved command when one is needed.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.support.repo_files import ROOT, candidate_files

APPROVED_COMMAND = re.compile(
    r"\bmodal (?:run|serve|deploy)\b[^`\n]*--approved\b|LLM_INFER_SERVE_APPROVED=1\s+modal\b"
)
DOC_SUFFIXES = {".md", ".txt", ".yml", ".yaml", ".sh"}


def approved_commands(path: Path, text: str) -> list[str]:
    if path.suffix not in DOC_SUFFIXES and path.name != "Makefile":
        return []
    return [
        f"{path.name}:{number}"
        for number, line in enumerate(text.splitlines(), start=1)
        if APPROVED_COMMAND.search(line)
    ]


def test_docs_show_no_approved_command() -> None:
    found = [
        f"{path.relative_to(ROOT).parent}/{hit}"
        for path in candidate_files()
        for hit in approved_commands(path, path.read_text(errors="replace"))
    ]
    assert not found, (
        "Show the command without `--approved` or `LLM_INFER_SERVE_APPROVED=1`; the refusal "
        "prints the approved form for Adam:\n  " + "\n  ".join(found)
    )


def test_the_scan_finds_each_approved_form_and_spares_the_warning() -> None:
    doc = Path("x.md")
    assert approved_commands(doc, "modal run scripts/modal_x.py --command run --approved")
    assert approved_commands(doc, "LLM_INFER_SERVE_APPROVED=1 modal deploy scripts/x.py")
    assert not approved_commands(doc, "Never add `--approved` or `LLM_INFER_SERVE_APPROVED=1`.")
    assert not approved_commands(doc, "modal run scripts/modal_x.py --command run")
