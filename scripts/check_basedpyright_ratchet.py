"""Fail when the basedpyright baseline holds more errors than it does on origin/main."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

BASELINE = ".basedpyright/baseline.json"


def count_errors(text: str) -> int:
    files: dict[str, list[object]] = json.loads(text)["files"]
    return sum(len(entries) for entries in files.values())


def main() -> int:
    here = count_errors(Path(BASELINE).read_text(encoding="utf-8"))
    show = subprocess.run(
        ["git", "show", f"origin/main:{BASELINE}"], capture_output=True, text=True, check=False
    )
    if show.returncode != 0:
        print(f"basedpyright baseline: {here} errors (first baseline; origin/main has none)")
        return 0
    on_main = count_errors(show.stdout)
    if here > on_main:
        print(
            f"basedpyright baseline grew: {here} errors, origin/main has {on_main}", file=sys.stderr
        )
        print("Fix the new errors instead of baselining them.", file=sys.stderr)
        return 1
    print(f"basedpyright baseline: {here} errors (origin/main: {on_main})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
