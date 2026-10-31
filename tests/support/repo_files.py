"""List the checkout's files the way an agent sees them, including new files it has not staged."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def is_python_file(path: Path) -> bool:
    if path.suffix:
        return path.suffix == ".py"
    with path.open("rb") as handle:
        first_line = handle.readline()
    return first_line.startswith(b"#!") and b"python" in first_line


def candidate_files() -> list[Path]:
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
    ).stdout.decode()
    paths = (ROOT / name for name in listing.split("\0") if name)
    return [path for path in paths if path.is_file()]
