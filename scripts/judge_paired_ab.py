#!/usr/bin/env python3
"""Apply a registered paired-A/B rule to two containers' records of one experiment."""

from __future__ import annotations

import sys
from pathlib import Path

from llm_infer.benchmarks.paired_ab import judge_json_records


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if len(paths) != 2:
        print(
            "usage: uv run scripts/judge_paired_ab.py RECORD_A.json RECORD_B.json", file=sys.stderr
        )
        return 2
    try:
        result = judge_json_records(
            Path(paths[0]).read_bytes(),
            Path(paths[1]).read_bytes(),
            sources=(paths[0], paths[1]),
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print("\n".join(result.lines))
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
