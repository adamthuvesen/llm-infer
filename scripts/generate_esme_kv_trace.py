"""Generate a committed schema-v3 KV trace from a real **Esme** engine run.

The actual ``InferenceEngine`` is run over the tiny ``llm_pretrain_dense_v1`` Esme bundle with
preemption enabled. Every event is emitted by the real engine, scheduler, and allocator on the
Esme backend. The tiny bundle is CPU-only and deterministic. A fixed step clock is used so the
artifact stays byte-stable and can be checked for regressions.

    uv run python scripts/generate_esme_kv_trace.py            # print to stdout
    uv run python scripts/generate_esme_kv_trace.py --write    # rewrite the committed artifact
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from itertools import count
from pathlib import Path

from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.request import Request
from llm_infer.tracing import TraceRecorder

# The visualizer's committed sample is recorded from the real engine.
FIXTURE_PATH = Path("docs/assets/esme_kv_trace_schema_v3.jsonl")

BLOCK_SIZE = 4
# Three short prompts of footprint 1 block each are all admitted into a 3-block pool, then evict
# as they decode — a real recompute preemption, exactly the shape the visualizer is built to show.
NUM_BLOCKS = 3
MAX_NEW_TOKENS = 6
PROMPTS = (("esme-a", [1, 4, 7]), ("esme-b", [2, 5, 8]), ("esme-c", [3, 6, 9]))
# Deterministic 6 ms-per-tick clock: stands in for a plausible step latency without reading the
# wall clock, so the emitted throughput samples are byte-identical across machines and runs.
STEP_SECONDS = 0.006


def _deterministic_clock() -> Callable[[], float]:
    ticks = count()
    return lambda: next(ticks) * STEP_SECONDS


def build_trace_jsonl() -> str:
    """Run the tiny Esme bundle through the real engine with tracing; return schema-v3 JSONL."""
    with tempfile.TemporaryDirectory() as tmp:
        from llm_infer.fixtures.tiny_pretrain_bundle import (
            write_tiny_pretrain_bundle as _write_tiny_bundle,
        )

        runtime = load_model_runtime("esme", bundle_path=_write_tiny_bundle(Path(tmp)))
        recorder = TraceRecorder()
        engine = InferenceEngine(
            runtime.model,
            block_size=BLOCK_SIZE,
            num_blocks=NUM_BLOCKS,
            capabilities=runtime.capabilities,
            preemption=True,
            trace=recorder,
            trace_clock=_deterministic_clock(),
        )
        for request_id, prompt in PROMPTS:
            engine.add_request(
                Request(request_id, list(prompt), MAX_NEW_TOKENS, runtime.eos_token_ids)
            )
        engine.run()
        return recorder.to_jsonl() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the committed Esme KV trace artifact.")
    parser.add_argument(
        "--write", action="store_true", help=f"rewrite {FIXTURE_PATH} instead of printing"
    )
    args = parser.parse_args()
    jsonl = build_trace_jsonl()
    if args.write:
        root = Path(__file__).resolve().parent.parent
        (root / FIXTURE_PATH).write_text(jsonl, encoding="utf-8")
        print(f"wrote {FIXTURE_PATH}")
    else:
        print(jsonl, end="")


if __name__ == "__main__":
    main()
