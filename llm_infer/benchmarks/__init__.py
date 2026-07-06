"""Benchmark helpers for Esme speed records.

The workload and report helpers are pure and CPU-importable; the per-system runners pull in
torch/transformers or vLLM inside the harnesses that need them. See ``docs/benchmark.md`` for
methodology.
"""

from __future__ import annotations

from llm_infer.benchmarks.report import (
    assemble_markdown,
    gpu_snapshot,
    library_versions,
    normalize_at_eos,
    throughput_rows,
    total_output_tokens,
)
from llm_infer.benchmarks.workload import (
    BenchRequest,
    Workload,
)

__all__ = [
    "BenchRequest",
    "Workload",
    "normalize_at_eos",
    "total_output_tokens",
    "throughput_rows",
    "gpu_snapshot",
    "library_versions",
    "assemble_markdown",
]
