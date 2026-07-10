"""Benchmark helpers for Esme speed records.

The workload and report helpers are pure and CPU-importable; the per-system runners pull in
torch/transformers or vLLM inside the harnesses that need them. See ``docs/benchmark.md`` for
methodology.
"""

from __future__ import annotations

from llm_infer.benchmarks.report import (
    gpu_snapshot,
    library_versions,
    normalize_at_eos,
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
    "gpu_snapshot",
    "library_versions",
]
