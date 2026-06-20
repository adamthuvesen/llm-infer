"""Phase D evidence: the three-way benchmark (naive HF vs llm-infer vs vLLM).

The workload and report helpers are pure and CPU-importable; the per-system runners pull in
torch/transformers (and, only inside ``run_vllm``, vLLM). The GPU runs live in
``scripts/modal_benchmark.py`` on the Modal A100. See ``docs/benchmark.md`` for methodology.
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
from llm_infer.benchmarks.workload import BenchRequest, Workload, build_workload

__all__ = [
    "BenchRequest",
    "Workload",
    "build_workload",
    "normalize_at_eos",
    "total_output_tokens",
    "throughput_rows",
    "gpu_snapshot",
    "library_versions",
    "assemble_markdown",
]
