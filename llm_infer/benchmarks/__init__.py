"""Phase D evidence: the three-way benchmark (naive HF vs llm-infer vs vLLM).

The workload and report helpers are pure and CPU-importable; the per-system runners pull in
torch/transformers (and, only inside ``run_vllm``, vLLM). The GPU runs live in
``scripts/modal_benchmark.py`` on the Modal A100. See ``docs/benchmark.md`` for methodology.
"""

from __future__ import annotations

from llm_infer.benchmarks.report import (
    A100_80GB_USD_PER_HOUR,
    assemble_markdown,
    assemble_rollout_markdown,
    gpu_snapshot,
    library_versions,
    normalize_at_eos,
    rollout_rows,
    throughput_rows,
    total_output_tokens,
)
from llm_infer.benchmarks.workload import (
    BenchRequest,
    SamplingConfig,
    Workload,
    build_rollout_workload,
    build_workload,
)

__all__ = [
    "A100_80GB_USD_PER_HOUR",
    "BenchRequest",
    "SamplingConfig",
    "Workload",
    "build_workload",
    "build_rollout_workload",
    "normalize_at_eos",
    "total_output_tokens",
    "throughput_rows",
    "rollout_rows",
    "gpu_snapshot",
    "library_versions",
    "assemble_markdown",
    "assemble_rollout_markdown",
]
