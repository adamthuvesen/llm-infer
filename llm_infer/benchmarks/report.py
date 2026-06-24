"""Turn raw runner outputs into an honest, fully-pinned benchmark record.

Three jobs, all in service of "validate before you brag":

* :func:`normalize_at_eos` defines the one continuation every system is scored on — up to
  and including the first EOS — so trailing differences in how each system represents stop
  never skew the token count or the equivalence check.
* :func:`throughput_rows` computes tokens/s the same way for every system: identical total
  output tokens (they decode the same continuation) divided by the median measured
  wall-clock. Median, not best, so one lucky iteration cannot flatter a system.
* :func:`gpu_snapshot` / :func:`library_versions` capture the environment the plan demands
  pinned — GPU, clocks, power cap, and every library version — into the result.

Cross-system *equivalence* (does each system decode the same tokens, and is any divergence
a genuine numerical tie) is adjudicated in ``scripts/modal_benchmark.py`` with the fp32
reference, reusing the correctness oracle's tie policy; this module only reports it.
"""

from __future__ import annotations

import statistics
import subprocess
from collections.abc import Iterable

# Modal A100-80GB on-demand rate: $0.000694/s = $2.50/hr (modal.com/pricing, 2026-06-21).
# Pinned here so $/1k-rollout is a transparent, reproducible derivation of one timed run.
A100_80GB_USD_PER_HOUR = 2.50


def normalize_at_eos(token_ids: Iterable[int], eos_token_ids: frozenset[int]) -> list[int]:
    """The scored continuation: tokens up to and including the first EOS (or all of them).

    Different systems include or omit the stop token differently; truncating every system
    at the first EOS gives one canonical continuation to count and compare.
    """
    out: list[int] = []
    for tok in token_ids:
        out.append(tok)
        if tok in eos_token_ids:
            break
    return out


def total_output_tokens(outputs: dict[str, list[int]], eos_token_ids: frozenset[int]) -> int:
    """Sum of the scored continuation lengths over all requests."""
    return sum(len(normalize_at_eos(ids, eos_token_ids)) for ids in outputs.values())


def throughput_rows(
    results: list[dict],
    eos_token_ids: frozenset[int],
    baseline_system: str = "hf_sequential",
) -> list[dict]:
    """One table row per system: median wall-clock, total tokens, tok/s, speedup vs baseline.

    ``results`` is a list of ``{"system", "outputs", "per_iter_seconds", "agrees_with_truth"}``.
    Every system decodes the *same* token count (equal work), so tok/s is reported for all —
    they are all valid greedy decoders (each's correctness established by its own oracle).
    ``agrees_with_truth`` is a transparency annotation (does the bf16 output match fp32
    full-recompute truth except at genuine ties), not a tok/s suppressor; a genuinely broken
    backend shows up as a gross divergence in the agreement profile.
    """
    eos = frozenset(eos_token_ids)
    rows: list[dict] = []
    baseline_tps: float | None = None
    for r in results:
        per_iter = r["per_iter_seconds"]
        median_s = statistics.median(per_iter) if per_iter else float("nan")
        tokens = total_output_tokens(r["outputs"], eos)
        tps = (tokens / median_s) if median_s and median_s > 0 else None
        if r["system"] == baseline_system and tps is not None:
            baseline_tps = tps
        rows.append(
            {
                "system": r["system"],
                "agrees_with_truth": r.get("agrees_with_truth"),
                "median_seconds": median_s,
                "total_output_tokens": tokens,
                "tokens_per_second": tps,
                "iters": len(per_iter),
            }
        )
    for row in rows:
        tps = row["tokens_per_second"]
        row["speedup_vs_baseline"] = (
            (tps / baseline_tps) if (tps is not None and baseline_tps) else None
        )
    return rows


def rollout_rows(
    results: list[dict],
    eos_token_ids: frozenset[int],
    num_completions: int,
    *,
    usd_per_hour: float = A100_80GB_USD_PER_HOUR,
    baseline_system: str = "hf_sequential",
) -> list[dict]:
    """Rollout table rows: median wall-clock, output tokens, tok/s, and $/1k rollouts.

    Builds on :func:`throughput_rows` (same median wall-clock + tok/s) and adds the rollout
    economics: ``$/1k rollouts`` = the cost to generate 1000 completions at the pinned A100
    rate, derived from this batch's wall-clock and its ``num_completions``. Under sampling the
    systems decode *different* token volumes (different RNG), so tok/s is the apples-to-apples
    speed number while wall-clock and $/1k reflect each system's own sampled batch.
    """
    rows = throughput_rows(results, eos_token_ids, baseline_system=baseline_system)
    for row in rows:
        seconds = row["median_seconds"]
        # ``seconds == seconds`` rejects NaN (the no-timing case) without importing math.
        valid = num_completions > 0 and seconds == seconds and seconds > 0
        row["usd_per_1k_rollouts"] = (
            (seconds / 3600.0) * usd_per_hour / num_completions * 1000.0 if valid else None
        )
    return rows


def gpu_snapshot() -> dict[str, object]:
    """GPU name, SM/max clocks, power cap, memory — best-effort from nvidia-smi."""
    fields = "name,clocks.sm,clocks.max.sm,power.limit,memory.total"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"nvidia-smi unavailable: {exc}"}
    values = [v.strip() for v in out.splitlines()[0].split(",")]
    keys = ["name", "clocks_sm", "clocks_max_sm", "power_limit", "memory_total"]
    return dict(zip(keys, values, strict=False))


def library_versions() -> dict[str, str]:
    """Versions of every library whose behavior the numbers depend on."""
    import torch

    versions: dict[str, str] = {"torch": torch.__version__}
    versions["cuda"] = torch.version.cuda or "none"
    for name in ("transformers", "flash_attn", "vllm", "numpy"):
        try:
            versions[name] = __import__(name).__version__
        except (ImportError, AttributeError):
            versions[name] = "absent"
    return versions


def assemble_markdown(rows: list[dict], config: dict) -> str:
    """A human-readable three-way table plus the pinned config, for the writeup/PR."""
    header = (
        "| system | agrees fp32 truth | median s | output tok | tok/s | speedup vs naive |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for row in rows:
        tps = row["tokens_per_second"]
        speedup = row["speedup_vs_baseline"]
        agrees = row.get("agrees_with_truth")
        agree_s = "—" if agrees is None else ("yes" if agrees else "diverges")
        tps_s = f"{tps:.1f}" if tps is not None else "—"
        speedup_s = f"{speedup:.2f}×" if speedup is not None else "—"
        lines.append(
            f"| {row['system']} | {agree_s} | {row['median_seconds']:.3f} "
            f"| {row['total_output_tokens']} | {tps_s} | {speedup_s} |"
        )
    table = "\n".join(lines)
    workload = config.get("workload", {})
    meta = (
        f"\n\nWorkload: {workload.get('num_requests', '?')} requests × "
        f"{workload.get('max_new_tokens', '?')} new tokens, greedy. "
        f"GPU: {_gpu_label(config.get('gpu', {}))}. "
        f"Source: {workload.get('source', '?')}"
    )
    return table + meta


def assemble_rollout_markdown(rows: list[dict], config: dict) -> str:
    """The rollout table (wall-clock · output tok · tok/s · $/1k) plus its pinned config."""
    header = (
        "| system | wall-clock s | output tok | tok/s | $/1k rollouts |\n"
        "| --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for row in rows:
        tps = row["tokens_per_second"]
        usd = row.get("usd_per_1k_rollouts")
        tps_s = f"{tps:.1f}" if tps is not None else "—"
        usd_s = f"${usd:.2f}" if usd is not None else "—"
        lines.append(
            f"| {row['system']} | {row['median_seconds']:.2f} "
            f"| {row['total_output_tokens']} | {tps_s} | {usd_s} |"
        )
    table = "\n".join(lines)
    workload = config.get("workload", {})
    meta = (
        f"\n\n{workload.get('completions', '?')} completions "
        f"({workload.get('num_prompts', '?')} prompts × G={workload.get('num_generations', '?')}), "
        f"≤{workload.get('max_completion_length', '?')} tokens, "
        f"temp={workload.get('temperature', '?')} top_p={workload.get('top_p', '?')} "
        f"seed={workload.get('seed', '?')}. GPU: {_gpu_label(config.get('gpu', {}))}. "
        f"$/1k at {config.get('usd_per_hour', A100_80GB_USD_PER_HOUR)}/hr A100-80GB. "
        f"Under sampling each system decodes its own token volume — tok/s is the speed metric; "
        f"wall-clock and $/1k include each system's sampled length."
    )
    return table + meta


def _gpu_label(gpu: object) -> str:
    """Render the GPU header from either a flat snapshot or a per-function dict of snapshots."""
    if isinstance(gpu, dict) and "name" in gpu:
        return str(gpu["name"])
    if isinstance(gpu, dict):
        parts = [f"{k}={v.get('name', '?')}" for k, v in gpu.items() if isinstance(v, dict)]
        return " | ".join(parts) if parts else "?"
    return str(gpu) if gpu else "?"
