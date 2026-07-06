"""Turn raw runner outputs into a clear benchmark record.

Three jobs, all in service of "match before measuring speed":

* :func:`normalize_at_eos` defines the one continuation every system is scored on — up to
  and including the first EOS — so trailing differences in how each system represents stop
  never skew the token count or the equivalence check.
* :func:`throughput_rows` computes tokens/s the same way for every system: identical total
  output tokens (they decode the same continuation) divided by the median measured
  wall-clock. Median, not best, so one lucky iteration cannot flatter a system.
* :func:`gpu_snapshot` / :func:`library_versions` capture the GPU and library versions
  that make the result reproducible.

Cross-system agreement is decided by the benchmark harness against the fp32 reference;
this module only reports those facts and suppresses speed claims for rows that fail.
"""

from __future__ import annotations

import statistics
import subprocess
from collections.abc import Iterable


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

    ``results`` is a list of ``{"system", "outputs", "per_iter_seconds", "matches_reference"}``.
    Benchmark rule (``docs/scoping.md``: *"a backend that fails the reference check
    reports no tok/s"*):
    a system whose ``matches_reference`` is explicitly ``False`` — its bf16 output diverged from
    the fp32 full-recompute reference beyond genuine ties — gets ``tokens_per_second`` and
    ``speedup_vs_baseline`` of ``None``, so a wrong backend can never post a speed number. Its
    token count and wall-clock stay (they are measured facts), and the agreement column shows the
    divergence. ``matches_reference`` of ``None`` means no reference comparison was
    recorded for this row, so it is reported normally.
    """
    eos = frozenset(eos_token_ids)
    rows: list[dict] = []
    baseline_tps: float | None = None
    for r in results:
        per_iter = r["per_iter_seconds"]
        median_s = statistics.median(per_iter) if per_iter else float("nan")
        tokens = total_output_tokens(r["outputs"], eos)
        tps = (tokens / median_s) if median_s and median_s > 0 else None
        # A backend that fails the reference check reports no throughput (and so no speedup).
        if r.get("matches_reference") is False:
            tps = None
        if r["system"] == baseline_system and tps is not None:
            baseline_tps = tps
        rows.append(
            {
                "system": r["system"],
                "matches_reference": r.get("matches_reference"),
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
    """A human-readable benchmark table plus the pinned config, for the writeup/PR."""
    header = (
        "| system | agrees fp32 reference | median s | output tok | tok/s | speedup vs naive |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for row in rows:
        tps = row["tokens_per_second"]
        speedup = row["speedup_vs_baseline"]
        agrees = row.get("matches_reference")
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


def _gpu_label(gpu: object) -> str:
    """Render the GPU header from either a flat snapshot or a per-function dict of snapshots."""
    if isinstance(gpu, dict) and "name" in gpu:
        return str(gpu["name"])
    if isinstance(gpu, dict):
        parts = [f"{k}={v.get('name', '?')}" for k, v in gpu.items() if isinstance(v, dict)]
        return " | ".join(parts) if parts else "?"
    return str(gpu) if gpu else "?"
