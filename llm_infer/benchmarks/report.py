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


def total_output_tokens(
    outputs: dict[str, list[int]], eos_token_ids: frozenset[int]
) -> int:
    """Sum of the scored continuation lengths over all requests."""
    return sum(len(normalize_at_eos(ids, eos_token_ids)) for ids in outputs.values())


def throughput_rows(
    results: list[dict],
    eos_token_ids: frozenset[int],
    baseline_system: str = "hf_sequential",
) -> list[dict]:
    """One table row per system: median wall-clock, total tokens, tok/s, speedup vs baseline.

    ``results`` is a list of ``{"system", "outputs", "per_iter_seconds", "equivalent"}``.
    Only systems flagged ``equivalent`` get a tok/s number — a system whose tokens do not
    match the reference (beyond a traced tie) reports no throughput, never a fast-but-wrong
    number.
    """
    eos = frozenset(eos_token_ids)
    rows: list[dict] = []
    baseline_tps: float | None = None
    for r in results:
        per_iter = r["per_iter_seconds"]
        median_s = statistics.median(per_iter) if per_iter else float("nan")
        tokens = total_output_tokens(r["outputs"], eos)
        equivalent = r.get("equivalent", False)
        tps = (tokens / median_s) if (equivalent and median_s > 0) else None
        if r["system"] == baseline_system and tps is not None:
            baseline_tps = tps
        rows.append(
            {
                "system": r["system"],
                "equivalent": equivalent,
                "median_seconds": median_s,
                "total_output_tokens": tokens,
                "tokens_per_second": tps,
                "iters": len(r["per_iter_seconds"]),
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
    """A human-readable three-way table plus the pinned config, for the writeup/PR."""
    header = (
        "| system | equivalent | median s | output tok | tok/s | speedup vs naive |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for row in rows:
        tps = row["tokens_per_second"]
        speedup = row["speedup_vs_baseline"]
        lines.append(
            f"| {row['system']} | {'yes' if row['equivalent'] else 'NO'} "
            f"| {row['median_seconds']:.3f} | {row['total_output_tokens']} "
            f"| {tps:.1f} | {speedup:.2f}× |"
            if tps is not None
            else f"| {row['system']} | {'yes' if row['equivalent'] else 'NO'} "
            f"| {row['median_seconds']:.3f} | {row['total_output_tokens']} | — | — |"
        )
    table = "\n".join(lines)
    workload = config.get("workload", {})
    meta = (
        f"\n\nWorkload: {workload.get('num_requests', '?')} requests × "
        f"{workload.get('max_new_tokens', '?')} new tokens, greedy. "
        f"GPU: {config.get('gpu', {}).get('name', '?')}. "
        f"Source: {workload.get('source', '?')}"
    )
    return table + meta
