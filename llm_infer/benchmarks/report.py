"""Shared scoring and environment helpers for benchmark records.

* :func:`normalize_at_eos` defines the one continuation every system is scored on — up to
  and including the first EOS — so trailing differences in how each system represents stop
  never skew the token count or the equivalence check.
* :func:`total_output_tokens` sums those scored continuations so every harness counts
  tokens the same way.
* :func:`gpu_snapshot` / :func:`library_versions` capture the GPU and library versions
  that make the result reproducible.

Reference gating and public tok/s eligibility live in
:mod:`llm_infer.benchmarks.reference_policy`.
"""

from __future__ import annotations

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
