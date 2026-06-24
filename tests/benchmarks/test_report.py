"""Benchmark report honesty: a backend that fails the oracle reports no throughput.

Doctrine (``docs/scoping.md``): *"Validate before you brag. A backend that fails the oracle
reports no tok/s."* A system whose bf16 output diverges from fp32 truth beyond genuine ties
(``agrees_with_truth=False``) must not post a tok/s or speedup, even if its wall-clock is the
fastest. Its measured facts (token count, wall-clock) remain; only the bragging numbers are
suppressed.
"""

from __future__ import annotations

from llm_infer.benchmarks.report import throughput_rows

EOS = frozenset({999})


def _result(system: str, agrees: bool | None, *, seconds: float, tokens: int = 10) -> dict:
    return {
        "system": system,
        "outputs": {"r0": list(range(tokens))},  # `tokens` non-EOS ids
        "per_iter_seconds": [seconds],
        "agrees_with_truth": agrees,
    }


def test_oracle_failing_backend_reports_no_throughput_or_speedup() -> None:
    results = [
        _result("hf_sequential", agrees=None, seconds=2.0),  # baseline / truth, not flagged
        _result("flash_good", agrees=True, seconds=1.0),
        _result("flash_broken", agrees=False, seconds=0.5),  # fastest wall-clock, but wrong
    ]
    rows = {r["system"]: r for r in throughput_rows(results, EOS)}

    # Baseline and the agreeing backend report tok/s and speedup normally.
    assert rows["hf_sequential"]["tokens_per_second"] == 5.0  # 10 tok / 2 s
    assert rows["flash_good"]["tokens_per_second"] == 10.0  # 10 tok / 1 s
    assert rows["flash_good"]["speedup_vs_baseline"] == 2.0

    # The oracle-failing backend posts NO tok/s and NO speedup despite being the fastest.
    assert rows["flash_broken"]["tokens_per_second"] is None
    assert rows["flash_broken"]["speedup_vs_baseline"] is None
    # But its measured facts stay — the divergence is reported, not hidden.
    assert rows["flash_broken"]["total_output_tokens"] == 10
    assert rows["flash_broken"]["median_seconds"] == 0.5
    assert rows["flash_broken"]["agrees_with_truth"] is False
