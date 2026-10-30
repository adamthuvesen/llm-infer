"""Shared harness semantics stay observable without a CUDA worker."""

from __future__ import annotations

import pytest
import torch

from llm_infer.benchmarks.esme_measurement import qualified_system_row
from llm_infer.benchmarks.esme_paged import (
    EsmeBenchRequest,
    ReferenceOutputCache,
    measure_repeated,
)


class _ConstantOracle:
    def __init__(self) -> None:
        self.calls = 0

    def logits(self, token_ids: list[int]) -> torch.Tensor:
        self.calls += 1
        return torch.tensor([[0.0, 0.0, 1.0]] * len(token_ids))


def test_reference_cache_scopes_oracle_budget_and_eos_and_returns_copies() -> None:
    oracle = _ConstantOracle()
    cache = ReferenceOutputCache(oracle, max_new_tokens=2, eos_token_ids=frozenset())
    first = EsmeBenchRequest("first", "one", (1,))
    repeated = EsmeBenchRequest("repeated", "same", (1,))
    outputs = cache([first, repeated])
    assert outputs == {"first": [2, 2], "repeated": [2, 2]}
    assert oracle.calls == 2
    outputs["first"].append(99)
    assert cache([repeated]) == {"repeated": [2, 2]}
    assert oracle.calls == 2

    eos_cache = ReferenceOutputCache(oracle, max_new_tokens=2, eos_token_ids=frozenset({2}))
    assert eos_cache([first]) == {"first": [2]}
    assert oracle.calls == 3
    shorter = ReferenceOutputCache(oracle, max_new_tokens=1, eos_token_ids=frozenset())
    assert shorter([first]) == {"first": [2]}
    other_oracle = _ConstantOracle()
    independent = ReferenceOutputCache(other_oracle, max_new_tokens=2, eos_token_ids=frozenset())
    assert independent([first]) == {"first": [2, 2]}
    assert other_oracle.calls == 2


def test_repeated_measurement_keeps_warmup_outside_clock_and_all_measured_durations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    clock_values = iter([10.0, 12.0, 20.0, 25.0])
    generations = 0

    def clock() -> float:
        events.append("clock")
        return next(clock_values)

    def decode_once() -> dict[str, list[int]]:
        nonlocal generations
        events.append("decode")
        generations += 1
        return {"request": [generations]}

    def synchronize() -> None:
        events.append("sync")

    monkeypatch.setattr("llm_infer.benchmarks.esme_paged.time.perf_counter", clock)
    durations, outputs = measure_repeated(decode_once, warmup=1, iters=2, synchronize=synchronize)
    assert durations == [2.0, 5.0]
    assert outputs == {"request": [3]}
    assert events == ["decode", "sync"] + ["sync", "clock", "decode", "sync", "clock"] * 2


def test_qualified_row_serializes_complete_policy_and_keeps_failed_raw_timing() -> None:
    oracle = _ConstantOracle()
    requests = [EsmeBenchRequest("request", "one", (1,))]
    reference = {"request": [2]}
    exact = qualified_system_row(oracle, requests, reference, reference, frozenset(), [1.0, 3.0])
    assert exact["median_seconds"] == 2.0
    assert exact["raw_tokens_per_second"] == 0.5
    assert exact["tokens_per_second"] == 0.5
    assert exact["agreement"] == {
        "exact": 1,
        "tie": 0,
        "nontie": 0,
        "total": 1,
        "ties_sample": [],
        "divergences_sample": [],
        "review_required": 0,
        "failed": 0,
        "numerical_evidence": [],
    }
    failed = qualified_system_row(oracle, requests, {}, reference, frozenset(), [1.0, 3.0])
    assert failed["reference_status"] == "failed"
    assert failed["headline_eligible"] is False
    assert failed["tokens_per_second"] is None
    assert failed["raw_tokens_per_second"] == 0.0
    assert failed["per_iter_seconds"] == [1.0, 3.0]
