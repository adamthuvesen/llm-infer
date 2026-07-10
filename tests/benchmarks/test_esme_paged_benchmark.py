"""Smoke test for the Esme paged-KV vs full-recompute benchmark comparison.

Exercises the shared comparison plumbing on the tiny synthetic bundle (CPU, zero spend): both
the paged engine path and the full-recompute baseline must match the same
``PretrainBundleModel.logits()`` greedy reference, and only a matching system reports tok/s.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_infer.benchmarks.esme_paged import (
    EsmeBenchRequest,
    SystemTiming,
    compare_paged_vs_recompute,
    reference_outputs,
    requests_at_context_length,
)
from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as _write_tiny_bundle
from llm_infer.model.runtime import load_model_runtime


@pytest.fixture
def tiny_runtime(tmp_path: Path):
    bundle = _write_tiny_bundle(tmp_path)
    # Give the tiny bundle the Esme-shaped manifest fields the runtime reads for ids/EOS.
    import json

    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = {"name": "tiny-dense", "id": "tiny-dense"}
    manifest["eos_token_ids"] = [9]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_model_runtime("esme", bundle_path=bundle, dtype=torch.float32)


def _requests() -> list[EsmeBenchRequest]:
    return [
        EsmeBenchRequest("r0", "p0", (1, 4, 7)),
        EsmeBenchRequest("r1", "p1", (2, 5)),
    ]


def test_requests_at_context_length_repeats_valid_prompts_to_exact_size() -> None:
    resized = requests_at_context_length(_requests(), 8)

    assert [len(request.prompt_ids) for request in resized] == [8, 8]
    assert resized[0].prompt_ids == (1, 4, 7, 1, 4, 7, 1, 4)
    assert resized[1].prompt_ids == (2, 5, 2, 5, 2, 5, 2, 5)
    assert [request.request_id for request in resized] == ["r0", "r1"]


def test_requests_at_context_length_rejects_zero() -> None:
    with pytest.raises(ValueError, match="context_length must be >= 1"):
        requests_at_context_length(_requests(), 0)


class _OracleMustNotBeUsed:
    def logits(self, _token_ids: list[int]) -> torch.Tensor:
        raise AssertionError("exact/missing-output comparison should not consult logits")


def test_tie_tolerant_agreement_fails_missing_or_extra_request_ids() -> None:
    requests = _requests()
    reference = {"r0": [1, 2], "r1": [3, 4]}

    missing = tie_tolerant_agreement(
        _OracleMustNotBeUsed(),
        requests,
        {"r0": [1, 2]},
        reference,
        frozenset({99}),
    )
    assert missing.total == 2
    assert missing.nontie == 1
    assert not missing.all_ties_or_exact
    assert "missing outputs" in str(missing.divergences_sample)

    extra = tie_tolerant_agreement(
        _OracleMustNotBeUsed(),
        requests,
        {"r0": [1, 2], "r1": [3, 4], "r2": [5]},
        reference,
        frozenset({99}),
    )
    assert extra.total == 2
    assert extra.nontie == 1
    assert not extra.all_ties_or_exact
    assert "unexpected outputs" in str(extra.divergences_sample)


def test_both_systems_match_reference_and_report_throughput(tiny_runtime) -> None:
    requests = _requests()
    timings = compare_paged_vs_recompute(
        tiny_runtime,
        requests,
        max_new_tokens=3,
        block_size=4,
        num_blocks=32,
        warmup=0,
        iters=1,
        device="cpu",
    )

    assert [t.system for t in timings] == ["llm_infer_paged", "full_recompute"]
    for timing in timings:
        assert timing.matches_reference is True
        assert timing.tokens_per_second is not None
        assert timing.total_output_tokens > 0


def test_paged_outputs_equal_full_recompute_reference(tiny_runtime) -> None:
    requests = _requests()
    reference = reference_outputs(
        tiny_runtime.model,
        requests,
        max_new_tokens=3,
        eos_token_ids=tiny_runtime.eos_token_ids,
    )
    timings = compare_paged_vs_recompute(
        tiny_runtime,
        requests,
        max_new_tokens=3,
        block_size=4,
        num_blocks=32,
        warmup=0,
        iters=1,
        device="cpu",
    )
    paged = next(t for t in timings if t.system == "llm_infer_paged")
    assert paged.outputs == reference


def test_diverging_system_reports_no_throughput() -> None:
    diverged = SystemTiming(
        system="llm_infer_paged",
        mode="paged KV + batched decode",
        matches_reference=False,
        median_seconds=1.0,
        total_output_tokens=10,
        outputs={},
    )
    assert diverged.tokens_per_second is None
