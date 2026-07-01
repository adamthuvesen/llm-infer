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
)
from llm_infer.model.runtime import load_model_runtime
from tests.correctness.test_pretrain_bundle import _write_tiny_bundle


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
