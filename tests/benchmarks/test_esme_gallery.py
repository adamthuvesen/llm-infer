"""CPU smoke tests for the technique-gallery experiments on the tiny synthetic bundle.

The gallery's A100 runs live in ``scripts/modal_esme_technique_gallery.py``; these tests pin
the experiment mechanics with zero spend: every experiment gates its outputs on the bundle
oracle, reports the shape of numbers the public story cites, and fails loudly when a
workload cannot demonstrate its technique (e.g. a starved pool that never preempts).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_infer.benchmarks.esme_gallery import (
    run_chunked_prefill_latency,
    run_preemption_starved_pool,
    run_prefix_cache_on_off,
    run_speculative_batch1,
)
from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as _write_tiny_bundle
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving.speculative import SpeculativeDecodingConfig


@pytest.fixture
def tiny_runtime(tmp_path: Path):
    bundle = _write_tiny_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = {"name": "tiny-dense", "id": "tiny-dense"}
    manifest["eos_token_ids"] = [9]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return load_model_runtime("esme", bundle_path=bundle, dtype=torch.float32)


def test_prefix_cache_on_off_gates_and_reports_both_rows(tiny_runtime) -> None:
    result = run_prefix_cache_on_off(
        tiny_runtime,
        tiny_runtime,
        prompt_ids=(1, 4, 7, 2, 5, 8),
        num_siblings=3,
        max_new_tokens=4,
        block_size=4,
        num_blocks=64,
        device="cpu",
        warmup=0,
        iters=1,
    )
    assert [row["label"] for row in result["rows"]] == ["prefix caching on", "prefix caching off"]
    assert all(row["matches_reference"] for row in result["rows"])
    assert all(row["tokens_per_second"] is not None for row in result["rows"])
    assert result["workload"]["prefilled_prompt_tokens_on"] == 6
    assert result["workload"]["prefilled_prompt_tokens_off"] == 18


def test_chunked_prefill_latency_reports_stall_for_both_configs(tiny_runtime) -> None:
    result = run_chunked_prefill_latency(
        tiny_runtime,
        tiny_runtime,
        active_prompts=[(1, 4, 7), (2, 5, 8)],
        active_max_new_tokens=8,
        long_prompts=[(1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8)],
        long_max_new_tokens=4,
        arrival_after_steps=2,
        prefill_chunk_size=4,
        block_size=4,
        num_blocks=64,
        device="cpu",
    )
    assert [row["label"] for row in result["rows"]] == ["whole-prompt prefill", "chunked prefill"]
    for row in result["rows"]:
        assert row["matches_reference"], row["agreement"]
        assert row["post_arrival_max_gap_s"] > 0
        assert row["stall_vs_pre_arrival_step"] > 0
    whole, chunked = result["rows"]
    # The structural budget claim: chunking bounds prompt tokens cached per engine step.
    assert whole["max_prefill_tokens_in_one_step"] == 16
    assert chunked["max_prefill_tokens_in_one_step"] <= 4 * 2  # chunk x concurrent prefills


def test_preemption_starved_pool_requires_real_preemptions(tiny_runtime) -> None:
    result = run_preemption_starved_pool(
        tiny_runtime,
        tiny_runtime,
        prompts=[(1, 4, 7), (2, 5, 8), (3, 6, 1)],
        max_new_tokens=6,
        block_size=4,
        num_blocks=3,
        device="cpu",
    )
    on, reserve = result["rows"]
    assert on["label"] == "preemption on"
    assert on["preemptions"] > 0
    assert on["matches_reference"], on["agreement"]
    assert reserve["preemptions"] == 0
    assert reserve["matches_reference"], reserve["agreement"]


def test_preemption_starved_pool_rejects_a_preemption_free_run(tiny_runtime) -> None:
    with pytest.raises(ValueError, match="zero preemptions"):
        run_preemption_starved_pool(
            tiny_runtime,
            tiny_runtime,
            prompts=[(1, 4, 7)],
            max_new_tokens=2,
            block_size=4,
            num_blocks=64,
            device="cpu",
        )


def test_speculative_batch1_reports_acceptance_profile(tiny_runtime) -> None:
    result = run_speculative_batch1(
        tiny_runtime,
        tiny_runtime,
        prompt_ids=(1, 5, 2, 1, 5, 2),
        max_new_tokens=8,
        speculative=SpeculativeDecodingConfig(max_draft_tokens=3, max_ngram_size=3),
        block_size=4,
        num_blocks=64,
        device="cpu",
        warmup=0,
        iters=1,
    )
    on, off = result["rows"]
    assert on["label"] == "speculative on"
    assert on["matches_reference"], on["agreement"]
    assert off["matches_reference"], off["agreement"]
    assert result["verify_steps"] >= 1
    assert result["mean_tokens_per_verify_step"] is not None
