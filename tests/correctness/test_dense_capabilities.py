"""Dense backend capability contract and logit-soft-cap forward contract.

Esme bundles serve through the same real paged-KV / batched-decode path as Qwen, so the engine
advertises the full capability set: paged KV, prefix caching, speculative decode, and
preemption. These tests confirm each capability is accepted at init/add_request and that the
served output still matches the per-sequence ``PretrainBundleModel.logits()`` reference.
"""

from __future__ import annotations

from pathlib import Path

import torch

from llm_infer.model.decode import greedy_decode
from llm_infer.model.interface import DENSE_CAPABILITIES, QWEN_CAPABILITIES
from llm_infer.model.pretrain_bundle import PretrainBundleModel
from llm_infer.serving import InferenceEngine, Request
from llm_infer.serving.speculative import SpeculativeDecodingConfig

from .test_pretrain_bundle import _write_tiny_bundle


def test_logit_soft_cap_changes_logits(tmp_path: Path) -> None:
    uncapped = PretrainBundleModel.load(_write_tiny_bundle(tmp_path / "a", logit_soft_cap=None))
    capped = PretrainBundleModel.load(_write_tiny_bundle(tmp_path / "b", logit_soft_cap=7.5))

    prompt = [1, 4, 7]
    uncapped_logits = uncapped.logits(prompt)[-1]
    capped_logits = capped.logits(prompt)[-1]

    assert not torch.allclose(uncapped_logits, capped_logits)


def test_dense_advertises_paged_kv_capabilities() -> None:
    assert DENSE_CAPABILITIES.paged_kv is True
    assert DENSE_CAPABILITIES == QWEN_CAPABILITIES


def test_dense_accepts_prefix_group_id(tmp_path: Path) -> None:
    """Prefix sharing keeps two siblings equal to per-sequence recompute reference."""
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))
    prompt = [1, 4, 7]
    reference = greedy_decode(model, prompt, max_new_tokens=3, eos_token_ids=set())

    engine = InferenceEngine(model, block_size=8, num_blocks=64)
    for request_id in ("sib-a", "sib-b"):
        engine.add_request(
            Request(
                request_id,
                list(prompt),
                max_new_tokens=3,
                eos_token_ids=frozenset(),
                prefix_group_id="g0",
            )
        )

    outputs = engine.run()
    assert outputs["sib-a"] == reference
    assert outputs["sib-b"] == reference


def test_dense_accepts_speculative_init(tmp_path: Path) -> None:
    """Speculative verification runs on real paged K/V and matches greedy recompute."""
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))
    prompt = [1, 4, 7]
    reference = greedy_decode(model, prompt, max_new_tokens=4, eos_token_ids=set())

    engine = InferenceEngine(
        model,
        block_size=8,
        num_blocks=64,
        speculative=SpeculativeDecodingConfig(max_draft_tokens=2, max_ngram_size=3),
    )
    engine.add_request(Request("spec", list(prompt), max_new_tokens=4, eos_token_ids=frozenset()))

    assert engine.run()["spec"] == reference


def test_dense_accepts_preemption(tmp_path: Path) -> None:
    """Preemption uses recompute-on-resume on the paged path."""
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))
    prompt = [1, 4, 7]
    reference = greedy_decode(model, prompt, max_new_tokens=3, eos_token_ids=set())

    engine = InferenceEngine(model, block_size=8, num_blocks=64, preemption=True)
    engine.add_request(Request("pre", list(prompt), max_new_tokens=3, eos_token_ids=frozenset()))

    assert engine.run()["pre"] == reference
