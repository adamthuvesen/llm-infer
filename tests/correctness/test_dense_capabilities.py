"""Dense backend capability guards and logit-soft-cap forward contract."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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


def test_dense_rejects_prefix_group_id(tmp_path: Path) -> None:
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))
    engine = InferenceEngine(model, block_size=8, num_blocks=32)

    with pytest.raises(ValueError, match="prefix_group_id"):
        engine.add_request(
            Request(
                "sibling",
                [1, 4, 7],
                max_new_tokens=2,
                eos_token_ids=frozenset(),
                prefix_group_id="g0",
            )
        )


def test_dense_rejects_speculative_init(tmp_path: Path) -> None:
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))

    with pytest.raises(ValueError, match="speculative decoding"):
        InferenceEngine(
            model,
            block_size=8,
            num_blocks=32,
            speculative=SpeculativeDecodingConfig(max_draft_tokens=2, max_ngram_size=3),
        )


def test_dense_rejects_preemption(tmp_path: Path) -> None:
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))

    with pytest.raises(ValueError, match="preemption"):
        InferenceEngine(model, block_size=8, num_blocks=32, preemption=True)


def test_history_released_after_request_finish(tmp_path: Path) -> None:
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))
    engine = InferenceEngine(model, block_size=8, num_blocks=32)
    engine.add_request(Request("one", [1, 4, 7], max_new_tokens=1, eos_token_ids=frozenset({99})))

    engine.run()

    assert model._history_by_table == {}
