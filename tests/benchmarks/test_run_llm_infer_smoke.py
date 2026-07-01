"""Smoke test that the llm-infer benchmark runner matches greedy decode on CPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_infer.benchmarks.runners import run_llm_infer
from llm_infer.benchmarks.workload import BenchRequest, Workload
from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as _write_tiny_bundle
from llm_infer.model.decode import greedy_decode
from llm_infer.model.pretrain_bundle import PretrainBundleModel


@pytest.fixture
def tiny_dense(tmp_path: Path) -> PretrainBundleModel:
    return PretrainBundleModel.load(_write_tiny_bundle(tmp_path, logit_soft_cap=None))


def test_run_llm_infer_matches_greedy_decode(tiny_dense: PretrainBundleModel) -> None:
    prompt = [1, 4, 7]
    workload = Workload(
        requests=(BenchRequest("req-0", tuple(prompt), "case"),),
        max_new_tokens=3,
        eos_token_ids=frozenset({99}),
        model_id="tiny-dense",
        model_revision=None,
        source="unit",
    )
    expected = greedy_decode(
        tiny_dense,
        prompt,
        max_new_tokens=3,
        eos_token_ids=set(workload.eos_token_ids),
    )

    result = run_llm_infer(
        tiny_dense,
        workload,
        num_blocks=8,
        warmup=0,
        iters=1,
        device="cpu",
    )

    assert result.outputs == {"req-0": expected}
