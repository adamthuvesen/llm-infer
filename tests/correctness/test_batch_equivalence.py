"""benchmark batch-correctness suite: N-way batching stays token-for-token correct.

paged-cache proved the two-request slice; the benchmark batches the whole case set at once, so
this is the gate that the *full* batch the benchmark times is still exactly correct. Same
``torch_naive`` backend (materialized softmax, no fast kernel), so the bar is EXACT — a
divergence is a batching/position/block bug, not FP noise. Two claims:

* every request in one all-cases batch decodes its committed golden continuation
  token-for-token, for the full generation;
* batched == serial: each request's tokens in the batch equal the same request run alone.

CPU-runnable (the fixture's fp32 dtype), zero GPU spend — the local complement to the
GPU throughput benchmark in ``scripts/modal_benchmark.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_infer.serving import InferenceEngine, Request

GOLDEN_PATH = Path(__file__).parent / "goldens" / "qwen2_5_coder_3b_instruct_cot.json"
FIXTURE = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
EOS = frozenset(FIXTURE["decoding"]["eos_token_ids"])
MAX_NEW = FIXTURE["decoding"]["max_new_tokens"]
CASES = FIXTURE["cases"]


@pytest.fixture(scope="module")
def model():
    """The engine in the fixture's pinned dtype — one load shared across this module."""
    from llm_infer.model.qwen import QwenModel

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[FIXTURE["environment"]["dtype"]]
    return QwenModel.load(dtype=dtype)


def _run_single(model, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    engine = InferenceEngine(model, block_size=128, num_blocks=8)
    engine.add_request(Request("only", list(prompt_ids), max_new_tokens, EOS))
    return engine.run()["only"]


def _first_divergence(ours: list[int], expected: list[int]) -> int | None:
    return next((i for i, (a, b) in enumerate(zip(ours, expected, strict=False)) if a != b), None)


@pytest.mark.slow
def test_all_cases_batched_match_goldens(model) -> None:
    """Every case in one batch decodes its full golden continuation, token-for-token."""
    # Pool sized to hold all cases at once: each prompt is < 2 blocks at block_size 128, so
    # 2 blocks per case plus headroom admits the whole batch simultaneously.
    num_blocks = 2 * len(CASES) + 4
    engine = InferenceEngine(model, block_size=128, num_blocks=num_blocks)
    for case in CASES:
        engine.add_request(Request(case["case_id"], list(case["prompt_ids"]), MAX_NEW, EOS))
    out = engine.run()

    for case in CASES:
        expected = case["continuation_ids"]
        got = out[case["case_id"]]
        div = _first_divergence(got, expected)
        assert got == expected, (
            f"{case['case_id']}: batched decode diverged from golden at step {div} "
            f"(ours={got}, golden={expected})"
        )


@pytest.mark.slow
def test_all_cases_batched_equals_serial(model) -> None:
    """Each request's batched tokens equal that request run alone — batching adds no drift."""
    steps = 8
    serial = {case["case_id"]: _run_single(model, case["prompt_ids"], steps) for case in CASES}

    engine = InferenceEngine(model, block_size=128, num_blocks=2 * len(CASES) + 4)
    for case in CASES:
        engine.add_request(Request(case["case_id"], list(case["prompt_ids"]), steps, EOS))
    batched = engine.run()

    for case in CASES:
        cid = case["case_id"]
        div = _first_divergence(batched[cid], serial[cid])
        assert batched[cid] == serial[cid], (
            f"{cid}: batched != serial at step {div} (batched={batched[cid]}, serial={serial[cid]})"
        )
