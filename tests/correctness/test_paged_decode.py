"""paged-cache correctness: the paged/cached/batched path reproduces the reference output.

Same model weights, same ``torch_naive`` backend (materialized softmax over gathered
K/V — no fast kernel), so this must be EXACT. The four claims:

* cached/paged single-request decode == the committed golden continuation (which is HF
  full-recompute greedy), token-for-token, for the full 40-token generation;
* the cached path == the engine's own full-recompute path (:meth:`QwenModel.logits`
  via ``greedy_decode``) — the same engine, two code paths, agreeing;
* batched == serial: two requests decoded together give each the same tokens as alone;
* the vertical slice runs end to end — admit 2, prefill both, decode, one finishes and
  frees its blocks, a third is admitted into the freed budget, all complete correctly.

Zero divergence is required. The only non-bit-exact op between cached and full-recompute
is the new position's MLP down-projection (one row vs a growing batch — a BLAS
reduction-order effect, ~1e-5 relative on real weights), far below the ~0.4-logit
greedy gaps the fixture doc documents, so the argmax does not flip. A divergence here
would point at a mask / position-id / block-gather bug, not FP noise — trace it, don't
paper over it with a tolerance.
"""

from __future__ import annotations

import json

import pytest
import torch

from llm_infer.fixtures import QWEN_COT_GOLDEN
from llm_infer.model.decode import greedy_decode
from llm_infer.model.qwen import QwenModel
from llm_infer.serving import InferenceEngine, Request

FIXTURE = json.loads(QWEN_COT_GOLDEN.read_text(encoding="utf-8"))
EOS = frozenset(FIXTURE["decoding"]["eos_token_ids"])
MAX_NEW = FIXTURE["decoding"]["max_new_tokens"]
CASES = FIXTURE["cases"]
CASES_BY_ID = {case["case_id"]: case for case in CASES}


@pytest.fixture(scope="module")
def model() -> QwenModel:
    """The engine in the fixture's pinned dtype — one load shared across this module."""
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[FIXTURE["environment"]["dtype"]]
    return QwenModel.load(dtype=dtype)


def _run_single(
    model: QwenModel,
    prompt_ids: list[int],
    max_new_tokens: int,
    *,
    block_size: int = 128,
    num_blocks: int = 8,
) -> list[int]:
    """Greedy-generate one request through the paged engine and return its tokens."""
    engine = InferenceEngine(model, block_size=block_size, num_blocks=num_blocks)
    engine.add_request(Request("only", list(prompt_ids), max_new_tokens, EOS))
    return engine.run()["only"]


def _first_divergence(ours: list[int], expected: list[int]) -> int | None:
    return next((i for i, (a, b) in enumerate(zip(ours, expected, strict=False)) if a != b), None)


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_cached_paged_decode_matches_golden(model: QwenModel, case: dict) -> None:
    """Full 40-token paged/cached decode == the committed HF full-recompute golden."""
    out = _run_single(model, case["prompt_ids"], MAX_NEW)
    expected = case["continuation_ids"]
    div = _first_divergence(out, expected)
    assert out == expected, (
        f"{case['case_id']}: paged decode diverged from golden at step {div} "
        f"(ours={out}, golden={expected})"
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_cached_matches_full_recompute_path(model: QwenModel, case: dict) -> None:
    """The cached path and the engine's own full-recompute path agree (same engine, two paths)."""
    steps = 6
    cached = _run_single(model, case["prompt_ids"], steps)
    recompute = greedy_decode(
        model, list(case["prompt_ids"]), max_new_tokens=steps, eos_token_ids=set(EOS)
    )
    div = _first_divergence(cached, recompute)
    assert cached == recompute, (
        f"{case['case_id']}: cached vs full-recompute diverged at step {div} "
        f"(cached={cached}, recompute={recompute})"
    )


@pytest.mark.slow
def test_batched_decode_equals_serial(model: QwenModel) -> None:
    """Two requests decoded together yield each the same tokens as run alone."""
    steps = 6
    first, second = CASES[0], CASES[1]
    serial_first = _run_single(model, first["prompt_ids"], steps)
    serial_second = _run_single(model, second["prompt_ids"], steps)

    engine = InferenceEngine(model, block_size=128, num_blocks=16)
    engine.add_request(Request("first", list(first["prompt_ids"]), steps, EOS))
    engine.add_request(Request("second", list(second["prompt_ids"]), steps, EOS))
    batched = engine.run()

    assert batched["first"] == serial_first
    assert batched["second"] == serial_second


@pytest.mark.slow
def test_vertical_slice_admits_third_after_one_finishes(model: QwenModel) -> None:
    """admit 2 -> prefill both -> decode -> one finishes -> admit a third -> continue."""
    r1_case = CASES_BY_ID["single_table_count"]  # prompt 110 -> 2 blocks at block_size 64
    r2_case = CASES_BY_ID["two_table_join"]  # prompt 152 -> 3 blocks
    r3_case = CASES_BY_ID["single_table_group_by"]  # prompt 106 -> 2 blocks

    # Pool holds exactly r1 + r2; r3 must wait until r1 finishes and frees its blocks.
    engine = InferenceEngine(model, block_size=64, num_blocks=5)
    r1 = Request("r1", list(r1_case["prompt_ids"]), 3, EOS)
    r2 = Request("r2", list(r2_case["prompt_ids"]), 6, EOS)
    r3 = Request("r3", list(r3_case["prompt_ids"]), 4, EOS)
    for request in (r1, r2, r3):
        engine.add_request(request)

    steps = []
    while engine.scheduler.has_work():
        steps.append(engine.step())

    # Step 0 admits and prefills the first two only; the third is held back.
    assert steps[0].admitted == ["r1", "r2"]
    assert set(steps[0].tokens) == {"r1", "r2"}
    assert "r3" not in steps[0].admitted

    r1_finish_step = next(i for i, s in enumerate(steps) if "r1" in s.finished)
    r3_admit_step = next(i for i, s in enumerate(steps) if "r3" in s.admitted)
    # r3 enters the step right after r1 finishes and frees its budget — never before.
    assert r3_admit_step == r1_finish_step + 1
    assert all("r3" not in s.admitted for s in steps[: r1_finish_step + 1])

    # Every request completed with the right tokens (golden prefixes — no early EOS).
    assert r1.finished and r2.finished and r3.finished
    assert r1.generated == r1_case["continuation_ids"][:3]
    assert r2.generated == r2_case["continuation_ids"][:6]
    assert r3.generated == r3_case["continuation_ids"][:4]

    # The pool was fully reclaimed once everything finished.
    assert engine.cache.allocator.num_free == engine.cache.allocator.num_blocks
