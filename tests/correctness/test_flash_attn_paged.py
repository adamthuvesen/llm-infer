"""flash-attn correctness: the flash-attn backend reproduces the golden under the tie rule.

The fused flash-attn kernel runs in bf16 with a different fp32 reduction order than the
``torch_naive`` reference, so a genuine near-tie greedy step can flip — exactly the
generate()-vs-recompute effect documented for reference check in docs/fixture-format.md. The rule
here is therefore not bit-exact tokens but **token-for-token vs the golden except at a
genuine numerical tie**, with every accepted divergence traced to a tie by recomputing
the step with the fp32 reference path (see ``tie_tolerance.py``). A divergence at a
non-tie step is a FAIL — a real kernel/layout bug.

CUDA-only: flash-attn needs a GPU build, so the whole module is skipped off CUDA. It is
run on the target GPU (Modal A100) via ``scripts/modal_reference_check.py``; the CPU reference check
(``torch_naive``, exact, unchanged) stays the local check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_infer.model.qwen import QwenModel
from llm_infer.serving import InferenceEngine, Request

from .tie_tolerance import DEFAULT_TIE_TOLERANCE, compare_under_tie_tolerance

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="flash-attn requires CUDA; run on the target GPU"
)

GOLDEN_PATH = Path(__file__).parent / "goldens" / "qwen2_5_coder_3b_instruct_cot.json"
FIXTURE = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
EOS = frozenset(FIXTURE["decoding"]["eos_token_ids"])
MAX_NEW = FIXTURE["decoding"]["max_new_tokens"]
CASES = FIXTURE["cases"]


@pytest.fixture(scope="module")
def flash_model() -> QwenModel:
    """The engine on the flash-attn backend, bf16 on CUDA — the fast path under test."""
    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention

    return QwenModel.load(dtype=torch.bfloat16, backend=FlashAttnPagedAttention(), device="cuda")


@pytest.fixture(scope="module")
def reference_model() -> QwenModel:
    """The fp32 full-recompute reference (``torch_naive``) on CUDA — the tie-test reference."""
    return QwenModel.load(dtype=torch.float32, device="cuda")


def _run_flash(model: QwenModel, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    """Greedy-generate one request through the paged engine on the flash backend."""
    engine = InferenceEngine(model, block_size=128, num_blocks=8, device="cuda")
    engine.add_request(Request("only", list(prompt_ids), max_new_tokens, EOS))
    return engine.run()["only"]


def _run_flash_batch(
    model: QwenModel, cases: list[dict], max_new_tokens: int
) -> dict[str, list[int]]:
    """Greedy-generate several requests together through ONE paged engine (fused ragged decode)."""
    engine = InferenceEngine(model, block_size=128, num_blocks=64, device="cuda")
    for case in cases:
        engine.add_request(Request(case["case_id"], list(case["prompt_ids"]), max_new_tokens, EOS))
    return engine.run()


@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_flash_decode_matches_golden_under_tie_tolerance(
    flash_model: QwenModel, reference_model: QwenModel, case: dict
) -> None:
    """flash-attn paged decode == golden, except at genuine ties traced to the reference."""
    out = _run_flash(flash_model, case["prompt_ids"], MAX_NEW)
    result = compare_under_tie_tolerance(
        reference_model,
        case["prompt_ids"],
        out,
        case["continuation_ids"],
        tolerance=DEFAULT_TIE_TOLERANCE,
    )
    if result.divergence is not None:
        d = result.divergence
        print(
            f"\n[tie accepted] {case['case_id']} step {d.step}: flash={d.fast_token} "
            f"golden={d.golden_token} reference top-2 gap={d.reference_gap:.6g} "
            f"(<= tol {DEFAULT_TIE_TOLERANCE:g}); flash_logit={d.fast_logit:.6g} "
            f"golden_logit={d.golden_logit:.6g}"
        )
    assert result.ok, f"{case['case_id']}: {result.failure}"


@pytest.mark.slow
def test_flash_multi_request_ragged_decode_matches_golden(
    flash_model: QwenModel, reference_model: QwenModel
) -> None:
    """Several different-length requests in ONE fused decode each match their golden under ties.

    The single-request test exercises a uniform batch; the riskier path is the **packed varlen**
    decode, where requests of different cached lengths are read from the paged cache together with
    per-request offsets/cu_seqlens. A bug there (wrong cu_seqlens, a swapped row, a misindexed
    block) flips tokens for one request while leaving others correct. We decode all golden cases
    together (their differing prompt lengths make the batch ragged from the first step) and assert
    each request's output still matches its own golden token-for-token, ties excepted.
    """
    outputs = _run_flash_batch(flash_model, CASES, MAX_NEW)
    assert len(outputs) == len(CASES)
    for case in CASES:
        result = compare_under_tie_tolerance(
            reference_model,
            case["prompt_ids"],
            outputs[case["case_id"]],
            case["continuation_ids"],
            tolerance=DEFAULT_TIE_TOLERANCE,
        )
        assert result.ok, f"{case['case_id']} (ragged batch): {result.failure}"
