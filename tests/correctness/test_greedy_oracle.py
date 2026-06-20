"""THE correctness oracle: single-request greedy decode, token-for-token vs HuggingFace.

Every attention backend — now and forever — is validated against this. The committed
golden fixture holds HF greedy continuations for the pinned model; here the llm-infer
engine decodes the same prompt ids through the ``torch_naive`` reference backend and
must produce the identical token ids. No re-running HF: the goldens are the frozen
HF truth (regenerate with ``scripts/generate_goldens.py``).

The model load is module-scoped (one 3B fp32 load for the whole suite) and marked
``slow`` so it can be deselected, but it is part of the default ``tests/correctness``
run — the oracle is the gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_infer.model.config import MODEL_ID, MODEL_REVISION
from llm_infer.model.decode import greedy_decode
from llm_infer.model.qwen import QwenModel

GOLDEN_PATH = Path(__file__).parent / "goldens" / "qwen2_5_coder_3b_instruct_cot.json"


def _load_fixture() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"golden fixture missing at {GOLDEN_PATH}; "
            "regenerate with `uv run python scripts/generate_goldens.py`"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


FIXTURE = _load_fixture()


@pytest.fixture(scope="module")
def engine() -> QwenModel:
    """The llm-infer engine on the reference backend, in the fixture's pinned dtype."""
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[FIXTURE["environment"]["dtype"]]
    return QwenModel.load(dtype=dtype)


def test_fixture_pins_the_model() -> None:
    """The fixture must be the pinned Instruct model — a mismatch invalidates the oracle."""
    assert FIXTURE["model"]["id"] == MODEL_ID
    assert FIXTURE["model"]["revision"] == MODEL_REVISION
    assert FIXTURE["decoding"]["do_sample"] is False
    assert FIXTURE["cases"], "fixture has no cases"


@pytest.mark.slow
@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["case_id"])
def test_greedy_matches_huggingface(engine: QwenModel, case: dict) -> None:
    """llm-infer greedy decode == HF greedy continuation, token for token."""
    eos = set(FIXTURE["decoding"]["eos_token_ids"])
    ours = greedy_decode(
        engine,
        case["prompt_ids"],
        max_new_tokens=FIXTURE["decoding"]["max_new_tokens"],
        eos_token_ids=eos,
    )
    expected = case["continuation_ids"]
    # Pinpoint the first divergence so a failure names the exact step, not just "not equal".
    first_div = next(
        (i for i, (a, b) in enumerate(zip(ours, expected, strict=False)) if a != b),
        None,
    )
    assert ours == expected, (
        f"{case['case_id']}: greedy diverged from HF at step {first_div} "
        f"(ours={ours}, hf={expected})"
    )
