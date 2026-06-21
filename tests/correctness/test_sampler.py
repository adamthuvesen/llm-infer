"""The sampler's honesty bar: at temperature 0 it equals the proven greedy path, exactly.

The engine now routes token selection through :class:`Sampler`. The rollout uses
temperature/top-p multinomial sampling (TIMING-only — sampled tokens are not asserted
across engines, since llm-infer and vLLM use different RNG). What *is* asserted here:

* the fast tensor-level checks — temperature 0 == argmax (single row and batched), top-p
  truncates to the nucleus, a pinned seed is reproducible, bad params are rejected loudly;
* the integration bar — the engine driven entirely through ``Sampler(temperature=0)``
  (prefill ``sample`` + batched-decode ``sample_many``) reproduces the committed HF greedy
  goldens token-for-token. That ties the sampling code path back to the Phase A/B oracle.

CPU-runnable, fp32 — the same exact bar the greedy oracle holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_infer.serving import InferenceEngine, Request
from llm_infer.serving.sampler import Sampler, greedy

GOLDEN_PATH = Path(__file__).parent / "goldens" / "qwen2_5_coder_3b_instruct_cot.json"
FIXTURE = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
EOS = frozenset(FIXTURE["decoding"]["eos_token_ids"])
MAX_NEW = FIXTURE["decoding"]["max_new_tokens"]
CASES = FIXTURE["cases"]


def test_temperature_zero_equals_argmax_single_and_batched() -> None:
    """temperature 0 returns the argmax — token-for-token greedy, per row and batched."""
    torch.manual_seed(0)
    logits = torch.randn(7, 4096)
    sampler = Sampler(temperature=0.0)

    assert sampler.sample_many(logits) == torch.argmax(logits, dim=-1).tolist()
    for row in logits:
        assert sampler.sample(row) == greedy(row) == int(torch.argmax(row).item())


def test_temperature_zero_breaks_ties_like_argmax() -> None:
    """On an exact tie, temperature 0 picks the first max index, exactly as ``greedy``."""
    logits = torch.tensor([1.0, 1.0, 0.5])
    assert Sampler(temperature=0.0).sample(logits) == greedy(logits) == 0


def test_top_p_truncates_to_the_nucleus() -> None:
    """A peaked distribution with a tight nucleus collapses to its single dominant token."""
    logits = torch.tensor([[10.0, 0.0, -10.0]])  # softmax ~ [0.99995, 4.5e-5, 2e-9]
    sampler = Sampler(temperature=1.0, top_p=0.5, seed=0)
    # The top token alone exceeds top_p=0.5, so it is the only one kept: every draw is token 0.
    assert all(sampler.sample_many(logits) == [0] for _ in range(8))


def test_top_p_one_is_a_no_op_full_softmax() -> None:
    """top_p == 1.0 keeps the whole distribution (a no-op nucleus), as the rollout requires."""
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    full = Sampler(temperature=1.0, top_p=1.0)._nucleus_probs(logits)
    expected = torch.softmax(logits.float(), dim=-1)
    assert torch.allclose(full, expected)


def test_seed_is_reproducible() -> None:
    """Two samplers with the same seed draw the same tokens from the same logits."""
    logits = torch.randn(16, 256, generator=torch.Generator().manual_seed(1))
    a = Sampler(temperature=1.0, seed=123).sample_many(logits)
    b = Sampler(temperature=1.0, seed=123).sample_many(logits)
    assert a == b


def test_different_seeds_diverge() -> None:
    """Different seeds explore differently — the point of sampling during rollouts."""
    logits = torch.randn(64, 4096, generator=torch.Generator().manual_seed(2))
    a = Sampler(temperature=1.0, seed=0).sample_many(logits)
    b = Sampler(temperature=1.0, seed=1).sample_many(logits)
    assert a != b


def test_invalid_params_rejected() -> None:
    """Out-of-range temperature/top_p fail loudly rather than silently coercing."""
    with pytest.raises(ValueError):
        Sampler(temperature=-0.1)
    with pytest.raises(ValueError):
        Sampler(top_p=0.0)
    with pytest.raises(ValueError):
        Sampler(top_p=1.5)


@pytest.fixture(scope="module")
def model():
    """The engine in the fixture's pinned dtype — one load shared across this module."""
    from llm_infer.model.qwen import QwenModel

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[FIXTURE["environment"]["dtype"]]
    return QwenModel.load(dtype=dtype)


@pytest.mark.slow
def test_engine_temperature_zero_sampler_matches_goldens(model) -> None:
    """The engine driven through ``Sampler(temperature=0)`` reproduces the greedy goldens.

    Every token here is chosen by the sampler — first tokens via ``sample`` in prefill, the
    rest via ``sample_many`` in the batched decode. Matching the committed HF greedy goldens
    token-for-token proves temperature 0 sampling *is* the greedy path, end to end.
    """
    engine = InferenceEngine(
        model,
        block_size=128,
        num_blocks=2 * len(CASES) + 4,
        sampler=Sampler(temperature=0.0),
    )
    for case in CASES:
        engine.add_request(Request(case["case_id"], list(case["prompt_ids"]), MAX_NEW, EOS))
    out = engine.run()

    for case in CASES:
        got, expected = out[case["case_id"]], case["continuation_ids"]
        diff = next(
            (i for i, (a, b) in enumerate(zip(got, expected, strict=False)) if a != b), None
        )
        assert got == expected, f"{case['case_id']}: temp=0 sampler diverged at step {diff}"
