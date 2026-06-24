"""The sampler's honesty bar: at temperature 0 it equals the proven greedy path, exactly.

The engine now samples **per row** under each request's :class:`SamplingParams`. The rollout
uses temperature/top-p multinomial sampling (TIMING-only — sampled tokens are not asserted
across engines, since llm-infer and vLLM use different RNG). What *is* asserted here:

* the fast tensor-level checks — temperature 0 == argmax, top-k / top-p truncate to the
  allowed set, penalties subtract OpenAI-style, a pinned seed is reproducible, bad params are
  rejected loudly;
* the integration bar — the engine driven entirely through ``SamplingParams(temperature=0)``
  (prefill + batched decode) reproduces the committed HF greedy goldens token-for-token. That
  ties the per-row sampling code path back to the Phase A/B oracle.

CPU-runnable, fp32 — the same exact bar the greedy oracle holds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_infer.serving import InferenceEngine, Request, SamplingParams
from llm_infer.serving.sampler import _apply_top_p, greedy, sample_row

GOLDEN_PATH = Path(__file__).parent / "goldens" / "qwen2_5_coder_3b_instruct_cot.json"
FIXTURE = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
EOS = frozenset(FIXTURE["decoding"]["eos_token_ids"])
MAX_NEW = FIXTURE["decoding"]["max_new_tokens"]
CASES = FIXTURE["cases"]


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_temperature_zero_equals_argmax_per_row() -> None:
    """temperature 0 returns the argmax — token-for-token greedy, with no RNG drawn."""
    torch.manual_seed(0)
    logits = torch.randn(7, 4096)
    params = SamplingParams(temperature=0.0)
    for row in logits:
        token = sample_row(row, params, generated=[], generator=_gen())
        assert token.ndim == 0
        assert int(token) == greedy(row) == int(torch.argmax(row).item())


def test_temperature_zero_breaks_ties_like_argmax() -> None:
    """On an exact tie, temperature 0 picks the first max index, exactly as ``greedy``."""
    logits = torch.tensor([1.0, 1.0, 0.5])
    token = sample_row(logits, SamplingParams(temperature=0.0), generated=[], generator=_gen())
    assert int(token) == greedy(logits) == 0


def test_top_p_truncates_to_the_nucleus() -> None:
    """A peaked distribution with a tight nucleus collapses to its single dominant token."""
    logits = torch.tensor([10.0, 0.0, -10.0])  # softmax ~ [0.99995, 4.5e-5, 2e-9]
    params = SamplingParams(temperature=1.0, top_p=0.5)
    # The top token alone exceeds top_p=0.5, so it is the only one kept: every draw is token 0.
    assert all(
        int(sample_row(logits, params, generated=[], generator=_gen(s))) == 0 for s in range(8)
    )


def test_top_k_never_samples_outside_the_top_k() -> None:
    """With top_k=2, only the two highest-logit tokens are ever drawn, across many seeds."""
    logits = torch.tensor([3.0, 2.5, 1.0, 0.5, 0.0])  # top-2 are indices 0 and 1
    params = SamplingParams(temperature=1.0, top_k=2)
    drawn = {int(sample_row(logits, params, generated=[], generator=_gen(s))) for s in range(64)}
    assert drawn <= {0, 1}


def test_top_p_one_is_a_no_op_full_softmax() -> None:
    """top_p == 1.0 keeps the whole distribution — no token is masked out."""
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    params = SamplingParams(temperature=1.0, top_p=1.0)
    drawn = {int(sample_row(logits, params, generated=[], generator=_gen(s))) for s in range(128)}
    assert drawn == {0, 1, 2, 3}


def test_top_p_drops_the_redundant_token_at_an_exact_boundary() -> None:
    """When the prefix mass reaches exactly top_p, the next token is redundant and is dropped.

    Regression for a ``>`` boundary that kept an extra token when the mass *before* it equalled
    top_p exactly: the nucleus must be the minimal set whose cumulative mass reaches top_p.
    """
    probs = torch.tensor([0.6, 0.4])  # token0 alone reaches top_p=0.6 exactly
    kept = _apply_top_p(probs, top_p=0.6)
    assert float(kept[1]) == 0.0  # token1 is redundant at the exact boundary -> dropped
    assert float(kept[0]) == 1.0  # and the nucleus renormalizes to the single kept token


def test_non_finite_params_rejected() -> None:
    """nan/inf slip past the ``<``/``<=`` range checks, so they must be rejected explicitly."""
    for kwargs in (
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"top_p": float("nan")},
        {"presence_penalty": float("inf")},
        {"frequency_penalty": float("nan")},
    ):
        with pytest.raises(ValueError):
            SamplingParams(**kwargs)


def test_frequency_penalty_pushes_off_a_repeated_token() -> None:
    """A token already generated many times is suppressed once frequency_penalty is high."""
    logits = torch.tensor([5.0, 0.0, 0.0])  # token 0 dominates with no penalty
    generated = [0] * 10
    penalized = SamplingParams(temperature=1.0, frequency_penalty=2.0)
    drawn = {
        int(sample_row(logits, penalized, generated=generated, generator=_gen(s)))
        for s in range(64)
    }
    # 5.0 - 2.0*10 = -15 for token 0; tokens 1/2 at 0.0 now dominate.
    assert drawn <= {1, 2}


def test_presence_penalty_is_flat_not_count_scaled() -> None:
    """Presence subtracts a single flat amount no matter how many times a token appeared."""
    logits = torch.tensor([1.0, 0.0, 0.0])
    once = sample_row(logits, SamplingParams(temperature=1e-6, presence_penalty=2.0), [0], _gen(0))
    many = sample_row(
        logits, SamplingParams(temperature=1e-6, presence_penalty=2.0), [0] * 9, _gen(0)
    )
    # Flat penalty: 1.0 - 2.0 = -1.0 for token 0 either way, so the argmax-like draw matches.
    assert int(once) == int(many)


def test_seed_is_reproducible() -> None:
    """Same params + seed draw the same token from the same logits."""
    logits = torch.randn(256, generator=_gen(1))
    params = SamplingParams(temperature=1.0, seed=123)
    a = sample_row(logits, params, generated=[], generator=_gen(123))
    b = sample_row(logits, params, generated=[], generator=_gen(123))
    assert torch.equal(a, b)


def test_different_seeds_diverge() -> None:
    """Different seeds explore differently — the point of sampling during rollouts."""
    logits = torch.randn(4096, generator=_gen(2))
    params = SamplingParams(temperature=1.0)
    draws = [int(sample_row(logits, params, generated=[], generator=_gen(s))) for s in range(8)]
    assert len(set(draws)) > 1


def test_invalid_params_rejected() -> None:
    """Out-of-range params fail loudly rather than silently coercing."""
    with pytest.raises(ValueError):
        SamplingParams(temperature=-0.1)
    with pytest.raises(ValueError):
        SamplingParams(top_p=0.0)
    with pytest.raises(ValueError):
        SamplingParams(top_p=1.5)
    with pytest.raises(ValueError):
        SamplingParams(top_k=-1)
    with pytest.raises(ValueError):
        SamplingParams(presence_penalty=3.0)
    with pytest.raises(ValueError):
        SamplingParams(frequency_penalty=-3.0)


@pytest.fixture(scope="module")
def model():
    """The engine in the fixture's pinned dtype — one load shared across this module."""
    from llm_infer.model.qwen import QwenModel

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[FIXTURE["environment"]["dtype"]]
    return QwenModel.load(dtype=dtype)


@pytest.mark.slow
def test_engine_temperature_zero_matches_goldens(model) -> None:
    """The engine driven through ``SamplingParams(temperature=0)`` reproduces the greedy goldens.

    Every token here is chosen by the per-row sampler — first tokens in prefill, the rest in the
    batched decode. Matching the committed HF greedy goldens token-for-token proves temperature 0
    sampling *is* the greedy path, end to end.
    """
    engine = InferenceEngine(
        model,
        block_size=128,
        num_blocks=2 * len(CASES) + 4,
        default_sampling=SamplingParams(temperature=0.0),
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
