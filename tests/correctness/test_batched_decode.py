"""benchmark batched-decode correctness: the fused batch decode equals the serial path.

``decode_many`` advances ``B`` running requests in one forward (the throughput win). It is
correct only if each request gets *exactly* the token it would get decoded alone — same
per-request RoPE position, same paged history, same attention. This pins that:

* ``decode_many`` over B requests == ``decode_one`` on each, token-for-token (the unit rule);
* an all-cases batch driven through the engine reproduces the committed goldens (the
  integration rule — this also guards the engine's internal ``decode_many`` path).

Same ``torch_naive`` backend, fp32 — so the rule is exact. A divergence is a batching/position/
gather bug, not FP noise. CPU-runnable, zero GPU spend; the flash backend's batched path is
validated on the A100 by the benchmark's fp32-reference agreement check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
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


def _cache(model, num_blocks: int) -> PagedKVCache:
    return PagedKVCache(
        num_layers=model.num_layers,
        num_blocks=num_blocks,
        block_size=128,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype=model.dtype,
        device="cpu",
    )


def _prefill_first_token(model, cache: PagedKVCache, prompt_ids: list[int]):
    """Prefill one request into ``cache`` and return (block_table, its first greedy token)."""
    table = cache.new_request()
    logits = model.prefill(list(prompt_ids), cache, table)
    return table, int(torch.argmax(logits))


@pytest.mark.slow
def test_decode_many_equals_decode_one_per_request(model) -> None:
    """One batched decode step over all cases == each case decoded alone, token-for-token."""
    # Serial: each request in its own cache — prefill, then a single decode_one.
    serial_second = []
    for case in CASES:
        cache = _cache(model, num_blocks=8)
        table, first = _prefill_first_token(model, cache, case["prompt_ids"])
        logits = model.decode_one(cache, table, first)
        serial_second.append(int(torch.argmax(logits)))

    # Batched: all requests share one cache, prefilled, then ONE decode_many step.
    shared = _cache(model, num_blocks=2 * len(CASES) + 4)
    tables, firsts = [], []
    for case in CASES:
        table, first = _prefill_first_token(model, shared, case["prompt_ids"])
        tables.append(table)
        firsts.append(first)
    batched_logits = model.decode_many(shared, tables, firsts)  # (B, vocab)
    batched_second = [int(torch.argmax(batched_logits[i])) for i in range(len(CASES))]

    assert batched_second == serial_second, (
        f"batched decode != serial decode_one: batched={batched_second} serial={serial_second}"
    )


@pytest.mark.slow
def test_engine_batched_decode_matches_goldens(model) -> None:
    """All cases admitted at once (every step is a real batched decode) reproduce the goldens."""
    engine = InferenceEngine(model, block_size=128, num_blocks=2 * len(CASES) + 4)
    for case in CASES:
        engine.add_request(Request(case["case_id"], list(case["prompt_ids"]), MAX_NEW, EOS))
    out = engine.run()

    for case in CASES:
        got, expected = out[case["case_id"]], case["continuation_ids"]
        diff = next(
            (i for i, (a, b) in enumerate(zip(got, expected, strict=False)) if a != b), None
        )
        assert got == expected, f"{case['case_id']}: batched decode diverged at step {diff}"
