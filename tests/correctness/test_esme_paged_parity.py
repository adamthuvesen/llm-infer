"""Esme paged-KV decode parity against the ``PretrainBundleModel.logits()`` oracle.

The hard gate from AGENTS.md: before any speed claim, the Esme paged-KV path must reproduce
the full-recompute reference output. "Exact" means **token ids / argmax agree token-for-token**
under fp32 — that is the falsifiable contract the repo holds backends to ("exact token ids on the
single-request unit path"). These checks drive the *same* engine prefill/decode primitives the
serving loop uses (``prefill`` / ``decode_one`` / ``decode_many``) and assert their next-token
argmax — and the underlying logits — match ``PretrainBundleModel.logits()`` for both
single-sequence and batched (multiple concurrent sequences) decode.

The logit values are additionally held close. The cache write is a pure side effect; the only
non-bit-exact op is the BLAS reduction-order difference between a full-sequence prefill matmul
and the gathered-history decode matmul (the same class of effect the Qwen prefill docstring and
``docs/fixture-format.md`` document). On the synthetic bundle this stays at fp32 bit-noise
(~1e-5); on the real Esme-214M-Chat weights it reaches ~1e-4, still ~24x below the model's
tightest observed top-2 decision margin (2.6e-3), so it can never flip a greedy argmax — which
is why the token-id assertion is the gate and the logit tolerance is the documented floor.

The tiny synthetic bundle runs everywhere with zero spend. The real Esme-214M-Chat bundle is
exercised too when ``ESME_BUNDLE_PATH`` points at it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.decode import greedy_decode
from llm_infer.model.pretrain_bundle import PretrainBundleModel

from .test_pretrain_bundle import _write_tiny_bundle

# Synthetic bundle: fp32 bit-noise, essentially exact.
SYNTH_RTOL = 1e-5
SYNTH_ATOL = 1e-5
# Real bundle: fp32 BLAS reduction-order noise between prefill and gathered-history decode
# matmuls. Measured max 1.08e-4 across the checked prompts; the floor sits well above that and
# hundreds of times below any real decision margin, so it launders reduction-order noise but
# never a real bug, and the token-id / argmax assertions remain the actual gate.
REAL_RTOL = 1e-3
REAL_ATOL = 5e-4


def _assert_logits_and_argmax_match(
    paged: torch.Tensor, reference: torch.Tensor, *, rtol: float, atol: float
) -> None:
    """Argmax must match token-for-token (the gate); logits close within the documented floor."""
    assert paged.shape == reference.shape
    torch.testing.assert_close(paged.argmax(dim=-1), reference.argmax(dim=-1))
    torch.testing.assert_close(paged, reference, rtol=rtol, atol=atol)


def _new_cache(model: PretrainBundleModel, *, block_size: int, num_blocks: int) -> PagedKVCache:
    return PagedKVCache(
        num_layers=model.num_layers,
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype=model.dtype,
    )


def _paged_single_sequence_logits(
    model: PretrainBundleModel, token_ids: list[int], *, block_size: int
) -> torch.Tensor:
    """Prefill the prompt, then decode each continuation token through the paged path.

    Returns the next-token logits at every position, stacked to match ``logits()`` shape so the
    two can be compared row-for-row.
    """
    num_blocks = -(-len(token_ids) // block_size) + 4
    cache = _new_cache(model, block_size=block_size, num_blocks=num_blocks)
    table = cache.new_request()

    rows: list[torch.Tensor] = [model.prefill(token_ids[:1], cache, table)]
    for token in token_ids[1:]:
        rows.append(model.decode_one(cache, table, int(token)))
    return torch.stack(rows)


def _real_esme_prompts(model: PretrainBundleModel) -> list[list[int]]:
    """A few short ragged prompts of valid token ids for the real bundle."""
    vocab = model.config.vocab_size
    prompts = [
        [1, 5, 9, 13, 21],
        [2, 7, 11],
        [3, 8, 16, 24],
    ]
    if any(token >= vocab for prompt in prompts for token in prompt):
        raise ValueError(f"fixed test prompts exceed bundle vocab size {vocab}")
    return prompts


@pytest.fixture
def tiny_model(tmp_path: Path) -> PretrainBundleModel:
    return PretrainBundleModel.load(_write_tiny_bundle(tmp_path, logit_soft_cap=7.5))


@pytest.fixture
def tiny_model_qk_norm(tmp_path: Path) -> PretrainBundleModel:
    return PretrainBundleModel.load(_write_tiny_bundle(tmp_path, qk_norm=True, logit_soft_cap=None))


def test_esme_paged_single_sequence_matches_reference(tiny_model: PretrainBundleModel) -> None:
    token_ids = [1, 4, 7, 2, 9, 5]
    paged = _paged_single_sequence_logits(tiny_model, token_ids, block_size=4)
    reference = tiny_model.logits(token_ids)

    _assert_logits_and_argmax_match(paged, reference, rtol=SYNTH_RTOL, atol=SYNTH_ATOL)


def test_esme_paged_single_sequence_matches_reference_qk_norm(
    tiny_model_qk_norm: PretrainBundleModel,
) -> None:
    """QK-norm K is written post-norm/post-RoPE; the paged decode must still match exactly."""
    token_ids = [1, 4, 7, 2, 9, 5]
    paged = _paged_single_sequence_logits(tiny_model_qk_norm, token_ids, block_size=4)
    reference = tiny_model_qk_norm.logits(token_ids)

    _assert_logits_and_argmax_match(paged, reference, rtol=SYNTH_RTOL, atol=SYNTH_ATOL)


def test_esme_paged_prefill_matches_reference_last_row(tiny_model: PretrainBundleModel) -> None:
    token_ids = [1, 4, 7, 2, 9]
    cache = _new_cache(tiny_model, block_size=4, num_blocks=8)
    table = cache.new_request()

    last = tiny_model.prefill(token_ids, cache, table)
    reference = tiny_model.logits(token_ids)[-1]

    torch.testing.assert_close(last, reference, rtol=SYNTH_RTOL, atol=SYNTH_ATOL)
    assert table.length == len(token_ids)


def test_esme_paged_batched_decode_matches_per_sequence_reference(
    tiny_model: PretrainBundleModel,
) -> None:
    """decode_many over concurrent ragged sequences == per-sequence full-recompute reference."""
    prompts = [[1, 4, 7], [2, 9], [3, 5, 8, 6]]
    block_size = 4
    cache = _new_cache(tiny_model, block_size=block_size, num_blocks=64)

    tables = [cache.new_request() for _ in prompts]
    histories = [list(prompt) for prompt in prompts]
    for table, prompt in zip(tables, prompts, strict=True):
        tiny_model.prefill(prompt, cache, table)

    # Two concurrent batched decode steps; each request appends a token and the batched logits
    # row must equal that request's own full-recompute next-token logits.
    next_tokens = [2, 5, 1]
    for _ in range(2):
        batched = tiny_model.decode_many(cache, tables, next_tokens)
        for index, (history, token) in enumerate(zip(histories, next_tokens, strict=True)):
            history.append(token)
            reference = tiny_model.logits(history)[-1]
            _assert_logits_and_argmax_match(
                batched[index], reference, rtol=SYNTH_RTOL, atol=SYNTH_ATOL
            )
        next_tokens = [int(torch.argmax(batched[i])) for i in range(len(prompts))]


def test_esme_paged_batched_decode_equals_single_sequence(
    tiny_model: PretrainBundleModel,
) -> None:
    """A request decoded in a batch gets identical logits to being decoded alone."""
    prompt = [1, 4, 7, 2]
    block_size = 4

    solo_cache = _new_cache(tiny_model, block_size=block_size, num_blocks=16)
    solo_table = solo_cache.new_request()
    tiny_model.prefill(prompt, solo_cache, solo_table)
    solo = tiny_model.decode_one(solo_cache, solo_table, 9)

    batch_cache = _new_cache(tiny_model, block_size=block_size, num_blocks=64)
    tables = [batch_cache.new_request() for _ in range(3)]
    for table in tables:
        tiny_model.prefill(prompt, batch_cache, table)
    batched = tiny_model.decode_many(batch_cache, tables, [9, 9, 9])

    for index in range(3):
        torch.testing.assert_close(batched[index], solo, rtol=SYNTH_RTOL, atol=SYNTH_ATOL)


def _require_real_esme() -> PretrainBundleModel:
    bundle = os.environ.get("ESME_BUNDLE_PATH")
    if bundle is None:
        pytest.skip("set ESME_BUNDLE_PATH to run the real Esme-214M-Chat paged parity check")
    return PretrainBundleModel.load(Path(bundle), dtype=torch.float32)


def test_esme_paged_single_sequence_matches_reference_real_bundle() -> None:
    model = _require_real_esme()
    token_ids = _real_esme_prompts(model)[0]
    paged = _paged_single_sequence_logits(model, token_ids, block_size=16)
    reference = model.logits(token_ids)

    _assert_logits_and_argmax_match(paged, reference, rtol=REAL_RTOL, atol=REAL_ATOL)


def test_esme_paged_greedy_token_ids_match_reference_real_bundle() -> None:
    """The actual repo gate: paged greedy decode == full-recompute greedy, token-for-token."""
    model = _require_real_esme()
    prompt = _real_esme_prompts(model)[0]
    max_new_tokens = 24

    reference = greedy_decode(model, list(prompt), max_new_tokens=max_new_tokens, eos_token_ids={2})

    cache = _new_cache(model, block_size=16, num_blocks=16)
    table = cache.new_request()
    next_token = int(torch.argmax(model.prefill(prompt, cache, table)))
    paged: list[int] = []
    for _ in range(max_new_tokens):
        paged.append(next_token)
        if next_token == 2:
            break
        next_token = int(torch.argmax(model.decode_one(cache, table, next_token)))

    assert paged == reference


def test_esme_paged_batched_decode_matches_reference_real_bundle() -> None:
    model = _require_real_esme()
    prompts = _real_esme_prompts(model)
    block_size = 16
    cache = _new_cache(model, block_size=block_size, num_blocks=256)

    tables = [cache.new_request() for _ in prompts]
    histories = [list(prompt) for prompt in prompts]
    for table, prompt in zip(tables, prompts, strict=True):
        model.prefill(prompt, cache, table)

    next_tokens = [int(torch.argmax(model.logits(history)[-1])) for history in histories]
    for _ in range(3):
        batched = model.decode_many(cache, tables, next_tokens)
        for index, (history, token) in enumerate(zip(histories, next_tokens, strict=True)):
            history.append(token)
            reference = model.logits(history)[-1]
            _assert_logits_and_argmax_match(
                batched[index], reference, rtol=REAL_RTOL, atol=REAL_ATOL
            )
        next_tokens = [int(torch.argmax(batched[i])) for i in range(len(prompts))]
