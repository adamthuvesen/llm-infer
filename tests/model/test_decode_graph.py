"""Decode-graph runner correctness on CPU: bucket selection, padding, and step parity.

``DecodeGraphRunner(mode="eager")`` runs the exact segment functions and static-buffer
``copy_`` flow that the CUDA capture replays, without the capture itself. These tests pin
that flow against the plain planned window step on the tiny bundle: same logits and tokens
with pad rows in play, correct buffer contents after a step, and clean eager fallbacks when
a batch or window cannot run through the buckets. On the GPU the only remaining delta is
graph capture, which the Modal reference gate covers.
"""

from __future__ import annotations

import pytest
import torch

from llm_infer.benchmarks.grouped_decode_graph import EngineOwnedGroupedDecodeGraphRunner
from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.decode_graph import select_bucket
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving import InferenceEngine, Request

_PROMPTS = ([1, 4, 7], [2, 5, 3, 6, 9], [8, 1])


@pytest.fixture()
def runtime(tmp_path):
    bundle = write_tiny_pretrain_bundle(tmp_path)
    return load_model_runtime("esme", bundle_path=bundle)


def test_select_bucket_picks_smallest_covering_size() -> None:
    sizes = (1, 2, 4, 8)
    assert select_bucket(sizes, 1) == 1
    assert select_bucket(sizes, 2) == 2
    assert select_bucket(sizes, 3) == 4
    assert select_bucket(sizes, 8) == 8
    assert select_bucket(sizes, 9) is None
    with pytest.raises(ValueError, match="batch must be"):
        select_bucket(sizes, 0)


def _fresh_cache(model) -> PagedKVCache:
    return PagedKVCache(
        num_layers=model.num_layers,
        num_blocks=32,
        block_size=4,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype=model.dtype,
    )


def _prefill(model, cache) -> tuple[list, torch.Tensor]:
    tables, first_tokens = [], []
    for prompt in _PROMPTS:
        table = cache.new_request()
        logits = model.prefill(list(prompt), cache, table)
        tables.append(table)
        first_tokens.append(torch.argmax(logits))
    return tables, torch.stack(first_tokens)


def _run_window(model, budget: int) -> list[torch.Tensor]:
    """Prefill a fresh cache, then run one planned window; returns per-step logits copies."""
    cache = _fresh_cache(model)
    tables, tokens = _prefill(model, cache)
    plan = model.open_decode_window(cache, tables, budget)
    assert plan is not None
    steps = []
    for _ in range(budget):
        logits = model.decode_window_step(cache, plan, tokens)
        steps.append(logits.clone())
        tokens = torch.argmax(logits, dim=-1)
    return steps


def test_padded_eager_runner_matches_plain_window_step(runtime) -> None:
    """Batch 3 padded to bucket 4: per-step logits match the plain planned path."""
    model = runtime.model
    budget = 4
    model.decode_graphs = None
    plain = _run_window(model, budget)
    model.enable_decode_graphs(capture_sizes=(4, 8), mode="eager")
    padded = _run_window(model, budget)
    for step, (expected, got) in enumerate(zip(plain, padded, strict=True)):
        torch.testing.assert_close(got, expected, msg=f"step {step} logits diverged")
        assert torch.equal(torch.argmax(got, dim=-1), torch.argmax(expected, dim=-1))


def test_static_buffers_hold_step_inputs_and_advance_positions(runtime) -> None:
    """After a step: tokens copied into the bucket prefix, positions advanced, pads contained."""
    model = runtime.model
    runner = model.enable_decode_graphs(capture_sizes=(4,), mode="eager")
    cache = _fresh_cache(model)
    tables, tokens = _prefill(model, cache)
    base_lengths = [table.length for table in tables]
    plan = model.open_decode_window(cache, tables, 2)
    assert plan is not None

    logits = model.decode_window_step(cache, plan, tokens)
    state = runner._buckets[4]
    assert logits.shape == (3, model.config.vocab_size)
    assert logits.data_ptr() == state.logits.data_ptr()  # a static-buffer view, not a copy
    assert state.tokens[:3].tolist() == tokens.tolist()
    assert state.tokens[3:].tolist() == [0]  # pad token rows are never written
    # Positions were loaded from the plan at window open (pads zeroed), then the final
    # segment advanced the whole buffer by one.
    assert state.positions.tolist() == [length + 1 for length in base_lengths] + [1]


def test_engine_outputs_unchanged_with_eager_runner(runtime) -> None:
    """End-to-end engine run with the padded runner equals the plain engine, token for token."""

    def run_engine() -> dict[str, list[int]]:
        engine = InferenceEngine(
            runtime.model,
            block_size=4,
            num_blocks=64,
            capabilities=runtime.capabilities,
            decode_window_size=4,
        )
        for index, prompt in enumerate(_PROMPTS):
            engine.add_request(Request(f"r{index}", list(prompt), 10, frozenset()))
        return engine.run()

    runtime.model.decode_graphs = None
    plain = run_engine()
    runtime.model.enable_decode_graphs(capture_sizes=(4,), mode="eager")
    assert run_engine() == plain


def test_exact_batch_grouped_eager_runner_matches_plain_window(runtime) -> None:
    """The benchmark runner's grouped tranche keeps tiny-bundle logits exact on CPU."""
    model = runtime.model

    def run(grouped: bool) -> list[torch.Tensor]:
        cache = _fresh_cache(model)
        table = cache.new_request()
        logits = model.prefill(list(_PROMPTS[0]), cache, table)
        tokens = torch.argmax(logits).reshape(1)
        if grouped:
            model.decode_graphs = EngineOwnedGroupedDecodeGraphRunner(
                model, cache, batch_size=1, grouped_layers=2, mode="eager"
            )
        else:
            model.decode_graphs = None
        plan = model.open_decode_window(cache, [table], budget=3)
        assert plan is not None
        rows = []
        for _ in range(3):
            logits = model.decode_window_step(cache, plan, tokens)
            rows.append(logits.clone())
            tokens = torch.argmax(logits, dim=-1)
        return rows

    expected = run(False)
    actual = run(True)
    for step, (plain, grouped) in enumerate(zip(expected, actual, strict=True)):
        torch.testing.assert_close(grouped, plain, msg=f"grouped step {step} diverged")


def test_grouped_runner_foreign_cache_falls_back_without_advancing(runtime) -> None:
    model = runtime.model
    bound_cache = _fresh_cache(model)
    foreign_cache = _fresh_cache(model)
    table = foreign_cache.new_request()
    logits = model.prefill(list(_PROMPTS[0]), foreign_cache, table)
    plan = model.open_decode_window(foreign_cache, [table], budget=2)
    assert plan is not None
    runner = EngineOwnedGroupedDecodeGraphRunner(
        model, bound_cache, batch_size=1, grouped_layers=2, mode="eager"
    )

    assert runner.window_step(foreign_cache, plan, torch.argmax(logits).reshape(1)) is None
    assert plan.steps_used == 0


def test_grouped_runner_validates_supported_group_sizes(runtime) -> None:
    model = runtime.model
    cache = _fresh_cache(model)

    with pytest.raises(ValueError, match="grouped_layers must be 2 or 4"):
        EngineOwnedGroupedDecodeGraphRunner(
            model, cache, batch_size=1, grouped_layers=3, mode="eager"
        )
    with pytest.raises(ValueError, match="needs 4 layers"):
        EngineOwnedGroupedDecodeGraphRunner(
            model, cache, batch_size=1, grouped_layers=4, mode="eager"
        )


def test_batch_above_largest_bucket_falls_back_to_eager(runtime) -> None:
    """A batch no bucket covers returns None from the runner and decodes eagerly, unchanged."""
    model = runtime.model
    model.decode_graphs = None
    plain = _run_window(model, 3)
    runner = model.enable_decode_graphs(capture_sizes=(1, 2), mode="eager")
    assert runner.bucket_for(len(_PROMPTS)) is None
    fallback = _run_window(model, 3)
    for expected, got in zip(plain, fallback, strict=True):
        torch.testing.assert_close(got, expected)


def test_window_past_max_position_falls_back_to_eager(runtime) -> None:
    """A window that could out-run the pinned RoPE rows stays on the eager path."""
    model = runtime.model
    model.decode_graphs = None
    plain = _run_window(model, 3)
    runner = model.enable_decode_graphs(capture_sizes=(4,), mode="eager")
    # The pinned table always covers at least 256 rows, so force the bound below the
    # window's reach to exercise the guard.
    runner.max_position = max(len(prompt) for prompt in _PROMPTS)
    fallback = _run_window(model, 3)
    assert runner._active_plan is None  # the runner never adopted the window
    for expected, got in zip(plain, fallback, strict=True):
        torch.testing.assert_close(got, expected)
