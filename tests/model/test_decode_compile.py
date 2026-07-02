"""Compiled decode-runner correctness on CPU: padding, parity, and eager fallbacks.

``CompiledDecodeRunner(compile_backend="eager")`` runs the exact traced step — Dynamo
capture, the opaque paged-attention custom op, bucket padding, static input buffers —
without Inductor or CUDA graphs, so these tests pin the whole caller contract on the tiny
bundle. On the GPU the remaining deltas are Inductor codegen and compiler-managed CUDA
graphs, which the runner's enable-time parity check and the Modal reference gate cover.
"""

from __future__ import annotations

import pytest
import torch

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving import InferenceEngine, Request

_PROMPTS = ([1, 4, 7], [2, 5, 3, 6, 9], [8, 1])


@pytest.fixture()
def runtime(tmp_path):
    bundle = write_tiny_pretrain_bundle(tmp_path)
    return load_model_runtime("esme", bundle_path=bundle)


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


def test_padded_compiled_runner_matches_plain_window_step(runtime) -> None:
    """Batch 3 padded to bucket 4: per-step logits match the plain planned path."""
    model = runtime.model
    budget = 4
    model.decode_graphs = None
    plain = _run_window(model, budget)
    model.enable_decode_compile(capture_sizes=(4, 8), mode=None, compile_backend="eager")
    padded = _run_window(model, budget)
    for step, (expected, got) in enumerate(zip(plain, padded, strict=True)):
        torch.testing.assert_close(got, expected, msg=f"step {step} logits diverged")
        assert torch.equal(torch.argmax(got, dim=-1), torch.argmax(expected, dim=-1))


def test_engine_outputs_unchanged_with_compiled_runner(runtime) -> None:
    """End-to-end engine run with the compiled runner equals the plain engine, token for token."""

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
    runtime.model.enable_decode_compile(capture_sizes=(4,), mode=None, compile_backend="eager")
    assert run_engine() == plain


def test_batch_above_largest_bucket_falls_back_to_eager(runtime) -> None:
    """A batch no bucket covers returns None from the runner and decodes eagerly, unchanged."""
    model = runtime.model
    model.decode_graphs = None
    plain = _run_window(model, 3)
    runner = model.enable_decode_compile(
        capture_sizes=(1, 2), mode=None, compile_backend="eager"
    )
    assert runner.bucket_for(len(_PROMPTS)) is None
    fallback = _run_window(model, 3)
    for expected, got in zip(plain, fallback, strict=True):
        torch.testing.assert_close(got, expected)


def test_window_past_max_position_falls_back_to_eager(runtime) -> None:
    """A window that could out-run the pinned RoPE rows stays on the eager path."""
    model = runtime.model
    model.decode_graphs = None
    plain = _run_window(model, 3)
    runner = model.enable_decode_compile(capture_sizes=(4,), mode=None, compile_backend="eager")
    # The pinned table always covers at least 256 rows, so force the bound below the
    # window's reach to exercise the guard.
    runner.max_position = max(len(prompt) for prompt in _PROMPTS)
    fallback = _run_window(model, 3)
    assert runner._active_plan is None  # the runner never adopted the window
    for expected, got in zip(plain, fallback, strict=True):
        torch.testing.assert_close(got, expected)


def test_static_buffers_hold_step_inputs_and_advance_positions(runtime) -> None:
    """After a step: tokens copied into the bucket prefix, positions advanced, pads contained."""
    model = runtime.model
    runner = model.enable_decode_compile(capture_sizes=(4,), mode=None, compile_backend="eager")
    cache = _fresh_cache(model)
    tables, tokens = _prefill(model, cache)
    base_lengths = [table.length for table in tables]
    plan = model.open_decode_window(cache, tables, 2)
    assert plan is not None

    state = runner._buckets[4]
    pads_before = state.tokens[3:].clone()  # whatever warmup left there
    logits = model.decode_window_step(cache, plan, tokens)
    assert logits.shape == (3, model.config.vocab_size)
    assert state.tokens[:3].tolist() == tokens.tolist()
    assert state.tokens[3:].tolist() == pads_before.tolist()  # pad rows are never written
    # Positions were loaded from the plan at window open (pads zeroed), then advanced by
    # one after the step.
    assert state.positions.tolist() == [length + 1 for length in base_lengths] + [1]
