"""Deferred-decode window correctness: deferred stop tracking equals the per-step path.

The decode window (``decode_window_size > 1``) keeps sampled tokens on device and syncs
EOS/stop state to the host once per window instead of once per step. These tests pin the
contract on the tiny bundle (CPU, deterministic): outputs are token-for-token identical to
the classic per-step engine — including requests that hit EOS mid-window (overshoot tokens
discarded) and requests that hit their length cap — while token *visibility* in step results
is allowed to arrive in window-sized bursts. Ineligible batches (sampled rows) must fall
back to the per-step path unchanged.
"""

from __future__ import annotations

import pytest

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.model.decode import greedy_decode
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving import InferenceEngine, Request
from llm_infer.serving.sampler import SamplingParams

_PROMPTS = ([1, 4, 7], [2, 5], [3, 6, 9, 10], [8, 1])


@pytest.fixture(scope="module")
def runtime(tmp_path_factory: pytest.TempPathFactory):
    bundle = write_tiny_pretrain_bundle(tmp_path_factory.mktemp("bundle"))
    return load_model_runtime("esme", bundle_path=bundle)


def _engine(runtime, *, window: int, num_blocks: int = 64) -> InferenceEngine:
    return InferenceEngine(
        runtime.model,
        block_size=8,
        num_blocks=num_blocks,
        capabilities=runtime.capabilities,
        decode_window_size=window,
    )


def _run(
    runtime,
    *,
    window: int,
    max_new_tokens: int,
    eos: frozenset[int],
    sampling: dict[int, SamplingParams] | None = None,
) -> dict[str, list[int]]:
    engine = _engine(runtime, window=window)
    for index, prompt in enumerate(_PROMPTS):
        engine.add_request(
            Request(
                f"r{index}",
                list(prompt),
                max_new_tokens,
                eos,
                sampling=(sampling or {}).get(index, SamplingParams()),
            )
        )
    return engine.run()


def _mid_run_eos(runtime, max_new_tokens: int) -> frozenset[int]:
    """An EOS id that fires mid-continuation for some prompts but not at step 0 for all."""
    open_ended = {
        f"r{index}": greedy_decode(
            runtime.model, list(prompt), max_new_tokens=max_new_tokens, eos_token_ids=set()
        )
        for index, prompt in enumerate(_PROMPTS)
    }
    candidates = [ids[len(ids) // 2] for ids in open_ended.values()]
    return frozenset({candidates[0]})


def test_window_outputs_match_per_step_engine_with_mid_run_eos(runtime) -> None:
    """Deferred windows reproduce the per-step engine exactly, EOS overshoot discarded."""
    max_new = 12
    eos = _mid_run_eos(runtime, max_new)
    reference = {
        f"r{index}": greedy_decode(
            runtime.model, list(prompt), max_new_tokens=max_new, eos_token_ids=set(eos)
        )
        for index, prompt in enumerate(_PROMPTS)
    }
    per_step = _run(runtime, window=1, max_new_tokens=max_new, eos=eos)
    windowed = _run(runtime, window=5, max_new_tokens=max_new, eos=eos)
    assert per_step == reference
    assert windowed == reference


def test_window_runs_per_scheduler_pass_and_flushes_one_window_behind(runtime) -> None:
    """One scheduler pass runs a whole window; a flushed window is consumed one window later."""
    engine = _engine(runtime, window=4)
    engine.add_request(Request("r0", [1, 4, 7], 9, frozenset()))

    prefill = engine.step()
    assert [len(tokens) for tokens in prefill.tokens.values()] == [1]

    # Budget is min(window=4, remaining=8) = 4: the pass runs all 4 steps and stages the
    # flush, but its tokens are not visible yet — the copy is consumed one window behind.
    staged = engine.step()
    assert not staged.tokens
    assert engine._pending_flush is not None

    # The next window pipelines behind the stage (budget 8-4=4); flushing it consumes the
    # first window's tokens.
    first_burst = engine.step()
    assert [len(tokens) for tokens in first_burst.tokens.values()] == [4]
    assert not first_burst.finished

    # No budget remains beyond the staged steps, so the next pass drains: the second
    # window's tokens land and the length cap finishes the request.
    second_burst = engine.step()
    assert [len(tokens) for tokens in second_burst.tokens.values()] == [4]
    assert second_burst.finished == ["r0"]
    assert engine._pending_flush is None


def test_window_budget_respects_length_cap(runtime) -> None:
    """A window never decodes a request past max_new_tokens, and the cap finishes it."""
    max_new = 3  # after the prefill token only 2 remain — well under the window size
    per_step = _run(runtime, window=1, max_new_tokens=max_new, eos=frozenset())
    windowed = _run(runtime, window=16, max_new_tokens=max_new, eos=frozenset())
    assert windowed == per_step
    assert all(len(ids) == max_new for ids in windowed.values())


def test_mixed_sampling_batch_falls_back_to_per_step_path(runtime) -> None:
    """A batch with a sampled row is window-ineligible and must match the per-step engine."""
    sampling = {1: SamplingParams(temperature=0.8, seed=1234)}
    per_step = _run(runtime, window=1, max_new_tokens=8, eos=frozenset(), sampling=sampling)
    windowed = _run(runtime, window=6, max_new_tokens=8, eos=frozenset(), sampling=sampling)
    assert windowed == per_step


def test_abort_flush_lands_in_next_step_result(runtime) -> None:
    """Aborting mid-window flushes survivors' deferred tokens into the next step result."""
    # num_blocks=4 admits only two of the three 2-block requests, so the waiting queue
    # pauses the multi-step loop and the window genuinely spans passes before the abort.
    engine = _engine(runtime, window=8, num_blocks=4)
    keep = Request("keep", [1, 4, 7], 14, frozenset())
    engine.add_request(keep)
    engine.add_request(Request("drop", [2, 5], 14, frozenset()))
    engine.add_request(Request("waiter", [3, 6], 14, frozenset()))

    first = engine.step()  # prefill keep+drop; waiter held back by the block budget
    assert first.admitted == ["keep", "drop"]
    engine.step()  # one deferred decode step, tokens pending on device
    assert engine._decode_window is not None
    assert engine.abort("drop") is True

    result = engine.step()  # carries the abort-time flush for the surviving requests
    assert result.tokens.get("keep"), "abort-time flush must reach the next step result"
    while engine.scheduler.has_work():
        engine.step()
    reference = greedy_decode(runtime.model, [1, 4, 7], max_new_tokens=14, eos_token_ids=set())
    assert keep.generated == reference
