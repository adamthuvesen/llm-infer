"""Engine-owned grouped decode dispatch: parity and every link of the fallback chain.

``InferenceEngine(grouped_decode_graphs=True)`` builds one grouped runner per capture size
against the engine's own cache. These tests run the runners in their CPU ``mode="eager"``
form (the exact tranche ordering the CUDA capture replays) on the tiny bundle and pin:

* token-for-token parity with the plain engine, including EOS mid-window and length caps;
* each link of the dispatch chain — grouped exact-batch hit, piecewise fallback on an
  off-bucket batch, eager fallback with no runner enabled at all;
* that a hit is a *real* hit (``steps_handled`` counters), so a silent fall-through can
  never masquerade as grouped coverage;
* that a runner bound to a stale cache is never consulted after the cache is rebuilt.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving import InferenceEngine, Request

_PROMPTS = ([1, 4, 7], [2, 5, 3, 6, 9], [8, 1])
_MAX_NEW_TOKENS = 10


@pytest.fixture()
def runtime(tmp_path):
    bundle = write_tiny_pretrain_bundle(tmp_path)
    return load_model_runtime("esme", bundle_path=bundle)


def _build_engine(runtime, *, grouped_sizes: tuple[int, ...] | None = None) -> InferenceEngine:
    kwargs = {}
    if grouped_sizes is not None:
        kwargs = {
            "grouped_decode_graphs": True,
            "grouped_capture_sizes": grouped_sizes,
            "grouped_layers": 2,
        }
    return InferenceEngine(
        runtime.model,
        block_size=4,
        num_blocks=64,
        capabilities=runtime.capabilities,
        decode_window_size=4,
        **kwargs,
    )


def _run(engine: InferenceEngine, eos: frozenset[int] = frozenset()) -> dict[str, list[int]]:
    for index, prompt in enumerate(_PROMPTS):
        engine.add_request(Request(f"r{index}", list(prompt), _MAX_NEW_TOKENS, eos))
    return engine.run()


def _grouped_steps(engine: InferenceEngine) -> int:
    return sum(runner.steps_handled for runner in engine.grouped_decode_runners.values())


def test_exact_batch_windows_decode_through_the_grouped_runner(runtime) -> None:
    """Batch 3 with a size-3 runner: identical outputs, and the runner really ran."""
    runtime.model.decode_graphs = None
    plain = _run(_build_engine(runtime))

    grouped_engine = _build_engine(runtime, grouped_sizes=(3,))
    assert _run(grouped_engine) == plain
    assert _grouped_steps(grouped_engine) > 0


def test_eos_mid_window_outputs_match_with_grouped_dispatch(runtime) -> None:
    """A request stopping on EOS mid-window records identical tokens either way.

    The tiny bundle continues ``[5, 2]`` as ``3, 6, 6, ...`` — the first decode token
    differs from the prefill-sampled one — so EOS ``{6}`` fires at decode step 1, inside
    a 4-step window, and the overshoot tokens must be discarded identically. The other
    prompts never emit 6 and run to their length cap; after the finisher is consumed the
    batch drops to 2, which the size-2 runner picks up.
    """
    prompts = ([5, 2], [2, 5, 3, 6, 9], [8, 1])
    eos = frozenset({6})

    def run(engine: InferenceEngine) -> dict[str, list[int]]:
        for index, prompt in enumerate(prompts):
            engine.add_request(Request(f"r{index}", list(prompt), _MAX_NEW_TOKENS, eos))
        return engine.run()

    runtime.model.decode_graphs = None
    plain = run(_build_engine(runtime))
    assert 1 < len(plain["r0"]) < _MAX_NEW_TOKENS  # EOS hit on a decode step, not at prefill

    grouped_engine = _build_engine(runtime, grouped_sizes=(2, 3))
    assert run(grouped_engine) == plain
    assert grouped_engine.grouped_decode_runners[3].steps_handled > 0  # before the finish
    assert grouped_engine.grouped_decode_runners[2].steps_handled > 0  # after it


def test_off_bucket_batch_falls_back_to_the_piecewise_runner(runtime) -> None:
    """Batch 3 with only a size-2 grouped runner lands on the model's piecewise path."""
    runtime.model.decode_graphs = None
    plain = _run(_build_engine(runtime))

    piecewise = runtime.model.enable_decode_graphs(capture_sizes=(4,), mode="eager")
    piecewise_steps = 0
    original_window_step = piecewise.window_step

    def counting_window_step(cache, plan, token_ids):
        nonlocal piecewise_steps
        logits = original_window_step(cache, plan, token_ids)
        if logits is not None:
            piecewise_steps += 1
        return logits

    piecewise.window_step = counting_window_step
    grouped_engine = _build_engine(runtime, grouped_sizes=(2,))
    assert _run(grouped_engine) == plain
    assert _grouped_steps(grouped_engine) == 0  # no size-3 runner exists
    assert piecewise_steps > 0  # the fallback chain landed on piecewise, not eager


def test_off_bucket_batch_without_piecewise_falls_back_to_eager(runtime) -> None:
    """With no piecewise runner enabled, the chain ends on the eager planned step."""
    runtime.model.decode_graphs = None
    plain = _run(_build_engine(runtime))

    grouped_engine = _build_engine(runtime, grouped_sizes=(2,))
    assert _run(grouped_engine) == plain
    assert _grouped_steps(grouped_engine) == 0


def test_stale_cache_runner_is_never_consulted_after_rebuild(runtime) -> None:
    """Rebuilding the engine cache strands the runners; decode still runs correctly."""
    runtime.model.decode_graphs = None
    plain = _run(_build_engine(runtime))

    engine = _build_engine(runtime, grouped_sizes=(3,))
    stale_cache = engine.cache
    engine.cache = PagedKVCache(
        num_layers=runtime.model.num_layers,
        num_blocks=64,
        block_size=4,
        num_kv_heads=runtime.model.num_kv_heads,
        head_dim=runtime.model.head_dim,
        dtype=runtime.model.dtype,
    )
    assert engine.grouped_decode_runners[3].cache is stale_cache
    assert _run(engine) == plain
    assert _grouped_steps(engine) == 0


def test_oversized_bucket_is_skipped_not_fatal(runtime) -> None:
    """A bucket the pool cannot hold a capture window for is dropped; the rest capture.

    With ``num_blocks=64`` and ``block_size=4``, one capture window needs 2 blocks per
    request, so bucket 40 (80 blocks) exceeds the pool while bucket 3 fits. Startup must
    not raise, the oversized bucket must be absent, and the surviving bucket must serve.
    """
    runtime.model.decode_graphs = None
    plain = _run(_build_engine(runtime))

    engine = _build_engine(runtime, grouped_sizes=(3, 40))
    assert sorted(engine.grouped_decode_runners) == [3]
    assert _run(engine) == plain
    assert engine.grouped_decode_runners[3].steps_handled > 0


def test_grouped_dispatch_requires_planned_decode(runtime) -> None:
    without_planned = replace(runtime.capabilities, planned_decode=False)
    with pytest.raises(ValueError, match="grouped_decode_graphs requires"):
        InferenceEngine(
            runtime.model,
            block_size=4,
            num_blocks=64,
            capabilities=without_planned,
            grouped_decode_graphs=True,
            grouped_capture_sizes=(1,),
            grouped_layers=2,
        )
