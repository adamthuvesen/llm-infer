"""Serving-lifecycle correctness for engine-owned grouped decode dispatch.

The grouped runner was measured on fixed-length A/B decodes; serving does more. Each test
here pins one lifecycle behavior on the tiny bundle (CPU ``mode="eager"``, the exact
tranche ordering the CUDA capture replays) with the plain engine as the oracle, and every
test asserts on ``grouped_decode_steps_total`` so a silent fall-through can never pass as
grouped coverage:

* aborts mid-window (the forced-flush drain path);
* preemption + re-admission — the window path is ineligible under preemption, so grouped
  must never fire and outputs must still match;
* prefix-cache sharing — grouped inherits the window path's private-blocks requirement:
  zero steps while siblings share blocks, real steps once the survivor's table is private;
* sampled requests — a penalty-free sampled batch runs planned windows, so grouped must
  fire and outputs must match the plain engine; a penalty-carrying batch keeps windows
  (and grouped) off;
* speculative decode — drafts read generated ids per step, so windows (and grouped) stay
  off.
"""

from __future__ import annotations

import pytest

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving import InferenceEngine, Request, SamplingParams
from llm_infer.serving.speculative import SpeculativeDecodingConfig

_PROMPTS = ([1, 4, 7], [2, 5, 3, 6, 9], [8, 1])
_MAX_NEW_TOKENS = 10


@pytest.fixture()
def runtime(tmp_path):
    bundle = write_tiny_pretrain_bundle(tmp_path)
    return load_model_runtime("esme", bundle_path=bundle)


def _engine(runtime, *, grouped: bool, **kwargs) -> InferenceEngine:
    if grouped:
        kwargs.update(grouped_decode_graphs=True, grouped_capture_sizes=(1, 2, 3))
    kwargs.setdefault("num_blocks", 64)
    return InferenceEngine(
        runtime.model,
        block_size=4,
        capabilities=runtime.capabilities,
        decode_window_size=4,
        **kwargs,
    )


def _add_default_requests(engine: InferenceEngine) -> None:
    for index, prompt in enumerate(_PROMPTS):
        engine.add_request(Request(f"r{index}", list(prompt), _MAX_NEW_TOKENS, frozenset()))


def _run_with_abort(engine: InferenceEngine, abort_after_step: int) -> dict[str, list[int]]:
    """Step to completion, aborting ``r1`` after a fixed number of steps.

    The abort lands while a decode window (or its staged flush) is live, so it exercises
    the forced drain: the survivors' flushed tokens must still be recorded.
    """
    _add_default_requests(engine)
    outputs: dict[str, list[int]] = {}
    steps = 0
    while engine.scheduler.has_work():
        outputs.update(engine.step().finished_outputs)
        steps += 1
        if steps == abort_after_step:
            assert engine.abort("r1") is True
    return outputs


def test_abort_mid_window_matches_plain_engine(runtime) -> None:
    runtime.model.decode_graphs = None
    plain = _run_with_abort(_engine(runtime, grouped=False), abort_after_step=2)
    assert set(plain) == {"r0", "r2"}  # r1 was aborted before finishing

    grouped_engine = _engine(runtime, grouped=True)
    assert _run_with_abort(grouped_engine, abort_after_step=2) == plain
    assert grouped_engine.grouped_decode_steps_total > 0


def test_preemption_keeps_grouped_dispatch_off_and_outputs_identical(runtime) -> None:
    """Preemption disables the window path; grouped must never fire, outputs must match."""

    def run(grouped: bool) -> tuple[dict[str, list[int]], InferenceEngine]:
        engine = _engine(runtime, grouped=grouped, preemption=True, num_blocks=4)
        _add_default_requests(engine)
        return engine.run(), engine

    runtime.model.decode_graphs = None
    plain, plain_engine = run(grouped=False)
    assert plain_engine.preemption_count > 0  # the tight pool really forced preemption

    grouped_outputs, grouped_engine = run(grouped=True)
    assert grouped_outputs == plain
    assert grouped_engine.preemption_count > 0
    assert grouped_engine.grouped_decode_steps_total == 0


def test_prefix_sharing_excludes_grouped_until_blocks_are_private(runtime) -> None:
    """Shared prefix blocks keep grouped off; the lone survivor re-enters the grouped path."""
    prompt = [1, 4, 7, 2]

    def run(grouped: bool, group_id: str | None) -> tuple[dict[str, list[int]], InferenceEngine]:
        engine = _engine(runtime, grouped=grouped)
        for index, max_new in enumerate((4, 4, _MAX_NEW_TOKENS)):
            engine.add_request(
                Request(f"s{index}", list(prompt), max_new, frozenset(), prefix_group_id=group_id)
            )
        return engine.run(), engine

    runtime.model.decode_graphs = None
    baseline, _ = run(grouped=False, group_id=None)

    shared_outputs, shared_engine = run(grouped=True, group_id="g")
    assert shared_outputs == baseline
    # s2 outlives its siblings; once their tables are freed its blocks are refcount-1 and
    # the batch-1 window decodes through the size-1 grouped runner.
    assert shared_engine.grouped_decode_runners[1].steps_handled > 0
    # While two or more siblings were live their prompt blocks stayed shared, so no
    # multi-request window (and no larger grouped bucket) may ever have formed.
    assert shared_engine.grouped_decode_runners[2].steps_handled == 0
    assert shared_engine.grouped_decode_runners[3].steps_handled == 0


def test_sampled_requests_take_grouped_dispatch(runtime) -> None:
    """A penalty-free sampled batch decodes through grouped windows, tokens unchanged.

    The grouped runner covers only the forward; every draw still comes from the row's own
    seeded generator outside the captured region, so tokens match the plain engine exactly
    on the tiny bundle.
    """
    sampling = SamplingParams(temperature=1.0, top_p=1.0, seed=123)

    def run(grouped: bool) -> tuple[dict[str, list[int]], InferenceEngine]:
        engine = _engine(runtime, grouped=grouped, default_sampling=sampling)
        _add_default_requests(engine)
        return engine.run(), engine

    runtime.model.decode_graphs = None
    plain, _ = run(grouped=False)

    grouped_outputs, grouped_engine = run(grouped=True)
    assert grouped_outputs == plain
    assert grouped_engine.grouped_decode_steps_total > 0


def test_penalty_requests_keep_grouped_dispatch_off(runtime) -> None:
    """A penalty-carrying batch reads history per step: windows and grouped stay off."""
    sampling = SamplingParams(temperature=1.0, seed=123, frequency_penalty=0.5)

    def run(grouped: bool) -> tuple[dict[str, list[int]], InferenceEngine]:
        engine = _engine(runtime, grouped=grouped, default_sampling=sampling)
        _add_default_requests(engine)
        return engine.run(), engine

    runtime.model.decode_graphs = None
    plain, _ = run(grouped=False)

    grouped_outputs, grouped_engine = run(grouped=True)
    assert grouped_outputs == plain
    assert grouped_engine.grouped_decode_steps_total == 0


def test_speculative_decode_keeps_grouped_dispatch_off(runtime) -> None:
    speculative = SpeculativeDecodingConfig(max_draft_tokens=3, max_ngram_size=2)

    def run(grouped: bool) -> tuple[dict[str, list[int]], InferenceEngine]:
        engine = _engine(runtime, grouped=grouped, speculative=speculative)
        _add_default_requests(engine)
        return engine.run(), engine

    runtime.model.decode_graphs = None
    plain, _ = run(grouped=False)

    grouped_outputs, grouped_engine = run(grouped=True)
    assert grouped_outputs == plain
    assert grouped_engine.grouped_decode_steps_total == 0
