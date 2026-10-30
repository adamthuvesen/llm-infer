"""Request phases keep cache ownership and recompute progress together."""

from __future__ import annotations

import pytest
import torch

from llm_infer.kv_cache.block_allocator import BlockAllocator
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.serving import InferenceEngine, Request, SamplingParams
from llm_infer.serving.request import DecodeReady, Pending, Prefilling, Terminal
from tests.support.tiny_qwen import tiny_qwen


def test_initial_prefill_requires_complete_cache_before_recording() -> None:
    request = Request("r", [1, 2, 3], 3, frozenset())
    table = BlockTable(BlockAllocator(2), 4)
    request.begin_prefill(table)
    table.reserve(3)
    table.length = 2
    request.advance_prefill(2)

    with pytest.raises(ValueError, match="before initial prefill completes"):
        request.record(4)
    with pytest.raises(ValueError, match="not pending"):
        request.begin_prefill(table)
    with pytest.raises(ValueError, match="must advance"):
        request.advance_prefill(2)
    with pytest.raises(ValueError, match="block table length"):
        request.advance_prefill(3)
    assert request.generated_count == 0
    assert isinstance(request.state, Prefilling)

    table.length = 3
    request.advance_prefill(3)
    request.record(torch.tensor(4), is_eos=False)
    assert request.state == DecodeReady(table)
    with pytest.raises(ValueError, match="not prefilling"):
        request.advance_prefill(3)
    with pytest.raises(ValueError, match="not recomputing"):
        request.finish_recompute()


def test_partial_initial_prefill_preemption_keeps_initial_mode() -> None:
    request = Request("r", [1, 2, 3], 3, frozenset())
    table = BlockTable(BlockAllocator(2), 4)
    request.begin_prefill(table)
    table.reserve(1)
    table.length = 1
    request.advance_prefill(1)
    table.free()
    request.reset_for_recompute()

    assert request.state == Pending("initial")
    assert request.block_table is None
    assert request.prompt_cached_tokens == 0
    assert request.generated_count == 0


def test_preemption_preserves_identity_device_tokens_and_rng_without_sampling_on_resume() -> None:
    engine = InferenceEngine(tiny_qwen(), block_size=4, num_blocks=4, preemption=True)
    request = Request(
        "r", [1, 5, 9], 6, frozenset(), sampling=SamplingParams(temperature=1.0, seed=17)
    )
    engine.add_request(request)
    engine.step()
    old_table = request.block_table
    assert old_table is not None and old_table.blocks
    token = request.last_token_tensor
    generator = request.generator(torch.device("cpu"))
    rng_state = generator.get_state().clone()

    engine._preempt(request)

    assert engine._requests["r"] is request
    assert next(iter(engine.scheduler.waiting)) is request
    assert request.state == Pending("recompute")
    assert old_table.blocks == []
    assert engine.cache.allocator.num_used == 0
    assert request.last_token_tensor is token
    with pytest.raises(ValueError, match="before recompute"):
        request.record(2)

    result = engine.step()

    assert result.tokens == {}
    assert isinstance(request.state, DecodeReady)
    assert request.state.table is not old_table
    assert engine.scheduler.running[0] is request
    assert request.last_token_tensor is token
    assert request.generator(torch.device("cpu")) is generator
    assert torch.equal(generator.get_state(), rng_state)

    engine.run()
    assert request.state == Terminal("length")
    assert engine.cache.allocator.num_used == 0
    assert not engine.scheduler.has_work()
    assert not engine._requests


def test_recompute_rejects_early_completion_and_token_recording() -> None:
    request = Request("r", [1, 2], 4, frozenset())
    table = BlockTable(BlockAllocator(2), 4)
    table.length = 2
    request.begin_prefill(table, cached_tokens=2)
    request.record(3)
    request.reset_for_recompute()
    rebuilt = BlockTable(table.allocator, 4)
    request.begin_prefill(rebuilt)

    with pytest.raises(ValueError, match="incomplete recompute"):
        request.finish_recompute()
    with pytest.raises(ValueError, match="before recompute"):
        request.record(4)
    assert request.generated == [3]
    request.reset_for_recompute()
    assert request.state == Pending("recompute")


@pytest.mark.parametrize("steps", [0, 1, 3])
def test_abort_releases_cache_and_cannot_resume(steps: int) -> None:
    engine = InferenceEngine(tiny_qwen(), block_size=4, num_blocks=4, prefill_chunk_size=1)
    request = Request("r", [1, 5, 9], 6, frozenset())
    engine.add_request(request)
    for _ in range(steps):
        engine.step()

    assert engine.abort("r")
    assert request.state == Terminal("cancelled")
    assert request.finished
    assert request.block_table is None
    assert engine.cache.allocator.num_used == 0
    assert not engine.scheduler.has_work()
    assert not engine.abort("r")
    with pytest.raises(ValueError, match="already finished"):
        request.record(4)
    with pytest.raises(ValueError, match="cannot preempt finished"):
        request.reset_for_recompute()
    with pytest.raises(ValueError, match="only an unstarted request"):
        engine.add_request(request)


def test_engine_rejects_token_only_request() -> None:
    engine = InferenceEngine(tiny_qwen(), block_size=4, num_blocks=4)
    request = Request("r", [1, 2], 4, frozenset())
    request.record(3)
    assert request.state == Pending()
    with pytest.raises(ValueError, match="only an unstarted request"):
        engine.add_request(request)
