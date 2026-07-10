"""Engine routing for the optional ragged batched-prefill backend contract."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from llm_infer.kv_cache import BlockTable, PagedKVCache
from llm_infer.model.interface import DENSE_CAPABILITIES, QWEN_CAPABILITIES, BackendCapabilities
from llm_infer.serving import InferenceEngine, Request
from llm_infer.tracing import TraceRecorder
from tests.support.fake_causal_lm import FakeCausalLMBase


def _capabilities(*, batched_prefill: bool) -> BackendCapabilities:
    return BackendCapabilities(
        paged_kv=True,
        prefix_caching=True,
        speculative=True,
        flash_attention=False,
        batched_prefill=batched_prefill,
    )


class SequentialPrefillModel(FakeCausalLMBase):
    def __init__(self) -> None:
        self.num_layers = 1
        self.num_kv_heads = 1
        self.head_dim = 2
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.profiler = None
        self.prefill_calls: list[list[int]] = []
        self.release_calls = 0

    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        self.prefill_calls.append(list(prompt_ids))
        return self._write_prompt(prompt_ids, cache, table, start_pos=0)

    def prefill_chunk(
        self,
        prompt_ids: list[int],
        cache: PagedKVCache,
        table: BlockTable,
        *,
        start_pos: int,
        chunk_size: int,
    ) -> torch.Tensor:
        end_pos = min(len(prompt_ids), start_pos + chunk_size)
        self.prefill_calls.append(list(prompt_ids[start_pos:end_pos]))
        return self._write_prompt(prompt_ids[:end_pos], cache, table, start_pos=start_pos)

    def _write_prompt(
        self,
        prompt_ids: list[int],
        cache: PagedKVCache,
        table: BlockTable,
        *,
        start_pos: int,
    ) -> torch.Tensor:
        token_count = len(prompt_ids) - start_pos
        table.reserve(token_count)
        rows = torch.arange(start_pos, len(prompt_ids), dtype=torch.float32).reshape(-1, 1, 1)
        key = torch.cat([rows, rows + 0.5], dim=-1)
        cache.write(table, layer=0, start_pos=start_pos, key=key, value=key + 100.0)
        table.length = len(prompt_ids)
        return self._logits(sum(prompt_ids) % 32)

    def _logits(self, token_id: int) -> torch.Tensor:
        logits = torch.full((32,), -100.0)
        logits[token_id] = 100.0
        return logits

    def release_table(self, table: BlockTable) -> None:
        self.release_calls += 1
        super().release_table(table)


class BatchedPrefillModel(SequentialPrefillModel):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_many_calls: list[list[list[int]]] = []

    def prefill_many(
        self,
        prompt_ids: list[list[int]],
        cache: PagedKVCache,
        tables: list[BlockTable],
    ) -> torch.Tensor:
        self.prefill_many_calls.append([list(prompt) for prompt in prompt_ids])
        return torch.stack(
            [
                self._write_prompt(prompt, cache, table, start_pos=0)
                for prompt, table in zip(prompt_ids, tables, strict=True)
            ]
        )


def _requests(*, prefix_groups: bool = False) -> list[Request]:
    return [
        Request(
            request_id,
            prompt,
            max_new_tokens=1,
            eos_token_ids=frozenset(),
            prefix_group_id=f"group-{request_id}" if prefix_groups else None,
        )
        for request_id, prompt in (("a", [1, 2]), ("b", [3, 4, 5]))
    ]


def _run_step(
    model: SequentialPrefillModel,
    *,
    batched_prefill: bool | None,
    capabilities: BackendCapabilities,
    request_factory: Callable[[], list[Request]] = _requests,
    prefill_chunk_size: int | None = None,
    preemption: bool = False,
    trace: TraceRecorder | None = None,
):
    kwargs: dict[str, object] = {}
    if batched_prefill is not None:
        kwargs["batched_prefill"] = batched_prefill
    engine = InferenceEngine(
        model,
        block_size=4,
        num_blocks=16,
        capabilities=capabilities,
        prefill_chunk_size=prefill_chunk_size,
        preemption=preemption,
        trace=trace,
        **kwargs,
    )
    for request in request_factory():
        engine.add_request(request)
    return engine, engine.step()


def test_batched_prefill_capability_is_opt_in() -> None:
    assert DENSE_CAPABILITIES.batched_prefill is False
    assert QWEN_CAPABILITIES.batched_prefill is False
    assert _capabilities(batched_prefill=False).batched_prefill is False


def test_enabled_batched_prefill_routes_once_and_preserves_step_semantics() -> None:
    model = BatchedPrefillModel()
    recorder = TraceRecorder()

    engine, result = _run_step(
        model,
        batched_prefill=True,
        capabilities=_capabilities(batched_prefill=True),
        trace=recorder,
    )

    assert model.prefill_many_calls == [[[1, 2], [3, 4, 5]]]
    assert model.prefill_calls == []
    assert result.prefill_chunks == {"a": (0, 2), "b": (0, 3)}
    assert result.finished_outputs == {"a": [3], "b": [12]}
    assert model.release_calls == 2
    assert engine.cache.allocator.num_free == engine.cache.num_blocks

    started = [event for event in recorder.events if event.event == "prefill_chunk_started"]
    progressed = [event for event in recorder.events if event.event == "prefill_chunk_progress"]
    decoded = [event for event in recorder.events if event.event == "decode_step"]
    assert [(event.request_id, event.start_pos, event.end_pos) for event in started] == [
        ("a", 0, 2),
        ("b", 0, 3),
    ]
    assert [(event.request_id, event.completed) for event in progressed] == [
        ("a", True),
        ("b", True),
    ]
    assert decoded[0].request_ids == ("a", "b")
    assert decoded[0].token_ids == (3, 12)
    assert decoded[0].token_source == "prefill"


def test_batched_prefill_is_the_default() -> None:
    """No explicit flag: an eligible batch routes through prefill_many (enabled by default)."""
    model = BatchedPrefillModel()

    _run_step(
        model,
        batched_prefill=None,
        capabilities=_capabilities(batched_prefill=True),
    )

    assert model.prefill_many_calls == [[[1, 2], [3, 4, 5]]]
    assert model.prefill_calls == []


@pytest.mark.parametrize(
    (
        "reason",
        "model_factory",
        "batched_prefill",
        "capabilities",
        "request_factory",
        "chunk",
        "preemption",
    ),
    [
        (
            "explicit-off",
            BatchedPrefillModel,
            False,
            _capabilities(batched_prefill=True),
            _requests,
            None,
            False,
        ),
        (
            "capability-off",
            BatchedPrefillModel,
            True,
            _capabilities(batched_prefill=False),
            _requests,
            None,
            False,
        ),
        (
            "protocol-missing",
            SequentialPrefillModel,
            True,
            _capabilities(batched_prefill=True),
            _requests,
            None,
            False,
        ),
        (
            "single-request",
            BatchedPrefillModel,
            True,
            _capabilities(batched_prefill=True),
            lambda: _requests()[:1],
            None,
            False,
        ),
        (
            "chunking",
            BatchedPrefillModel,
            True,
            _capabilities(batched_prefill=True),
            _requests,
            1,
            False,
        ),
        (
            "prefix-groups",
            BatchedPrefillModel,
            True,
            _capabilities(batched_prefill=True),
            lambda: _requests(prefix_groups=True),
            None,
            False,
        ),
        (
            "preemption",
            BatchedPrefillModel,
            True,
            _capabilities(batched_prefill=True),
            _requests,
            None,
            True,
        ),
    ],
)
def test_ineligible_prefill_uses_existing_sequential_path(
    reason: str,
    model_factory: type[SequentialPrefillModel],
    batched_prefill: bool | None,
    capabilities: BackendCapabilities,
    request_factory: Callable[[], list[Request]],
    chunk: int | None,
    preemption: bool,
) -> None:
    model = model_factory()

    _run_step(
        model,
        batched_prefill=batched_prefill,
        capabilities=capabilities,
        request_factory=request_factory,
        prefill_chunk_size=chunk,
        preemption=preemption,
    )

    assert getattr(model, "prefill_many_calls", []) == [], reason
    assert len(model.prefill_calls) == len(request_factory()), reason
