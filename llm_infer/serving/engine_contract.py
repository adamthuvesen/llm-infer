"""Shared host surface expected by the engine mixins."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol

import torch

from llm_infer.kv_cache.block_allocator import BlockPoolEvent
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.interface import CausalLMBackend
from llm_infer.profiling import TimingProfiler
from llm_infer.scheduler.scheduler import Scheduler
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import SamplingParams
from llm_infer.serving.speculative import PromptLookupDraft
from llm_infer.tracing import TokenSource, TraceEventName, TraceRecorder

if TYPE_CHECKING:
    from llm_infer.serving.engine import StepResult


class EngineMixinHost(Protocol):
    """Attributes and cross-mixin hooks provided by :class:`InferenceEngine`.

    The engine is intentionally split across mixins. This base keeps their shared
    ``self`` contract explicit without adding another runtime abstraction layer.
    """

    model: CausalLMBackend
    cache: PagedKVCache
    scheduler: Scheduler
    default_sampling: SamplingParams
    profiler: TimingProfiler | None
    prefill_chunk_size: int | None
    preemption: bool
    speculative: PromptLookupDraft | None
    trace: TraceRecorder | None
    preemption_count: int

    def _emit_trace(self, event: TraceEventName, **fields: object) -> None:
        ...

    def _trace_decode_step(
        self,
        requests: list[Request],
        tokens: list[int | torch.Tensor],
        *,
        token_source: TokenSource,
    ) -> None:
        ...

    def _trace_prefill_chunk_started(
        self, request_id: str, *, start_pos: int, end_pos: int, total_prompt_tokens: int
    ) -> None:
        ...

    def _trace_prefill_chunk_progress(
        self,
        request_id: str,
        *,
        start_pos: int,
        end_pos: int,
        cached_tokens: int,
        total_prompt_tokens: int,
        completed: bool,
    ) -> None:
        ...

    def _trace_pool_event(self, event: BlockPoolEvent) -> None:
        ...

    def _trace_request_finished(self, request: Request) -> None:
        ...

    def _record_time(self, name: str) -> AbstractContextManager[object]:
        ...

    def _record_host_time(self, name: str) -> AbstractContextManager[object]:
        ...

    def _ensure_pool_room(self, request: Request, new_tokens: int) -> None:
        ...

    def _blocks_to_grow(self, request: Request, new_tokens: int) -> int:
        ...

    def _is_running(self, request: Request) -> bool:
        ...

    def _live_running(self, requests: Iterable[Request]) -> list[Request]:
        ...

    def _sample_one(self, logits: torch.Tensor, request: Request) -> torch.Tensor:
        ...

    def _record(
        self, request: Request, token: int | torch.Tensor, is_eos: bool, result: StepResult
    ) -> None:
        ...

    def _eos_flags(self, tokens: torch.Tensor, requests: list[Request]) -> list[bool]:
        ...

    def _release_finished_in(self, requests: list[Request], result: StepResult) -> None:
        ...

    def _free_block_table(self, table: BlockTable) -> None:
        ...
