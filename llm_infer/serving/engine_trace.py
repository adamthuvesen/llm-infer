"""Engine trace emission: step-scoped sequencing of :class:`~llm_infer.tracing.TraceEvent`."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence

import torch

from llm_infer.kv_cache.block_allocator import BlockPoolEvent
from llm_infer.profiling import TimingProfiler
from llm_infer.request.request import Request
from llm_infer.tracing import (
    FinishReason,
    PreemptReason,
    TokenSource,
    TraceEvent,
    TraceEventName,
    TraceRecorder,
)


class EngineTrace:
    """Numbers events by step and sequence and sends them to the recorder; inert without one."""

    def __init__(
        self,
        recorder: TraceRecorder | None,
        *,
        clock: Callable[[], float] | None = None,
        profiler: TimingProfiler | None = None,
    ) -> None:
        self.recorder = recorder
        self._clock = clock or time.perf_counter
        self._profiler = profiler
        self._step_index = 0
        self._sequence = 0
        self._step: int | None = None
        self._start_time = self._clock()
        self._total_tokens = 0
        self._last_batch_size = 0

    def begin_step(self) -> None:
        self._step = self._step_index
        self._step_index += 1

    def end_step(self) -> None:
        self._step = None

    def decode_step(
        self,
        requests: list[Request],
        tokens: Sequence[int | torch.Tensor],
        *,
        token_source: TokenSource,
    ) -> None:
        if self.recorder is None:
            return
        self.emit(
            "decode_step",
            request_ids=tuple(request.request_id for request in requests),
            batch_size=len(requests),
            token_ids=tuple(self._token_id(token) for token in tokens),
            tokens_emitted=len(tokens),
            token_source=token_source,
        )

    def prefill_chunk_started(
        self, request_id: str, *, start_pos: int, end_pos: int, total_prompt_tokens: int
    ) -> None:
        self.emit(
            "prefill_chunk_started",
            request_id=request_id,
            start_pos=start_pos,
            end_pos=end_pos,
            total_prompt_tokens=total_prompt_tokens,
        )

    def prefill_chunk_progress(
        self,
        request_id: str,
        *,
        start_pos: int,
        end_pos: int,
        cached_tokens: int,
        total_prompt_tokens: int,
        completed: bool,
    ) -> None:
        self.emit(
            "prefill_chunk_progress",
            request_id=request_id,
            start_pos=start_pos,
            end_pos=end_pos,
            cached_tokens=cached_tokens,
            total_prompt_tokens=total_prompt_tokens,
            completed=completed,
        )

    def pool_event(self, event: BlockPoolEvent) -> None:
        """Emit block lifecycle from the allocator's physical free-pool boundary.

        Each :class:`BlockPoolEvent` is purely an allocation or a real free (refcount-0):
        ``allocate`` reports no frees, ``free`` reports only the blocks that returned to the
        pool, and ``retain`` (prefix sharing) reports nothing at all.
        """
        if event.allocated:
            self.emit(
                "block_allocated",
                request_id=event.owner,
                block_count=len(event.allocated),
                block_ids=event.allocated,
                pool_used=event.num_used,
                pool_free=event.num_free,
            )
        if event.freed:
            self.emit(
                "block_freed",
                request_id=event.owner,
                block_count=len(event.freed),
                block_ids=event.freed,
                pool_used=event.num_used,
                pool_free=event.num_free,
            )

    def request_finished(self, request: Request) -> None:
        if self.recorder is None:
            return
        reason: FinishReason = "eos" if request.last_token in request.eos_token_ids else "length"
        self.emit(
            "request_finished",
            request_id=request.request_id,
            token_ids=tuple(request.generated),
            generated_tokens=request.generated_count,
            reason=reason,
        )

    def batch_size_changed(self, batch_size: int, waiting: int) -> None:
        if self.recorder is None or batch_size == self._last_batch_size:
            return
        self.emit(
            "batch_size_changed",
            previous_batch_size=self._last_batch_size,
            batch_size=batch_size,
            waiting=waiting,
        )
        self._last_batch_size = batch_size

    def throughput_sample(self, tokens: Mapping[str, list[int | torch.Tensor]]) -> None:
        if self.recorder is None:
            return
        tokens_this_step = sum(len(row) for row in tokens.values())
        if tokens_this_step == 0:
            return
        self._total_tokens += tokens_this_step
        elapsed = max(self._clock() - self._start_time, 1e-12)
        self.emit(
            "tokens_per_second_sampled",
            tokens_emitted=tokens_this_step,
            total_generated_tokens=self._total_tokens,
            elapsed_seconds=elapsed,
            tokens_per_second=self._total_tokens / elapsed,
        )

    def emit(
        self,
        event: TraceEventName,
        *,
        request_id: str | None = None,
        request_ids: tuple[str, ...] = (),
        prompt_tokens: int | None = None,
        max_new_tokens: int | None = None,
        prefix_group_id: str | None = None,
        reserved_blocks: int | None = None,
        start_pos: int | None = None,
        end_pos: int | None = None,
        cached_tokens: int | None = None,
        total_prompt_tokens: int | None = None,
        completed: bool | None = None,
        block_count: int | None = None,
        block_ids: tuple[int, ...] = (),
        pool_used: int | None = None,
        pool_free: int | None = None,
        batch_size: int | None = None,
        previous_batch_size: int | None = None,
        waiting: int | None = None,
        token_ids: tuple[int, ...] = (),
        tokens_emitted: int | None = None,
        token_source: TokenSource | None = None,
        generated_tokens: int | None = None,
        reason: FinishReason | None = None,
        preempt_reason: PreemptReason | None = None,
        total_generated_tokens: int | None = None,
        elapsed_seconds: float | None = None,
        tokens_per_second: float | None = None,
    ) -> None:
        if self.recorder is None:
            return
        if self._step is None:
            raise RuntimeError("trace events can only be emitted during engine.step()")
        self._sequence += 1
        self.recorder.record(
            TraceEvent(
                event=event,
                sequence=self._sequence,
                step=self._step,
                request_id=request_id,
                request_ids=request_ids,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                prefix_group_id=prefix_group_id,
                reserved_blocks=reserved_blocks,
                start_pos=start_pos,
                end_pos=end_pos,
                cached_tokens=cached_tokens,
                total_prompt_tokens=total_prompt_tokens,
                completed=completed,
                block_count=block_count,
                block_ids=block_ids,
                pool_used=pool_used,
                pool_free=pool_free,
                batch_size=batch_size,
                previous_batch_size=previous_batch_size,
                waiting=waiting,
                token_ids=token_ids,
                tokens_emitted=tokens_emitted,
                token_source=token_source,
                generated_tokens=generated_tokens,
                reason=reason,
                preempt_reason=preempt_reason,
                total_generated_tokens=total_generated_tokens,
                elapsed_seconds=elapsed_seconds,
                tokens_per_second=tokens_per_second,
            )
        )

    def _token_id(self, token: int | torch.Tensor) -> int:
        if not isinstance(token, torch.Tensor):
            return token
        if self._profiler is None:
            return int(token.cpu().item())
        with self._profiler.host("cpu_gpu_sync"):
            return int(token.cpu().item())
