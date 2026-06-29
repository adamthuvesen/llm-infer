"""Tracing and profiling hooks for the continuous-batching decode loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llm_infer.kv_cache.block_allocator import BlockPoolEvent
from llm_infer.serving.request import Request
from llm_infer.tracing import FinishReason, TokenSource, TraceEvent, TraceEventName

if TYPE_CHECKING:
    from llm_infer.serving.engine import StepResult


class EngineTraceMixin:
    def _trace_decode_step(
        self,
        requests: list[Request],
        tokens: list[int | torch.Tensor],
        *,
        token_source: TokenSource,
    ) -> None:
        if self.trace is None:
            return
        self._emit_trace(
            "decode_step",
            request_ids=tuple(request.request_id for request in requests),
            batch_size=len(requests),
            token_ids=tuple(self._trace_token_id(token) for token in tokens),
            tokens_emitted=len(tokens),
            token_source=token_source,
        )

    def _trace_pool_event(self, event: BlockPoolEvent) -> None:
        """Emit block lifecycle from the allocator's physical free-pool boundary.

        Each :class:`BlockPoolEvent` is purely an allocation or a real free (refcount-0):
        ``allocate`` reports no frees, ``free`` reports only the blocks that returned to the
        pool, and ``retain`` (prefix sharing) reports nothing at all.
        """
        if event.allocated:
            self._emit_trace(
                "block_allocated",
                request_id=event.owner,
                block_count=len(event.allocated),
                block_ids=event.allocated,
                pool_used=event.num_used,
                pool_free=event.num_free,
            )
        if event.freed:
            self._emit_trace(
                "block_freed",
                request_id=event.owner,
                block_count=len(event.freed),
                block_ids=event.freed,
                pool_used=event.num_used,
                pool_free=event.num_free,
            )

    def _trace_request_finished(self, request: Request) -> None:
        if self.trace is None:
            return
        reason: FinishReason = "eos" if request.last_token in request.eos_token_ids else "length"
        self._emit_trace(
            "request_finished",
            request_id=request.request_id,
            token_ids=tuple(request.generated),
            generated_tokens=len(request.generated),
            reason=reason,
        )

    def _trace_batch_size_changed(self) -> None:
        if self.trace is None:
            return
        batch_size = len(self.scheduler.running)
        if batch_size == self._last_traced_batch_size:
            return
        self._emit_trace(
            "batch_size_changed",
            previous_batch_size=self._last_traced_batch_size,
            batch_size=batch_size,
            waiting=len(self.scheduler.waiting),
        )
        self._last_traced_batch_size = batch_size

    def _trace_throughput_sample(self, result: StepResult) -> None:
        if self.trace is None:
            return
        tokens_this_step = sum(len(tokens) for tokens in result.tokens.values())
        if tokens_this_step == 0:
            return
        self._trace_total_tokens += tokens_this_step
        elapsed = max(self._trace_clock() - self._trace_start_time, 1e-12)
        self._emit_trace(
            "tokens_per_second_sampled",
            tokens_emitted=tokens_this_step,
            total_generated_tokens=self._trace_total_tokens,
            elapsed_seconds=elapsed,
            tokens_per_second=self._trace_total_tokens / elapsed,
        )

    def _emit_trace(self, event: TraceEventName, **fields: object) -> None:
        if self.trace is None:
            return
        if self._trace_step is None:
            raise RuntimeError("trace events can only be emitted during engine.step()")
        self._trace_sequence += 1
        self.trace.record(
            TraceEvent(
                event=event,
                sequence=self._trace_sequence,
                step=self._trace_step,
                **fields,
            )
        )

    def _trace_token_id(self, token: int | torch.Tensor) -> int:
        if isinstance(token, torch.Tensor):
            with self._record_host_time("cpu_gpu_sync"):
                return int(token.cpu().item())
        return token

    def _record_time(self, name: str):
        if self.profiler is None:
            return _NullTimer()
        return self.profiler.record(name)

    def _record_host_time(self, name: str):
        if self.profiler is None:
            return _NullTimer()
        return self.profiler.host(name)


class _NullTimer:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
