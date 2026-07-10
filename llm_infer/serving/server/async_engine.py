"""Run InferenceEngine.step on a background thread; handlers stream per-request tokens."""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

import torch

from llm_infer.scheduler.scheduler import blocks_for_length
from llm_infer.serving.engine import InferenceEngine, StepResult
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import GREEDY, SamplingParams
from llm_infer.serving.server.metrics import ServerMetrics


@dataclass
class TokenStreamItem:
    """One token handed to a streaming handler, plus the finish reason on the last item."""

    token_id: int
    finish_reason: str | None = None


class AsyncEngineRequestObserver(Protocol):
    """Optional request-output capture after normal token materialization."""

    def submitted(self, request: Request, submitted_s: float) -> None: ...

    def admitted(self, request_id: str, admitted_s: float) -> None: ...

    def finished(self, request_id: str, token_ids: list[int], finished_s: float) -> None: ...


@dataclass
class _Stream:
    """The loop's view of one in-flight request: where to push tokens and whether to stop."""

    request_id: str
    queue: asyncio.Queue[TokenStreamItem | None]
    loop: asyncio.AbstractEventLoop
    max_new_tokens: int
    submitted_s: float
    aborted: bool = False
    eos_token_ids: frozenset[int] = field(default_factory=frozenset)
    # Set when the engine refuses this submission on the background thread; the consumer
    # re-raises it so the failure surfaces loudly instead of hanging on an empty stream.
    error: Exception | None = None


class AsyncInferenceEngine:
    """Drive a synchronous :class:`InferenceEngine` from asyncio over a background thread.

    Construct it around an already-built engine (the app factory injects one wrapping a tiny
    CPU model in tests or a real runtime in production). Call :meth:`start` to spin up the loop
    thread and :meth:`stream` per request; :meth:`stop` joins the thread on shutdown.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        idle_sleep_s: float = 0.001,
        metrics: ServerMetrics | None = None,
        request_observer: AsyncEngineRequestObserver | None = None,
    ) -> None:
        self._engine = engine
        self._idle_sleep_s = idle_sleep_s
        self._metrics = metrics
        self._request_observer = request_observer
        self._submissions: list[tuple[Request, _Stream]] = []
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._ids = itertools.count()
        # Set if the engine step loop dies: the engine is then unhealthy, every active stream is
        # failed, and new submissions are rejected fast instead of hanging forever on the queue.
        self._fatal: BaseException | None = None
        if metrics is not None:
            self._bind_gauges(metrics)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("async engine already started")
        self._thread = threading.Thread(target=self._run_loop, name="infer-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def next_request_id(self) -> str:
        """A process-unique request id; the engine rejects duplicates loudly."""
        return f"req-{next(self._ids)}"

    def assert_admissible(self, *, prompt_len: int, max_new_tokens: int) -> None:
        """Reject a request too large for the pool *before* it reaches the engine thread.

        The scheduler enforces the same worst-case-fits bound in ``add()``, but that runs on the
        background loop where a raise would kill the engine for every client. Checking the request
        shape here lets the handler return a clean 4xx and keeps the one loop alive. Raises
        ``ValueError`` (the handler maps it to a 400) when even an empty pool could not hold it.
        """
        scheduler = self._engine.scheduler
        need = blocks_for_length(prompt_len + max_new_tokens - 1, scheduler.block_size)
        if need > scheduler.num_blocks:
            raise ValueError(
                f"request needs up to {need} KV blocks but the pool holds {scheduler.num_blocks}; "
                "reduce the prompt length or max_tokens"
            )

    def _bind_gauges(self, metrics: ServerMetrics) -> None:
        """Point the live-read gauges at real engine/scheduler/allocator state.

        These are read-only scalar snapshots (running/waiting counts, free-pool size) taken at
        scrape time, so a scrape always reflects the engine's state the instant ``/metrics`` is
        hit rather than a value pushed on some earlier event.
        """
        engine = self._engine
        scheduler = engine.scheduler
        allocator = engine.cache.allocator
        metrics.bind_engine_gauges(
            preemptions=lambda: engine.preemption_count,
            grouped_decode_steps=lambda: engine.grouped_decode_steps_total,
            running_requests=lambda: len(scheduler.running),
            waiting_requests=lambda: len(scheduler.waiting),
            kv_blocks_used=lambda: allocator.num_used,
            kv_blocks_free=lambda: allocator.num_free,
            kv_blocks_total=lambda: allocator.num_blocks,
            kv_utilization=lambda: allocator.num_used / allocator.num_blocks,
        )

    async def stream(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        max_new_tokens: int,
        eos_token_ids: frozenset[int],
        sampling: SamplingParams = GREEDY,
        prefix_group_id: str | None = None,
    ) -> AsyncIterator[TokenStreamItem]:
        """Yield generated tokens for one request until it finishes or the caller cancels.

        Submits the request to the loop (carrying its per-request ``sampling`` and optional
        prefix group), then drains its asyncio queue. If the consumer is cancelled (client
        disconnect), the ``finally`` aborts the request so the loop stops decoding it and frees
        its KV — no orphaned work keeps running.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[TokenStreamItem | None] = asyncio.Queue()
        submitted_s = time.perf_counter()
        stream = _Stream(
            request_id=request_id,
            queue=queue,
            loop=loop,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            submitted_s=submitted_s,
        )
        request = Request(
            request_id=request_id,
            prompt_ids=list(prompt_ids),
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            prefix_group_id=prefix_group_id,
            sampling=sampling,
        )
        if self._request_observer is not None:
            self._request_observer.submitted(request, submitted_s)
        with self._lock:
            if self._fatal is not None:
                raise RuntimeError("inference engine is no longer running") from self._fatal
            self._submissions.append((request, stream))
        self._wake.set()

        if self._metrics is not None:
            self._metrics.requests_total.inc()
        first_token_seen = False
        previous_token_s: float | None = None
        try:
            while True:
                item = await queue.get()
                if item is None:  # loop signalled end-of-stream (or refused the submission)
                    if stream.error is not None:
                        raise stream.error
                    return
                if self._metrics is not None:
                    token_s = time.perf_counter()
                    self._metrics.generated_tokens_total.inc()
                    self._metrics.stream_tokens_total.inc()
                    if not first_token_seen:
                        first_token_seen = True
                        self._metrics.ttft_seconds.observe(token_s - submitted_s)
                    elif previous_token_s is not None:
                        self._metrics.itl_seconds.observe(token_s - previous_token_s)
                    previous_token_s = token_s
                yield item
                if item.finish_reason is not None:
                    finished_s = time.perf_counter()
                    if self._metrics is not None:
                        self._metrics.request_latency_seconds.observe(finished_s - submitted_s)
                        self._metrics.requests_completed_total.inc(finish_reason=item.finish_reason)
                    return
        finally:
            stream.aborted = True
            self._wake.set()

    # --- background thread: the only place that touches the engine ---------------------

    def _run_loop(self) -> None:
        try:
            while not self._shutdown.is_set():
                self._drain_submissions()
                self._apply_aborts()
                if not self._engine.scheduler.has_work():
                    # Nothing to do: block until a submission or abort wakes us, idle-quiet.
                    self._wake.wait(timeout=self._idle_sleep_s)
                    self._wake.clear()
                    continue
                step_started_s = time.perf_counter()
                result = self._engine.step()
                self._dispatch(result, admitted_s=step_started_s)
        except BaseException as exc:  # noqa: BLE001 — contain a dead loop, never hang clients
            self._fail_all(exc)

    def _fail_all(self, exc: BaseException) -> None:
        """The step loop died: mark the engine unhealthy and fail everyone instead of hanging.

        Every in-flight stream is closed carrying ``exc`` (its consumer re-raises it), every
        queued-but-undrained submission is failed the same way, and ``_fatal`` is set so later
        submissions are rejected fast in :meth:`stream`. Runs on the exiting loop thread,
        the only one that touches ``_streams``; ``_submissions``/``_fatal`` are shared with handler
        threads, so those are touched under the lock.
        """
        for stream in list(self._streams.values()):
            stream.error = exc
            self._enqueue(stream, None)
        self._streams.clear()
        with self._lock:
            self._fatal = exc
            pending = self._submissions
            self._submissions = []
        for _, stream in pending:
            stream.error = exc
            self._enqueue(stream, None)

    def _drain_submissions(self) -> None:
        with self._lock:
            pending = self._submissions
            self._submissions = []
        for request, stream in pending:
            try:
                self._engine.add_request(request)
            except Exception as exc:  # noqa: BLE001 — a bad submission must not kill the loop
                # Preflight (assert_admissible) already rejects the common oversized case with a
                # clean 4xx, so reaching here means an unexpected engine rejection. Fail just this
                # stream — the consumer re-raises ``error`` — and keep serving every other client.
                stream.error = exc
                self._enqueue(stream, None)
                continue
            self._streams[request.request_id] = stream

    def _apply_aborts(self) -> None:
        """Drop streams whose consumer disconnected, freeing their engine state.

        ``engine.abort`` removes the request from the scheduler and frees its KV at the
        allocator boundary, so the next step never decodes it; the stream is closed so the
        (already gone) consumer never blocks. Idempotent if the request already finished.
        """
        for request_id, stream in list(self._streams.items()):
            if not stream.aborted:
                continue
            self._engine.abort(request_id)
            self._enqueue(stream, None)  # close the (gone) consumer's stream
            self._streams.pop(request_id, None)

    def _dispatch(self, result: StepResult, *, admitted_s: float) -> None:
        if self._metrics is not None:
            for request_id in result.admitted:
                stream = self._streams.get(request_id)
                if stream is None:
                    continue
                self._metrics.requests_admitted_total.inc()
                self._metrics.queue_time_seconds.observe(max(0.0, admitted_s - stream.submitted_s))
        if self._request_observer is not None:
            for request_id in result.admitted:
                if request_id in self._streams:
                    self._request_observer.admitted(request_id, admitted_s)
        for request_id, tokens in result.tokens.items():
            stream = self._streams.get(request_id)
            if stream is None:
                continue
            finished = request_id in result.finished
            last_index = len(tokens) - 1
            for index, token in enumerate(tokens):
                token_id = _as_int(token)
                is_last = finished and index == last_index
                reason = self._finish_reason(stream, token_id) if is_last else None
                self._enqueue(stream, TokenStreamItem(token_id=token_id, finish_reason=reason))
        for request_id in result.finished:
            if self._request_observer is not None:
                self._request_observer.finished(
                    request_id,
                    result.finished_outputs[request_id],
                    time.perf_counter(),
                )
            stream = self._streams.pop(request_id, None)
            if stream is not None:
                self._enqueue(stream, None)  # end-of-stream sentinel

    def _finish_reason(self, stream: _Stream, token_id: int) -> str:
        return "stop" if token_id in stream.eos_token_ids else "length"

    def _enqueue(self, stream: _Stream, item: TokenStreamItem | None) -> None:
        """Hand one item (or the ``None`` end-of-stream sentinel) to the consumer's queue.

        ``call_soon_threadsafe`` lands the put on the consumer's event-loop thread, which is the
        whole bridge: the engine thread never touches asyncio state directly.
        """
        if stream.loop.is_closed():
            return
        stream.loop.call_soon_threadsafe(stream.queue.put_nowait, item)


def _as_int(token: int | torch.Tensor) -> int:
    if isinstance(token, torch.Tensor):
        return int(token.item())
    return token
