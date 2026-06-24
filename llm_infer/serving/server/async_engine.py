"""Bridge the synchronous step loop to asyncio so one batching loop serves many clients.

The engine's ``step()`` runs a model forward; that must never run on the asyncio event
loop or it would stall every other connection. So the forward stays on a dedicated
**background thread** that owns the engine exclusively, and HTTP handlers talk to it
through two queues:

* a thread-safe **submission queue** the handlers push new requests onto, drained at the
  top of each step (so a request that arrives mid-step is admitted next step, exactly the
  continuous-batching behavior we want);
* one **per-request asyncio queue** the loop pushes generated tokens onto, via
  ``loop.call_soon_threadsafe`` so the push lands on the event loop thread. A handler
  ``async for``-s its own queue, fully concurrent with every other handler — many clients
  stream off the single ``step()`` loop, which is the whole point.

Cancellation is cooperative: a disconnected client sets the stream's abort flag; the loop
sees it at the next step boundary, finishes the engine request cleanly, and frees its KV.
The loop is the *only* thread that touches the engine, so there are no locks on engine
state — just the two queue boundaries.
"""

from __future__ import annotations

import asyncio
import itertools
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import torch

from llm_infer.serving.engine import InferenceEngine, StepResult
from llm_infer.serving.request import Request
from llm_infer.serving.server.metrics import ServerMetrics


@dataclass
class TokenStreamItem:
    """One token handed to a streaming handler, plus the finish reason on the last item."""

    token_id: int
    finish_reason: str | None = None


@dataclass
class _Stream:
    """The loop's view of one in-flight request: where to push tokens and whether to stop."""

    request_id: str
    queue: asyncio.Queue[TokenStreamItem | None]
    loop: asyncio.AbstractEventLoop
    max_new_tokens: int
    aborted: bool = False
    eos_token_ids: frozenset[int] = field(default_factory=frozenset)


class AsyncInferenceEngine:
    """Drive a synchronous :class:`InferenceEngine` from asyncio over a background thread.

    Construct it around an already-built engine (the app factory injects one wrapping the
    tiny CPU model in tests, the real Qwen in production). Call :meth:`start` to spin up the
    loop thread and :meth:`stream` per request; :meth:`stop` joins the thread on shutdown.
    """

    def __init__(
        self,
        engine: InferenceEngine,
        *,
        idle_sleep_s: float = 0.001,
        metrics: ServerMetrics | None = None,
    ) -> None:
        self._engine = engine
        self._idle_sleep_s = idle_sleep_s
        self._metrics = metrics
        self._submissions: list[tuple[Request, _Stream]] = []
        self._streams: dict[str, _Stream] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._ids = itertools.count()
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
    ) -> AsyncIterator[TokenStreamItem]:
        """Yield generated tokens for one request until it finishes or the caller cancels.

        Submits the request to the loop, then drains its asyncio queue. If the consumer is
        cancelled (client disconnect), the ``finally`` aborts the request so the loop stops
        decoding it and frees its KV — no orphaned work keeps running.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[TokenStreamItem | None] = asyncio.Queue()
        stream = _Stream(
            request_id=request_id,
            queue=queue,
            loop=loop,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        request = Request(
            request_id=request_id,
            prompt_ids=list(prompt_ids),
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        with self._lock:
            self._submissions.append((request, stream))
        self._wake.set()

        arrival = time.perf_counter()
        if self._metrics is not None:
            self._metrics.requests_total.inc()
        first_token_seen = False
        try:
            while True:
                item = await queue.get()
                if item is None:  # loop signalled end-of-stream
                    return
                if self._metrics is not None:
                    self._metrics.generated_tokens_total.inc()
                    if not first_token_seen:
                        first_token_seen = True
                        self._metrics.ttft_seconds.observe(time.perf_counter() - arrival)
                yield item
                if item.finish_reason is not None:
                    if self._metrics is not None:
                        self._metrics.request_latency_seconds.observe(time.perf_counter() - arrival)
                        self._metrics.requests_completed_total.inc(finish_reason=item.finish_reason)
                    return
        finally:
            stream.aborted = True
            self._wake.set()

    # --- background thread: the only place that touches the engine ---------------------

    def _run_loop(self) -> None:
        while not self._shutdown.is_set():
            self._drain_submissions()
            self._apply_aborts()
            if not self._engine.scheduler.has_work():
                # Nothing to do: block until a submission or abort wakes us, cheap and idle-quiet.
                self._wake.wait(timeout=self._idle_sleep_s)
                self._wake.clear()
                continue
            result = self._engine.step()
            self._dispatch(result)

    def _drain_submissions(self) -> None:
        with self._lock:
            pending = self._submissions
            self._submissions = []
        for request, stream in pending:
            self._streams[request.request_id] = stream
            self._engine.add_request(request)

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

    def _dispatch(self, result: StepResult) -> None:
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
