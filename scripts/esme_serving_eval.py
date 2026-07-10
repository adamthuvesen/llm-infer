"""Serving-pressure evaluation harness for Esme bundle backends.

The harness is deliberately not a headline benchmark. It runs compact workloads that make the
serving techniques interact, records latency/queue/KV/preemption behavior, and only reports
tok/s when the row is reference-gated. Local runs build an in-process engine and ASGI HTTP app so
the same OpenAI handlers, async serving loop, scheduler, paged KV, prefix sharing, speculative
decode, and preemption paths can be observed without Modal/GPU spend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import socket
import sys
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import torch

from llm_infer.benchmarks.report import normalize_at_eos
from llm_infer.model.decode import greedy_decode
from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
from llm_infer.model.grouped_decode_graph import DEFAULT_GROUPED_CAPTURE_SIZES
from llm_infer.model.interface import ModelRuntime
from llm_infer.model.runtime import ATTENTION_BACKEND_CHOICES, load_model_runtime
from llm_infer.serve import BUNDLE_BACKENDS
from llm_infer.serving.engine import InferenceEngine, StepResult
from llm_infer.serving.request import Request
from llm_infer.serving.sampler import GREEDY, SamplingParams
from llm_infer.serving.server import AsyncInferenceEngine, ServerMetrics, create_app
from llm_infer.serving.speculative import SpeculativeDecodingConfig
from llm_infer.tracing import TraceRecorder

Surface = Literal["engine", "asgi-http", "network-http", "external-http"]
Endpoint = Literal["chat", "completions"]
ReferenceState = Literal["pass", "fail", "partial", "skipped", "unavailable"]


@dataclass(frozen=True)
class RequestSignature:
    """Stable key for matching observed engine requests back to their submitted arrivals."""

    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    frequency_penalty: float
    seed: int

    @classmethod
    def from_parts(
        cls, prompt_ids: Sequence[int], max_new_tokens: int, sampling: SamplingParams
    ) -> RequestSignature:
        return cls(
            prompt_ids=tuple(int(token_id) for token_id in prompt_ids),
            max_new_tokens=max_new_tokens,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k,
            presence_penalty=sampling.presence_penalty,
            frequency_penalty=sampling.frequency_penalty,
            seed=sampling.seed,
        )


@dataclass
class ObservedEngineRequest:
    """Engine-side request timings and token ids captured from StepResult."""

    request_id: str
    signature: RequestSignature
    sampling: SamplingParams
    arrival_s: float | None
    admitted_s: float | None = None
    first_token_s: float | None = None
    finished_s: float | None = None
    token_ids: list[int] = field(default_factory=list)
    token_times_s: list[float] = field(default_factory=list)

    @property
    def queue_s(self) -> float | None:
        if self.arrival_s is None or self.admitted_s is None:
            return None
        return max(0.0, self.admitted_s - self.arrival_s)

    @property
    def ttft_s(self) -> float | None:
        if self.arrival_s is None or self.first_token_s is None:
            return None
        return max(0.0, self.first_token_s - self.arrival_s)

    @property
    def latency_s(self) -> float | None:
        if self.arrival_s is None or self.finished_s is None:
            return None
        return max(0.0, self.finished_s - self.arrival_s)


class StepObserver:
    """Capture request, queue, KV, and preemption data from a local engine step loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._expected: dict[RequestSignature, deque[float]] = {}
        self._records: dict[str, ObservedEngineRequest] = {}
        self._kv_utilization: list[float] = []
        self._running_samples: list[int] = []
        self._waiting_samples: list[int] = []
        self._preemption_count = 0

    def expect(self, signature: RequestSignature, arrival_s: float) -> None:
        with self._lock:
            self._expected.setdefault(signature, deque()).append(arrival_s)

    def wrap(self, engine: InferenceEngine) -> None:
        original_step = engine.step

        def observed_step() -> StepResult:
            started = time.perf_counter()
            result = original_step()
            finished = time.perf_counter()
            self.record_step(engine, result, started_s=started, finished_s=finished)
            return result

        engine.step = observed_step

    def record_step(
        self,
        engine: InferenceEngine,
        result: StepResult,
        *,
        started_s: float,
        finished_s: float,
    ) -> None:
        with self._lock:
            for request_id in result.admitted:
                request = engine._requests[request_id]
                sampling = _effective_sampling(engine, request)
                signature = RequestSignature.from_parts(
                    request.prompt_ids, request.max_new_tokens, sampling
                )
                record = self._records.get(request_id)
                if record is None:
                    self._records[request_id] = ObservedEngineRequest(
                        request_id=request_id,
                        signature=signature,
                        sampling=sampling,
                        arrival_s=self._pop_expected_arrival(signature),
                        admitted_s=started_s,
                    )
                elif record.admitted_s is None:
                    record.admitted_s = started_s

            for request_id, tokens in result.tokens.items():
                record = self._records.get(request_id)
                if record is None:
                    continue
                if record.first_token_s is None:
                    record.first_token_s = finished_s
                for token in tokens:
                    record.token_ids.append(_token_to_int(token))
                    record.token_times_s.append(finished_s)

            for request_id in result.finished:
                record = self._records.get(request_id)
                if record is not None:
                    record.finished_s = finished_s

            allocator = engine.cache.allocator
            self._kv_utilization.append(allocator.num_used / allocator.num_blocks)
            self._running_samples.append(len(engine.scheduler.running))
            self._waiting_samples.append(len(engine.scheduler.waiting))
            self._preemption_count = engine.preemption_count

    def snapshot(self) -> tuple[list[ObservedEngineRequest], dict[str, object]]:
        with self._lock:
            records = list(self._records.values())
            kv_samples = list(self._kv_utilization)
            running = list(self._running_samples)
            waiting = list(self._waiting_samples)
            preemptions = self._preemption_count
        return records, {
            "preemption_count": preemptions,
            "kv_utilization_peak": max(kv_samples) if kv_samples else None,
            "kv_utilization_final": kv_samples[-1] if kv_samples else None,
            "max_running_requests": max(running) if running else 0,
            "max_waiting_requests": max(waiting) if waiting else 0,
        }

    def _pop_expected_arrival(self, signature: RequestSignature) -> float | None:
        arrivals = self._expected.get(signature)
        if not arrivals:
            return None
        arrival = arrivals.popleft()
        if not arrivals:
            self._expected.pop(signature, None)
        return arrival


class StreamOutputObserver:
    """Capture server outputs after its existing tensor-to-int conversion."""

    def __init__(self) -> None:
        self._records: dict[str, ObservedEngineRequest] = {}

    def submitted(self, request: Request, submitted_s: float) -> None:
        sampling = request.sampling or GREEDY
        record = ObservedEngineRequest(
            request_id=request.request_id,
            signature=RequestSignature.from_parts(
                request.prompt_ids, request.max_new_tokens, sampling
            ),
            sampling=sampling,
            arrival_s=submitted_s,
        )
        self._records[request.request_id] = record

    def admitted(self, request_id: str, admitted_s: float) -> None:
        record = self._records.get(request_id)
        if record is not None:
            record.admitted_s = admitted_s

    def finished(self, request_id: str, token_ids: list[int], finished_s: float) -> None:
        record = self._records.get(request_id)
        if record is not None:
            record.token_ids = list(token_ids)
            record.finished_s = finished_s

    def reset(self) -> None:
        self._records.clear()

    def snapshot(
        self, engine: InferenceEngine
    ) -> tuple[list[ObservedEngineRequest], dict[str, object]]:
        records = list(self._records.values())
        return records, {
            "preemption_count": engine.preemption_count,
            "kv_utilization_peak": None,
            "kv_utilization_final": engine.cache.allocator.num_used
            / engine.cache.allocator.num_blocks,
            "max_running_requests": None,
            "max_waiting_requests": None,
        }


@dataclass(frozen=True)
class EngineRequestSpec:
    request_id: str
    prompt_ids: tuple[int, ...]
    max_new_tokens: int
    sampling: SamplingParams = GREEDY
    prefix_group_id: str | None = None


@dataclass(frozen=True)
class HttpRequestSpec:
    request_id: str
    endpoint: Endpoint
    prompt: str
    max_new_tokens: int
    stream: bool
    sampling: SamplingParams = GREEDY
    prefix_group_id: str | None = None
    include_stream_usage: bool = True


@dataclass(frozen=True)
class WorkloadConfig:
    block_size: int
    num_blocks: int
    preemption: bool = False
    prefill_chunk_size: int | None = None
    speculative: SpeculativeDecodingConfig | None = None
    grouped_decode_graphs: bool = False
    grouped_capture_sizes: tuple[int, ...] | None = None


@dataclass(frozen=True)
class EngineWorkload:
    name: str
    description: str
    config: WorkloadConfig
    requests: tuple[EngineRequestSpec, ...]


@dataclass(frozen=True)
class HttpWorkload:
    name: str
    description: str
    config: WorkloadConfig
    requests: tuple[HttpRequestSpec, ...]


@dataclass
class ClientRequestResult:
    request_id: str
    status: Literal["ok", "error"]
    output_count: int
    output_unit: str
    start_s: float
    end_s: float
    first_token_s: float | None = None
    token_times_s: list[float] = field(default_factory=list)
    finish_reason: str | None = None
    http_status: int | None = None
    error: str | None = None

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_s is None:
            return None
        return max(0.0, self.first_token_s - self.start_s)

    @property
    def latency_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    @property
    def per_output_s(self) -> float | None:
        if self.output_count < 1:
            return None
        return self.latency_s / self.output_count

    @property
    def itls_s(self) -> list[float]:
        return [
            max(0.0, right - left)
            for left, right in zip(self.token_times_s, self.token_times_s[1:], strict=False)
        ]


def default_engine_workloads() -> tuple[EngineWorkload, ...]:
    repeated = (1, 5, 9, 1, 5, 9)
    long_prompt = (1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7)
    return (
        EngineWorkload(
            name="shared-prefix-greedy",
            description="Four sibling greedy requests share one prompt block group.",
            config=WorkloadConfig(block_size=8, num_blocks=64),
            requests=tuple(
                EngineRequestSpec(
                    request_id=f"prefix-{idx}",
                    prompt_ids=repeated,
                    max_new_tokens=6,
                    prefix_group_id="shared-prefix",
                )
                for idx in range(4)
            ),
        ),
        EngineWorkload(
            name="chunked-long-short",
            description=(
                "One long prompt and three short prompts share the loop under chunked prefill."
            ),
            config=WorkloadConfig(block_size=8, num_blocks=64, prefill_chunk_size=4),
            requests=(
                EngineRequestSpec("long-0", long_prompt, 5),
                EngineRequestSpec("short-0", (2, 5, 8), 5),
                EngineRequestSpec("short-1", (3, 6, 9), 5),
                EngineRequestSpec("short-2", (4, 7, 10), 5),
            ),
        ),
        EngineWorkload(
            name="tight-kv-preemption",
            description=(
                "Three greedy requests overcommit a three-block pool and force recompute "
                "preemption."
            ),
            config=WorkloadConfig(block_size=4, num_blocks=3, preemption=True),
            requests=(
                EngineRequestSpec("preempt-a", (1, 4, 7), 6),
                EngineRequestSpec("preempt-b", (2, 5, 8), 6),
                EngineRequestSpec("preempt-c", (3, 6, 9), 6),
            ),
        ),
        EngineWorkload(
            name="speculative-greedy",
            description=(
                "Repeated prompt suffixes enable prompt-lookup speculation on greedy requests."
            ),
            config=WorkloadConfig(
                block_size=8,
                num_blocks=64,
                speculative=SpeculativeDecodingConfig(max_draft_tokens=3, max_ngram_size=3),
            ),
            requests=(
                EngineRequestSpec("spec-0", repeated, 8),
                EngineRequestSpec("spec-1", (2, 4, 6, 2, 4, 6), 8),
            ),
        ),
    )


def default_http_workloads() -> tuple[HttpWorkload, ...]:
    sampled = SamplingParams(temperature=0.8, top_p=0.95, top_k=8, seed=17)
    return (
        HttpWorkload(
            name="api-streaming-greedy",
            description="Persistent streaming greedy requests for TTFT, ITL, and output rate.",
            config=WorkloadConfig(block_size=8, num_blocks=64),
            requests=tuple(
                HttpRequestSpec(
                    f"greedy-{idx}",
                    "chat",
                    f"Name one inference metric. Request {idx}.",
                    16,
                    True,
                )
                for idx in range(8)
            ),
        ),
        HttpWorkload(
            name="api-streaming-sampled",
            description="Persistent seeded sampling requests for TTFT, ITL, and output rate.",
            config=WorkloadConfig(block_size=8, num_blocks=64),
            requests=tuple(
                HttpRequestSpec(
                    f"sampled-{idx}",
                    "chat",
                    f"Name one inference metric. Request {idx}.",
                    16,
                    True,
                    sampled,
                )
                for idx in range(8)
            ),
        ),
        HttpWorkload(
            name="api-streaming-mixed",
            description=(
                "Streaming chat requests mix short and long prompts through the OpenAI path."
            ),
            config=WorkloadConfig(block_size=8, num_blocks=64, prefill_chunk_size=4),
            requests=(
                HttpRequestSpec(
                    "stream-short-0",
                    "chat",
                    "Name one queueing metric.",
                    6,
                    True,
                    prefix_group_id="api-shared-prefix",
                ),
                HttpRequestSpec(
                    "stream-long-0",
                    "chat",
                    (
                        "Explain why a serving benchmark must check reference output "
                        "before it reports throughput."
                    ),
                    6,
                    True,
                ),
                HttpRequestSpec(
                    "stream-short-1",
                    "chat",
                    "Name one queueing metric.",
                    6,
                    True,
                    prefix_group_id="api-shared-prefix",
                ),
            ),
        ),
        HttpWorkload(
            name="api-blocking-sampled",
            description=(
                "Blocking completions mix greedy and sampled requests on the same HTTP server."
            ),
            config=WorkloadConfig(block_size=8, num_blocks=64),
            requests=(
                HttpRequestSpec("block-greedy-0", "completions", "tok_1 tok_4 tok_7", 6, False),
                HttpRequestSpec(
                    "block-sampled-0",
                    "completions",
                    "tok_2 tok_5 tok_8",
                    6,
                    False,
                    sampled,
                ),
                HttpRequestSpec("block-greedy-1", "completions", "tok_3 tok_6 tok_9", 6, False),
                HttpRequestSpec(
                    "block-sampled-1",
                    "completions",
                    "tok_1 tok_5 tok_9",
                    6,
                    False,
                    sampled,
                ),
            ),
        ),
    )


def phase0_http_workloads(
    request_count: int,
    *,
    max_new_tokens: int = 128,
    block_size: int = 64,
    num_blocks: int = 1024,
    grouped_decode_graphs: bool = False,
    grouped_capture_sizes: tuple[int, ...] | None = None,
) -> tuple[HttpWorkload, HttpWorkload]:
    """Build paired greedy and seeded-sampling workloads with the same HTTP shape."""
    if request_count < 1:
        raise ValueError(f"request_count must be >= 1; got {request_count}")
    sampled = SamplingParams(temperature=0.8, top_p=0.95, top_k=32, seed=17)
    config = WorkloadConfig(
        block_size=block_size,
        num_blocks=num_blocks,
        grouped_decode_graphs=grouped_decode_graphs,
        grouped_capture_sizes=grouped_capture_sizes,
    )

    def requests(prefix: str, sampling: SamplingParams) -> tuple[HttpRequestSpec, ...]:
        return tuple(
            HttpRequestSpec(
                request_id=f"{prefix}-{index}",
                endpoint="chat",
                prompt="Explain one inference performance metric.",
                max_new_tokens=max_new_tokens,
                stream=True,
                sampling=sampling,
            )
            for index in range(request_count)
        )

    return (
        HttpWorkload(
            name=f"phase0-http-greedy-b{request_count}",
            description="Persistent streaming greedy Phase 0 serving measurement.",
            config=config,
            requests=requests("greedy", GREEDY),
        ),
        HttpWorkload(
            name=f"phase0-http-sampled-b{request_count}",
            description="Persistent streaming sampled Phase 0 serving measurement.",
            config=config,
            requests=requests("sampled", sampled),
        ),
    )


async def run_local_eval(
    runtime: ModelRuntime,
    *,
    workload_names: set[str] | None = None,
    device: str = "cpu",
) -> list[dict[str, object]]:
    """Run selected local engine and ASGI HTTP workloads against one loaded runtime."""
    results: list[dict[str, object]] = []
    for workload in default_engine_workloads():
        if _selected(workload.name, workload_names):
            results.append(await run_engine_workload(runtime, workload, device=device))
    for workload in default_http_workloads():
        if _selected(workload.name, workload_names):
            results.append(await run_asgi_http_workload(runtime, workload, device=device))
    return results


async def run_engine_workload(
    runtime: ModelRuntime,
    workload: EngineWorkload,
    *,
    device: str = "cpu",
) -> dict[str, object]:
    observer = StepObserver()
    trace = TraceRecorder()
    engine_build_started_s = time.perf_counter()
    engine = _build_engine(runtime, workload.config, device=device, trace=trace)
    _synchronize_device(device)
    engine_build_s = time.perf_counter() - engine_build_started_s
    observer.wrap(engine)
    async_engine = AsyncInferenceEngine(engine)

    async def consume(spec: EngineRequestSpec) -> ClientRequestResult:
        start = time.perf_counter()
        signature = RequestSignature.from_parts(spec.prompt_ids, spec.max_new_tokens, spec.sampling)
        observer.expect(signature, start)
        token_count = 0
        first_token_s: float | None = None
        token_times_s: list[float] = []
        finish_reason: str | None = None
        try:
            async for item in async_engine.stream(
                request_id=spec.request_id,
                prompt_ids=list(spec.prompt_ids),
                max_new_tokens=spec.max_new_tokens,
                eos_token_ids=runtime.eos_token_ids,
                sampling=spec.sampling,
                prefix_group_id=spec.prefix_group_id,
            ):
                token_count += 1
                token_s = time.perf_counter()
                if first_token_s is None:
                    first_token_s = token_s
                token_times_s.append(token_s)
                if item.finish_reason is not None:
                    finish_reason = item.finish_reason
            return ClientRequestResult(
                request_id=spec.request_id,
                status="ok",
                output_count=token_count,
                output_unit="tokens",
                start_s=start,
                end_s=time.perf_counter(),
                first_token_s=first_token_s,
                token_times_s=token_times_s,
                finish_reason=finish_reason,
            )
        except Exception as exc:  # noqa: BLE001 - one bad request is an eval record, not a crash.
            return ClientRequestResult(
                request_id=spec.request_id,
                status="error",
                output_count=token_count,
                output_unit="tokens",
                start_s=start,
                end_s=time.perf_counter(),
                first_token_s=first_token_s,
                token_times_s=token_times_s,
                error=f"{type(exc).__name__}: {exc}",
            )

    # Submit every request before the engine loop starts: a consumer's ``stream()`` call
    # enqueues its submission synchronously before its first await, so one loop pass here
    # registers the whole workload. Starting the engine first races admission against the
    # step loop — a shared-prefix group could be split across steps and prefill its
    # siblings separately, which the workload's trace expectations must not depend on.
    consumers = [asyncio.create_task(consume(spec)) for spec in workload.requests]
    await asyncio.sleep(0)
    engine_start_started_s = time.perf_counter()
    async_engine.start()
    engine_start_s = time.perf_counter() - engine_start_started_s
    wall_start = time.perf_counter()
    try:
        client_results = await asyncio.gather(*consumers)
    finally:
        async_engine.stop()
    wall_s = time.perf_counter() - wall_start

    observed, engine_metrics = observer.snapshot()
    return _workload_result(
        name=workload.name,
        surface="engine",
        description=workload.description,
        config=workload.config,
        client_results=client_results,
        observed=observed,
        engine_metrics=engine_metrics,
        trace=trace,
        runtime=runtime,
        wall_s=wall_s,
        timing_scope={
            "engine_build_s": engine_build_s,
            "server_start_s": engine_start_s,
            "warmup_s": 0.0,
            "steady_state_wall_s": wall_s,
            "steady_state_excludes": ["engine_build", "engine_start"],
        },
    )


async def run_asgi_http_workload(
    runtime: ModelRuntime,
    workload: HttpWorkload,
    *,
    device: str = "cpu",
    reference_runtime: ModelRuntime | None = None,
) -> dict[str, object]:
    observer = StepObserver()
    trace = TraceRecorder()
    engine_build_started_s = time.perf_counter()
    engine = _build_engine(runtime, workload.config, device=device, trace=trace)
    _synchronize_device(device)
    engine_build_s = time.perf_counter() - engine_build_started_s
    observer.wrap(engine)
    metrics = ServerMetrics()
    async_engine = AsyncInferenceEngine(engine, metrics=metrics)
    app = create_app(
        async_engine=async_engine,
        tokenizer=runtime.tokenizer,
        model_id=runtime.model_id,
        eos_token_ids=runtime.eos_token_ids,
        metrics=metrics,
    )
    transport = httpx.ASGITransport(app=app)
    timeout = httpx.Timeout(60.0)

    async def one(client: httpx.AsyncClient, spec: HttpRequestSpec) -> ClientRequestResult:
        prompt_ids = _http_prompt_ids(runtime.tokenizer, spec)
        signature = RequestSignature.from_parts(prompt_ids, spec.max_new_tokens, spec.sampling)
        start = time.perf_counter()
        observer.expect(signature, start)
        if spec.stream:
            return await _run_streaming_http(client, spec, runtime.model_id, start)
        return await _run_blocking_http(client, spec, runtime.model_id, start)

    server_start_started_s = time.perf_counter()
    async with app.router.lifespan_context(app):
        server_start_s = time.perf_counter() - server_start_started_s
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=timeout
        ) as client:
            before_metrics = _parse_metrics((await client.get("/metrics")).text)
            _synchronize_device(device)
            wall_start = time.perf_counter()
            client_results = await asyncio.gather(
                *(one(client, spec) for spec in workload.requests)
            )
            wall_s = time.perf_counter() - wall_start
            after_metrics = _parse_metrics((await client.get("/metrics")).text)

    observed, engine_metrics = observer.snapshot()
    engine_metrics.update(_public_metric_delta(before_metrics, after_metrics))
    return _workload_result(
        name=workload.name,
        surface="asgi-http",
        description=workload.description,
        config=workload.config,
        client_results=client_results,
        observed=observed,
        engine_metrics=engine_metrics,
        trace=trace,
        runtime=reference_runtime or runtime,
        wall_s=wall_s,
        timing_scope={
            "engine_build_s": engine_build_s,
            "server_start_s": server_start_s,
            "warmup_s": 0.0,
            "steady_state_wall_s": wall_s,
            "steady_state_excludes": ["engine_build", "server_start", "metrics_scrape"],
        },
    )


async def run_network_http_workload(
    runtime: ModelRuntime,
    workload: HttpWorkload,
    *,
    device: str = "cpu",
    reference_runtime: ModelRuntime | None = None,
    warmup_runs: int = 0,
    measured_runs: int = 1,
) -> dict[str, object]:
    """Measure streaming through localhost Uvicorn on the normal untraced decode path."""
    import uvicorn

    if warmup_runs < 0:
        raise ValueError(f"warmup_runs must be >= 0; got {warmup_runs}")
    if measured_runs < 1:
        raise ValueError(f"measured_runs must be >= 1; got {measured_runs}")
    observer = StreamOutputObserver()
    engine_build_started_s = time.perf_counter()
    # Tracing forces one-token decode windows, so attaching a TraceRecorder here would change
    # the production path being measured. StreamOutputObserver captures the already-materialized
    # token ids at the server boundary without a second CUDA tensor-to-host conversion.
    engine = _build_engine(runtime, workload.config, device=device, trace=None)
    _synchronize_device(device)
    engine_build_s = time.perf_counter() - engine_build_started_s
    metrics = ServerMetrics()
    async_engine = AsyncInferenceEngine(engine, metrics=metrics, request_observer=observer)
    app = create_app(
        async_engine=async_engine,
        tokenizer=runtime.tokenizer,
        model_id=runtime.model_id,
        eos_token_ids=runtime.eos_token_ids,
        metrics=metrics,
    )

    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = int(port_socket.getsockname()[1])

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    server.install_signal_handlers = lambda: None
    server_started_s = time.perf_counter()
    server_task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if server_task.done():
                await server_task
                raise RuntimeError("Uvicorn exited before accepting requests")
            if time.perf_counter() - server_started_s > 30.0:
                raise TimeoutError("Uvicorn did not start within 30 seconds")
            await asyncio.sleep(0.01)
        server_start_s = time.perf_counter() - server_started_s

        timeout = httpx.Timeout(180.0)
        connection_limit = max(1, len(workload.requests))
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}",
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=connection_limit,
                max_keepalive_connections=connection_limit,
            ),
        ) as client:

            async def one(spec: HttpRequestSpec) -> ClientRequestResult:
                start = time.perf_counter()
                if spec.stream:
                    return await _run_streaming_http(client, spec, runtime.model_id, start)
                return await _run_blocking_http(client, spec, runtime.model_id, start)

            warmup_started_s = time.perf_counter()
            for _ in range(warmup_runs):
                warmup_results = await asyncio.gather(*(one(spec) for spec in workload.requests))
                failures = [result.error for result in warmup_results if result.status == "error"]
                if failures:
                    raise RuntimeError(f"serving warmup failed: {failures[0]}")
            warmup_s = time.perf_counter() - warmup_started_s
            observer.reset()
            before_metrics = _parse_metrics((await client.get("/metrics")).text)
            client_results: list[ClientRequestResult] = []
            wall_s = 0.0
            for _ in range(measured_runs):
                _synchronize_device(device)
                wall_start = time.perf_counter()
                run_results = await asyncio.gather(*(one(spec) for spec in workload.requests))
                _synchronize_device(device)
                wall_s += time.perf_counter() - wall_start
                client_results.extend(run_results)
            after_metrics = _parse_metrics((await client.get("/metrics")).text)
    finally:
        server.should_exit = True
        await server_task

    observed, engine_metrics = observer.snapshot(engine)
    engine_metrics.update(_public_metric_delta(before_metrics, after_metrics))
    return _workload_result(
        name=workload.name,
        surface="network-http",
        description=workload.description,
        config=workload.config,
        client_results=client_results,
        observed=observed,
        engine_metrics=engine_metrics,
        trace=None,
        runtime=reference_runtime or runtime,
        wall_s=wall_s,
        timing_scope={
            "engine_build_s": engine_build_s,
            "server_start_s": server_start_s,
            "warmup_runs": warmup_runs,
            "warmup_s": warmup_s,
            "measured_runs": measured_runs,
            "steady_state_wall_s": wall_s,
            "transport": "localhost Uvicorn TCP",
            "max_connections": connection_limit,
            "reference_capture": "one finished-output list copy per request",
            "steady_state_excludes": ["engine_build", "server_start", "metrics_scrape"],
        },
    )


async def run_external_http_eval(
    *,
    base_url: str,
    model_id: str,
    workload_names: set[str] | None = None,
    timeout_s: float = 60.0,
    warmup_requests: int = 1,
) -> list[dict[str, object]]:
    """Run HTTP-only workloads against an already running server.

    External mode cannot see engine request ids or the source bundle oracle, so it records
    transport metrics and marks reference-gated throughput unavailable.
    """
    timeout = httpx.Timeout(timeout_s)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout) as client:
        results: list[dict[str, object]] = []
        for workload in default_http_workloads():
            if not _selected(workload.name, workload_names):
                continue
            warmup_started_s = time.perf_counter()
            if warmup_requests:
                warmup_spec = workload.requests[0]
                await asyncio.gather(
                    *(
                        _run_streaming_http(client, warmup_spec, model_id, time.perf_counter())
                        if warmup_spec.stream
                        else _run_blocking_http(client, warmup_spec, model_id, time.perf_counter())
                        for _ in range(warmup_requests)
                    )
                )
            warmup_s = time.perf_counter() - warmup_started_s
            before_metrics = await _external_metric_samples(client)
            wall_start = time.perf_counter()
            client_results = await asyncio.gather(
                *(
                    _run_streaming_http(client, spec, model_id, time.perf_counter())
                    if spec.stream
                    else _run_blocking_http(client, spec, model_id, time.perf_counter())
                    for spec in workload.requests
                )
            )
            wall_s = time.perf_counter() - wall_start
            after_metrics = await _external_metric_samples(client)
            engine_metrics = _external_metrics(before_metrics, after_metrics)
            results.append(
                _external_workload_result(
                    workload=workload,
                    client_results=client_results,
                    wall_s=wall_s,
                    engine_metrics=engine_metrics,
                    timing_scope={
                        "engine_build_s": None,
                        "server_start_s": None,
                        "warmup_requests": warmup_requests,
                        "warmup_s": warmup_s,
                        "steady_state_wall_s": wall_s,
                        "steady_state_excludes": [
                            "external_server_startup",
                            "engine_build",
                            "kv_pool_allocation",
                            "warmup",
                            "metrics_scrape",
                        ],
                    },
                )
            )
        return results


def _build_engine(
    runtime: ModelRuntime,
    config: WorkloadConfig,
    *,
    device: str,
    trace: TraceRecorder | None,
) -> InferenceEngine:
    return InferenceEngine(
        runtime.model,
        block_size=config.block_size,
        num_blocks=config.num_blocks,
        device=device,
        capabilities=runtime.capabilities,
        preemption=config.preemption,
        prefill_chunk_size=config.prefill_chunk_size,
        speculative=config.speculative,
        trace=trace,
        grouped_decode_graphs=config.grouped_decode_graphs,
        grouped_capture_sizes=config.grouped_capture_sizes or DEFAULT_GROUPED_CAPTURE_SIZES,
    )


def _synchronize_device(device: str) -> None:
    target = torch.device(device)
    if target.type == "cuda":
        torch.cuda.synchronize(target)


async def _run_streaming_http(
    client: httpx.AsyncClient,
    spec: HttpRequestSpec,
    model_id: str,
    start: float,
) -> ClientRequestResult:
    output_deltas = 0
    first_token_s: float | None = None
    token_times_s: list[float] = []
    finish_reason: str | None = None
    usage_output_tokens: int | None = None
    try:
        async with client.stream(
            "POST", _endpoint_path(spec.endpoint), json=_payload(spec, model_id)
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:240]
                return ClientRequestResult(
                    request_id=spec.request_id,
                    status="error",
                    output_count=0,
                    output_unit="deltas",
                    start_s=start,
                    end_s=time.perf_counter(),
                    http_status=resp.status_code,
                    error=f"HTTP {resp.status_code}: {body}",
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    usage_output_tokens = int(usage.get("completion_tokens", 0))
                    continue
                choice = chunk["choices"][0]
                if spec.endpoint == "chat":
                    content = choice.get("delta", {}).get("content")
                else:
                    content = choice.get("text")
                if content:
                    token_s = time.perf_counter()
                    output_deltas += 1
                    if first_token_s is None:
                        first_token_s = token_s
                    token_times_s.append(token_s)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        return ClientRequestResult(
            request_id=spec.request_id,
            status="ok",
            output_count=usage_output_tokens if usage_output_tokens is not None else output_deltas,
            output_unit="tokens" if usage_output_tokens is not None else "deltas",
            start_s=start,
            end_s=time.perf_counter(),
            first_token_s=first_token_s,
            token_times_s=token_times_s,
            finish_reason=finish_reason,
            http_status=200,
        )
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        return ClientRequestResult(
            request_id=spec.request_id,
            status="error",
            output_count=output_deltas,
            output_unit="deltas",
            start_s=start,
            end_s=time.perf_counter(),
            first_token_s=first_token_s,
            token_times_s=token_times_s,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _run_blocking_http(
    client: httpx.AsyncClient,
    spec: HttpRequestSpec,
    model_id: str,
    start: float,
) -> ClientRequestResult:
    try:
        resp = await client.post(_endpoint_path(spec.endpoint), json=_payload(spec, model_id))
        end = time.perf_counter()
        if resp.status_code != 200:
            return ClientRequestResult(
                request_id=spec.request_id,
                status="error",
                output_count=0,
                output_unit="tokens",
                start_s=start,
                end_s=end,
                first_token_s=None,
                http_status=resp.status_code,
                error=f"HTTP {resp.status_code}: {resp.text[:240]}",
            )
        body = resp.json()
        usage = body.get("usage", {})
        finish_reason = body["choices"][0].get("finish_reason")
        return ClientRequestResult(
            request_id=spec.request_id,
            status="ok",
            output_count=int(usage.get("completion_tokens", 0)),
            output_unit="tokens",
            start_s=start,
            end_s=end,
            first_token_s=end,
            finish_reason=finish_reason,
            http_status=resp.status_code,
        )
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
        return ClientRequestResult(
            request_id=spec.request_id,
            status="error",
            output_count=0,
            output_unit="tokens",
            start_s=start,
            end_s=time.perf_counter(),
            error=f"{type(exc).__name__}: {exc}",
        )


def _workload_result(
    *,
    name: str,
    surface: Surface,
    description: str,
    config: WorkloadConfig,
    client_results: list[ClientRequestResult],
    observed: list[ObservedEngineRequest],
    engine_metrics: dict[str, object],
    trace: TraceRecorder | None,
    runtime: ModelRuntime,
    wall_s: float,
    timing_scope: dict[str, object],
) -> dict[str, object]:
    reference = _reference_summary(runtime, observed, block_size=config.block_size)
    output_tokens = sum(len(record.token_ids) for record in observed)
    client_summary = _client_summary(client_results)
    requests_total = len(client_results)
    requests_completed = sum(1 for result in client_results if result.status == "ok")
    requests_failed = sum(1 for result in client_results if result.status == "error")
    observed_count = len(observed)
    unobserved_count = max(0, requests_total - observed_count)
    queue_times = [record.queue_s for record in observed if record.queue_s is not None]
    engine_ttfts = [record.ttft_s for record in observed if record.ttft_s is not None]
    client_ttfts = [value for value in client_summary["ttft_values"] if isinstance(value, float)]
    latencies = [
        value for value in client_summary["latency_values"] if isinstance(value, float)
    ] or [value for record in observed if isinstance((value := record.latency_s), float)]
    itls = (
        [itl for result in client_results for itl in result.itls_s]
        if surface in {"asgi-http", "network-http"}
        else _per_token_latencies(observed, client_results)
    )
    speed_status = _speed_status(
        reference["status"],
        requests_total=requests_total,
        requests_completed=requests_completed,
        observed_count=observed_count,
    )
    throughput = output_tokens / wall_s if speed_status == "reported" and wall_s > 0 else None
    observed_rate = output_tokens / wall_s if output_tokens > 0 and wall_s > 0 else None
    reference_policy_status = (
        "accepted_numerical"
        if reference["status"] == "pass"
        and any(detail.get("status") == "pass_tie" for detail in reference["details"])
        else "exact"
        if reference["status"] == "pass"
        else "failed"
    )

    return {
        "policy_version": 2,
        "reference_status": reference_policy_status,
        "parity_status": "not_applicable",
        "headline_eligible": (
            speed_status == "reported" and throughput is not None and throughput > 0
        ),
        "name": name,
        "surface": surface,
        "description": description,
        "config": _config_dict(config),
        "wall_s": wall_s,
        "timing_scope": timing_scope,
        "requests": [result_to_dict(result) for result in client_results],
        "metrics": {
            "requests_total": requests_total,
            "requests_completed": requests_completed,
            "requests_failed": requests_failed,
            "observed_requests": observed_count,
            "unobserved_requests": unobserved_count,
            "output_tokens": output_tokens,
            "output_token_source": "engine_observer",
            "throughput_tokens_per_s": throughput,
            "raw_throughput_tokens_per_s": observed_rate,
            "throughput_status": speed_status,
            "observed_output_tokens_per_s": observed_rate,
            "ttft_p50_s": _percentile(client_ttfts, 50),
            "ttft_p95_s": _percentile(client_ttfts, 95),
            "ttft_p99_s": _percentile(client_ttfts, 99),
            "engine_ttft_p50_s": _percentile(engine_ttfts, 50),
            "engine_ttft_p95_s": _percentile(engine_ttfts, 95),
            "engine_ttft_p99_s": _percentile(engine_ttfts, 99),
            "latency_p50_s": _percentile(latencies, 50),
            "latency_p95_s": _percentile(latencies, 95),
            "latency_p99_s": _percentile(latencies, 99),
            "itl_p50_s": _percentile(itls, 50),
            "itl_p95_s": _percentile(itls, 95),
            "per_token_latency_p50_s": _percentile(itls, 50),
            "per_token_latency_p99_s": _percentile(itls, 99),
            "queue_time_p50_s": _percentile(queue_times, 50),
            "queue_time_p95_s": _percentile(queue_times, 95),
            "queue_time_p99_s": _percentile(queue_times, 99),
            **engine_metrics,
        },
        "reference": reference,
        "trace": _trace_summary(trace),
    }


def _external_workload_result(
    *,
    workload: HttpWorkload,
    client_results: list[ClientRequestResult],
    wall_s: float,
    engine_metrics: dict[str, object],
    timing_scope: dict[str, object],
) -> dict[str, object]:
    client_summary = _client_summary(client_results)
    public_output_tokens = engine_metrics.get("public_stream_tokens")
    total_output = (
        int(public_output_tokens)
        if isinstance(public_output_tokens, float | int)
        else sum(result.output_count for result in client_results if result.status == "ok")
    )
    return {
        "name": workload.name,
        "surface": "external-http",
        "description": workload.description,
        "config": _config_dict(workload.config),
        "wall_s": wall_s,
        "timing_scope": timing_scope,
        "requests": [result_to_dict(result) for result in client_results],
        "metrics": {
            "requests_total": len(client_results),
            "requests_completed": sum(1 for result in client_results if result.status == "ok"),
            "requests_failed": sum(1 for result in client_results if result.status == "error"),
            "output_tokens": total_output,
            "output_token_source": (
                "public_server_metrics"
                if isinstance(public_output_tokens, float | int)
                else "client_usage_or_deltas"
            ),
            "throughput_tokens_per_s": None,
            "throughput_status": "not_reported_reference_unavailable",
            "observed_output_tokens_per_s": total_output / wall_s if wall_s > 0 else None,
            "ttft_p50_s": _percentile(client_summary["ttft_values"], 50),
            "ttft_p95_s": _percentile(client_summary["ttft_values"], 95),
            "ttft_p99_s": _percentile(client_summary["ttft_values"], 99),
            "latency_p50_s": _percentile(client_summary["latency_values"], 50),
            "latency_p95_s": _percentile(client_summary["latency_values"], 95),
            "latency_p99_s": _percentile(client_summary["latency_values"], 99),
            "itl_p50_s": _percentile(client_summary["itl_values"], 50),
            "itl_p95_s": _percentile(client_summary["itl_values"], 95),
            "per_token_latency_p50_s": _percentile(client_summary["itl_values"], 50),
            "per_token_latency_p99_s": _percentile(client_summary["itl_values"], 99),
            "queue_time_p50_s": engine_metrics.get("public_queue_time_p50_bucket_upper_s"),
            "queue_time_p95_s": engine_metrics.get("public_queue_time_p95_bucket_upper_s"),
            "queue_time_p99_s": None,
            **engine_metrics,
        },
        "reference": {
            "status": "unavailable",
            "passed": 0,
            "failed": 0,
            "skipped_sampled": 0,
            "details": [
                {
                    "reason": (
                        "external HTTP mode cannot observe engine token ids or the bundle "
                        "oracle; no speed row is reported"
                    )
                }
            ],
        },
        "trace": {"available": False},
    }


def _reference_summary(
    runtime: ModelRuntime,
    records: list[ObservedEngineRequest],
    *,
    block_size: int,
) -> dict[str, object]:
    from llm_infer.benchmarks.esme_three_way import BF16_AGREEMENT_TOLERANCE
    from llm_infer.validation.tie_tolerance import compare_under_tie_tolerance

    details: list[dict[str, object]] = []
    references: dict[RequestSignature, list[int]] = {}
    passed = 0
    failed = 0
    skipped = 0
    for record in records:
        reference = references.get(record.signature)
        if reference is None:
            reference = _reference_decode(runtime, record, block_size=block_size)
            references[record.signature] = reference
        got = normalize_at_eos(record.token_ids, runtime.eos_token_ids)
        expected = normalize_at_eos(reference, runtime.eos_token_ids)
        if got == expected:
            passed += 1
            details.append(
                {
                    "request_id": record.request_id,
                    "status": "pass",
                    "output_tokens": len(record.token_ids),
                }
            )
            continue
        if record.sampling.is_greedy:
            tie_result = compare_under_tie_tolerance(
                runtime.model,
                list(record.signature.prompt_ids),
                got,
                expected,
                tolerance=BF16_AGREEMENT_TOLERANCE,
            )
            if tie_result.ok and tie_result.divergence is not None:
                passed += 1
                details.append(
                    {
                        "request_id": record.request_id,
                        "status": "pass_tie",
                        "step": tie_result.divergence.step,
                        "gap": tie_result.divergence.reference_gap,
                        "output_tokens": len(record.token_ids),
                    }
                )
                continue
        failed += 1
        details.append(
            {
                "request_id": record.request_id,
                "status": "fail",
                "first_mismatch": _first_mismatch(got, expected),
                "got_prefix": got[:12],
                "reference_prefix": expected[:12],
            }
        )

    if not records:
        status: ReferenceState = "unavailable"
    elif failed:
        status = "fail"
    elif skipped and passed:
        status = "partial"
    elif skipped:
        status = "skipped"
    else:
        status = "pass"
    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "skipped_sampled": skipped,
        "details": details,
    }


def _reference_decode(
    runtime: ModelRuntime, record: ObservedEngineRequest, *, block_size: int
) -> list[int]:
    if record.sampling.is_greedy:
        return greedy_decode(
            runtime.model,
            list(record.signature.prompt_ids),
            max_new_tokens=record.signature.max_new_tokens,
            eos_token_ids=set(runtime.eos_token_ids),
        )

    # Seeded sampled output is not stable across fp32 and bf16 logits. Its serving reference is
    # therefore one request through the same paged backend and dtype. This checks that batching,
    # HTTP dispatch, and per-request RNG state do not change the sampled continuation; greedy
    # model math remains gated against the fp32 full-recompute oracle above.
    required_blocks = (
        len(record.signature.prompt_ids) + record.signature.max_new_tokens + block_size - 1
    ) // block_size
    engine = InferenceEngine(
        runtime.model,
        block_size=block_size,
        num_blocks=required_blocks + 2,
        device=str(runtime.model.device),
        capabilities=runtime.capabilities,
    )
    request_id = "sampled-reference"
    engine.add_request(
        Request(
            request_id,
            list(record.signature.prompt_ids),
            record.signature.max_new_tokens,
            runtime.eos_token_ids,
            sampling=record.sampling,
        )
    )
    return engine.run()[request_id]


def _trace_summary(trace: TraceRecorder | None) -> dict[str, object]:
    if trace is None:
        return {
            "available": False,
            "reason": "disabled because tracing changes deferred decode-window behavior",
        }
    events = trace.events
    by_event: dict[str, int] = {}
    by_token_source: dict[str, int] = {}
    for event in events:
        by_event[event.event] = by_event.get(event.event, 0) + 1
        if event.token_source is not None:
            by_token_source[event.token_source] = by_token_source.get(event.token_source, 0) + 1
    return {
        "available": True,
        "events": len(events),
        "by_event": by_event,
        "decode_steps_by_source": by_token_source,
    }


def _client_summary(results: list[ClientRequestResult]) -> dict[str, list[float | None]]:
    return {
        "ttft_values": [result.ttft_s for result in results if result.ttft_s is not None],
        "latency_values": [result.latency_s for result in results],
        "per_output_values": [result.per_output_s for result in results],
        "itl_values": [itl for result in results for itl in result.itls_s],
    }


def _per_token_latencies(
    observed: list[ObservedEngineRequest], client_results: list[ClientRequestResult]
) -> list[float]:
    values: list[float] = []
    for record in observed:
        if len(record.token_times_s) > 1:
            values.extend(
                max(0.0, right - left)
                for left, right in zip(record.token_times_s, record.token_times_s[1:], strict=False)
            )
        elif record.latency_s is not None and record.token_ids:
            values.append(record.latency_s / len(record.token_ids))
    if values:
        return values
    return [value for result in client_results if (value := result.per_output_s) is not None]


def result_to_dict(result: ClientRequestResult) -> dict[str, object]:
    return {
        "request_id": result.request_id,
        "status": result.status,
        "http_status": result.http_status,
        "finish_reason": result.finish_reason,
        "output_count": result.output_count,
        "output_unit": result.output_unit,
        "ttft_s": result.ttft_s,
        "latency_s": result.latency_s,
        "per_output_s": result.per_output_s,
        "itls_s": result.itls_s,
        "error": result.error,
    }


def _speed_status(
    reference_status: object,
    *,
    requests_total: int,
    requests_completed: int,
    observed_count: int,
) -> str:
    if requests_completed < requests_total:
        return "not_reported_requests_failed"
    if observed_count < requests_total:
        return "not_reported_requests_unobserved"
    if reference_status == "pass":
        return "reported"
    if reference_status == "fail":
        return "not_reported_reference_failed"
    if reference_status == "partial":
        return "not_reported_contains_sampled_requests"
    if reference_status == "skipped":
        return "not_reported_all_requests_sampled"
    return "not_reported_reference_unavailable"


def _first_mismatch(got: list[int], expected: list[int]) -> dict[str, object]:
    for index, (left, right) in enumerate(zip(got, expected, strict=False)):
        if left != right:
            return {"index": index, "got": left, "reference": right}
    return {
        "index": min(len(got), len(expected)),
        "got_len": len(got),
        "reference_len": len(expected),
    }


def _percentile(values: Iterable[float | None], pct: float) -> float | None:
    clean = sorted(value for value in values if value is not None)
    if not clean:
        return None
    rank = 0 if pct <= 0 else math.ceil(pct / 100.0 * len(clean)) - 1
    rank = max(0, min(len(clean) - 1, rank))
    return clean[rank]


def _endpoint_path(endpoint: Endpoint) -> str:
    if endpoint == "chat":
        return "/v1/chat/completions"
    return "/v1/completions"


def _payload(spec: HttpRequestSpec, model_id: str) -> dict[str, object]:
    base: dict[str, object] = {
        "model": model_id,
        "max_tokens": spec.max_new_tokens,
        "stream": spec.stream,
        **_sampling_dict(spec.sampling),
    }
    if spec.stream and spec.include_stream_usage:
        base["stream_options"] = {"include_usage": True}
    if spec.prefix_group_id is not None:
        base["llm_infer_prefix_group_id"] = spec.prefix_group_id
    if spec.endpoint == "chat":
        base["messages"] = [{"role": "user", "content": spec.prompt}]
    else:
        base["prompt"] = spec.prompt
    return base


def _http_prompt_ids(tokenizer: object, spec: HttpRequestSpec) -> tuple[int, ...]:
    if spec.endpoint == "chat":
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": spec.prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        return tuple(_template_token_ids(rendered))
    return tuple(int(token_id) for token_id in tokenizer.encode(spec.prompt))


def _template_token_ids(rendered: object) -> list[int]:
    ids = rendered["input_ids"] if isinstance(rendered, dict) else rendered
    values = list(ids)
    if values and isinstance(values[0], (list, tuple)):
        values = list(values[0])
    return [int(token_id) for token_id in values]


def _sampling_dict(sampling: SamplingParams) -> dict[str, object]:
    return {
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "presence_penalty": sampling.presence_penalty,
        "frequency_penalty": sampling.frequency_penalty,
        "seed": sampling.seed,
    }


def _config_dict(config: WorkloadConfig) -> dict[str, object]:
    return {
        "block_size": config.block_size,
        "num_blocks": config.num_blocks,
        "preemption": config.preemption,
        "prefill_chunk_size": config.prefill_chunk_size,
        "speculative": (
            {
                "max_draft_tokens": config.speculative.max_draft_tokens,
                "max_ngram_size": config.speculative.max_ngram_size,
            }
            if config.speculative is not None
            else None
        ),
    }


async def _external_metric_samples(client: httpx.AsyncClient) -> dict[str, float] | None:
    try:
        response = await client.get("/metrics")
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    return _parse_metrics(response.text)


def _external_metrics(
    before: dict[str, float] | None, after: dict[str, float] | None
) -> dict[str, object]:
    if after is None:
        return {
            "preemption_count": None,
            "kv_utilization_peak": None,
            "kv_utilization_final": None,
            "max_running_requests": None,
            "max_waiting_requests": None,
            **_empty_public_metrics(),
        }
    public = _public_metric_delta(before or {}, after)
    return {
        "preemption_count": after.get("llm_infer_preemptions_total"),
        "kv_utilization_peak": None,
        "kv_utilization_final": after.get("llm_infer_kv_utilization_ratio"),
        "max_running_requests": after.get("llm_infer_running_requests"),
        "max_waiting_requests": after.get("llm_infer_waiting_requests"),
        **public,
    }


def _public_metric_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, object]:
    admitted = _sample_delta(before, after, "llm_infer_requests_admitted_total")
    queue_count = _sample_delta(before, after, "llm_infer_queue_time_seconds_count")
    queue_sum = _sample_delta(before, after, "llm_infer_queue_time_seconds_sum")
    return {
        "public_requests_admitted": admitted,
        "public_stream_tokens": _sample_delta(before, after, "llm_infer_stream_tokens_total"),
        "public_grouped_decode_steps": _sample_delta(
            before, after, "llm_infer_grouped_decode_steps_total"
        ),
        "public_queue_time_count": queue_count,
        "public_queue_time_sum_s": queue_sum,
        "public_queue_time_avg_s": (
            queue_sum / queue_count
            if queue_sum is not None and queue_count is not None and queue_count > 0
            else None
        ),
        "public_queue_time_p50_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_queue_time_seconds", 50
        ),
        "public_queue_time_p95_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_queue_time_seconds", 95
        ),
        "public_ttft_p50_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_ttft_seconds", 50
        ),
        "public_ttft_p95_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_ttft_seconds", 95
        ),
        "public_itl_p50_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_itl_seconds", 50
        ),
        "public_itl_p95_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_itl_seconds", 95
        ),
        "public_latency_p50_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_request_latency_seconds", 50
        ),
        "public_latency_p95_bucket_upper_s": _histogram_delta_quantile(
            before, after, "llm_infer_request_latency_seconds", 95
        ),
    }


def _empty_public_metrics() -> dict[str, object]:
    return {
        "public_requests_admitted": None,
        "public_stream_tokens": None,
        "public_queue_time_count": None,
        "public_queue_time_sum_s": None,
        "public_queue_time_avg_s": None,
        "public_queue_time_p50_bucket_upper_s": None,
        "public_queue_time_p95_bucket_upper_s": None,
        "public_ttft_p50_bucket_upper_s": None,
        "public_ttft_p95_bucket_upper_s": None,
        "public_itl_p50_bucket_upper_s": None,
        "public_itl_p95_bucket_upper_s": None,
        "public_latency_p50_bucket_upper_s": None,
        "public_latency_p95_bucket_upper_s": None,
    }


def _sample_delta(before: dict[str, float], after: dict[str, float], name: str) -> float | None:
    value = after.get(name)
    if value is None:
        return None
    return value - before.get(name, 0.0)


def _histogram_delta_quantile(
    before: dict[str, float],
    after: dict[str, float],
    metric: str,
    percentile: float,
) -> float | None:
    count_name = f"{metric}_count"
    total = after.get(count_name, 0.0) - before.get(count_name, 0.0)
    if total <= 0:
        return None
    target = total * percentile / 100.0
    prefix = f'{metric}_bucket{{le="'
    buckets: list[tuple[float, float]] = []
    for name, value in after.items():
        if not name.startswith(prefix):
            continue
        upper_text = name[len(prefix) :].split('"', 1)[0]
        if upper_text == "+Inf":
            continue
        buckets.append((float(upper_text), value - before.get(name, 0.0)))
    for upper, cumulative in sorted(buckets):
        if cumulative >= target:
            return upper
    return None


def _parse_metrics(text: str) -> dict[str, float]:
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        try:
            samples[name] = float(value)
        except ValueError:
            continue
    return samples


def _effective_sampling(engine: InferenceEngine, request: Request) -> SamplingParams:
    return request.sampling if request.sampling is not GREEDY else engine.default_sampling


def _token_to_int(token: object) -> int:
    if isinstance(token, torch.Tensor):
        return int(token.item())
    return int(token)


def _selected(name: str, selected: set[str] | None) -> bool:
    return selected is None or name in selected or "all" in selected


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise argparse.ArgumentTypeError("dtype must be one of: float32, bfloat16, float16")


def _load_runtime(args: argparse.Namespace) -> ModelRuntime:
    bundle_path = (
        args.bundle or os.environ.get("ESME_BUNDLE_PATH") or os.environ.get("LLM_INFER_BUNDLE")
    )
    if args.backend in BUNDLE_BACKENDS and bundle_path is None:
        raise SystemExit("--backend esme requires --bundle or $ESME_BUNDLE_PATH")
    return load_model_runtime(
        args.backend,
        dtype=args.dtype,
        device=args.device,
        bundle_path=Path(bundle_path) if bundle_path is not None else None,
        attention_backend_name=args.attention_backend,
    )


def _write_outputs(
    report: dict[str, object],
    *,
    output: Path,
    jsonl_output: Path | None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if jsonl_output is None:
        return
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    workloads = report["workloads"]
    if isinstance(workloads, list):
        for workload in workloads:
            if not isinstance(workload, dict):
                continue
            for request in workload.get("requests", []):
                lines.append(
                    json.dumps(
                        {
                            "workload": workload.get("name"),
                            "surface": workload.get("surface"),
                            "request": request,
                        },
                        sort_keys=True,
                    )
                )
    jsonl_output.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _default_output_path() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("bench-results") / f"esme-serving-eval-{stamp}.json"


def _format_terminal(report: dict[str, object]) -> str:
    workloads = report.get("workloads", [])
    lines = ["Esme serving eval", "================="]
    if isinstance(workloads, list):
        for item in workloads:
            if not isinstance(item, dict):
                continue
            metrics = item.get("metrics", {})
            reference = item.get("reference", {})
            if not isinstance(metrics, dict) or not isinstance(reference, dict):
                continue
            throughput = metrics.get("throughput_tokens_per_s")
            throughput_text = (
                f"{throughput:.2f} tok/s" if isinstance(throughput, float) else "not reported"
            )
            completed = f"{metrics.get('requests_completed')}/{metrics.get('requests_total')}"
            lines.append(
                f"{item.get('name')} [{item.get('surface')}]: "
                f"ref={reference.get('status')} "
                f"completed={completed} "
                f"tok={metrics.get('output_tokens')} "
                f"throughput={throughput_text} "
                f"preemptions={metrics.get('preemption_count')}"
            )
    output = report.get("output_path")
    if output:
        lines.append(f"json={output}")
    jsonl = report.get("jsonl_output_path")
    if jsonl:
        lines.append(f"jsonl={jsonl}")
    return "\n".join(lines)


async def _main_async(args: argparse.Namespace) -> int:
    started_at = datetime.now(UTC).isoformat()
    selected = set(args.workloads.split(",")) if args.workloads else None
    if args.target == "external-http":
        workloads = await run_external_http_eval(
            base_url=args.base_url,
            model_id=args.model,
            workload_names=selected,
            timeout_s=args.timeout,
            warmup_requests=args.warmup_requests,
        )
        metadata: dict[str, object] = {
            "target": "external-http",
            "base_url": args.base_url,
            "model_id": args.model,
        }
    else:
        runtime = _load_runtime(args)
        # Same default as serve.py: a CUDA bundle model captures the decode-window graphs
        # before any workload runs, so eval rows measure the path serving actually uses.
        # No-op on CPU, where local eval usually runs.
        capture_s = enable_decode_graphs_if_cuda(runtime.model) if args.decode_graphs else None
        workloads = await run_local_eval(runtime, workload_names=selected, device=args.device)
        metadata = {
            "target": "local",
            "backend": args.backend,
            "model_id": runtime.model_id,
            "device": args.device,
            "dtype": str(args.dtype).replace("torch.", ""),
            "attention_backend": type(runtime.model.backend).__name__,
            "attention_backend_choice": args.attention_backend,
            "bundle_path": str(runtime.bundle_path) if runtime.bundle_path is not None else None,
            "decode_graphs": {
                "enabled": capture_s is not None,
                "capture_s": capture_s,
            },
        }

    output = args.output or _default_output_path()
    jsonl_output = args.jsonl_output
    report = {
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "metadata": metadata,
        "workloads": workloads,
        "output_path": str(output),
        "jsonl_output_path": str(jsonl_output) if jsonl_output is not None else None,
    }
    _write_outputs(report, output=output, jsonl_output=jsonl_output)
    print(_format_terminal(report))
    return 1 if any(_workload_failed(workload) for workload in workloads) else 0


def _workload_failed(workload: dict[str, object]) -> bool:
    metrics = workload.get("metrics")
    reference = workload.get("reference")
    if not isinstance(metrics, dict) or not isinstance(reference, dict):
        return True
    return bool(metrics.get("requests_failed")) or reference.get("status") == "fail"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Esme serving behavior under compact mixed workloads."
    )
    parser.add_argument("--target", choices=("local", "external-http"), default="local")
    parser.add_argument("--backend", default="esme", choices=("esme", "dense", "qwen"))
    parser.add_argument("--bundle", type=Path, help="Esme/Dense export bundle path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", type=_dtype, default=torch.float32)
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default="auto",
        help="Attention backend selector. auto uses FlashInfer for CUDA bf16/fp16 Esme bundles.",
    )
    parser.add_argument(
        "--workloads",
        help=(
            "Comma-separated workload names, or all. Defaults to all local workloads. "
            "Known: "
            + ", ".join(w.name for w in (*default_engine_workloads(), *default_http_workloads()))
        ),
    )
    parser.add_argument(
        "--no-decode-graphs",
        dest="decode_graphs",
        action="store_false",
        help="Skip decode-graph capture on CUDA and eval the eager decode window instead.",
    )
    parser.add_argument("--output", type=Path, help="JSON summary path")
    parser.add_argument("--jsonl-output", type=Path, help="Optional per-request JSONL path")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="esme-214m-chat")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="external-server requests run before each measured workload",
    )
    args = parser.parse_args()
    if args.warmup_requests < 0:
        parser.error(f"--warmup-requests must be >= 0; got {args.warmup_requests}")
    try:
        raise SystemExit(asyncio.run(_main_async(args)))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
