"""Tests for the hand-rolled ``/metrics`` endpoint on the tiny CPU model.

We drive real traffic through the app, scrape ``/metrics``, and assert it is valid Prometheus
text whose numbers reflect what actually happened: generated tokens are non-zero, the block
total matches the engine's pool, and the request/completion counters track the requests served.
Everything runs greedy on the tiny random-weight Qwen — fast, deterministic, no GPU.
"""

from __future__ import annotations

import asyncio

import httpx

from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.server import AsyncInferenceEngine, ServerMetrics, create_app
from llm_infer.serving.server.metrics import Histogram, Registry
from tests.correctness.test_chunked_prefill import _tiny_qwen
from tests.serving.test_api import EOS_ID, TinyTokenizer

NUM_BLOCKS = 64


def _build_app():
    model = _tiny_qwen()
    engine = InferenceEngine(model, block_size=8, num_blocks=NUM_BLOCKS)
    metrics = ServerMetrics()
    async_engine = AsyncInferenceEngine(engine, metrics=metrics)
    app = create_app(
        async_engine=async_engine,
        tokenizer=TinyTokenizer(),
        model_id="tiny-qwen",
        eos_token_ids=frozenset({EOS_ID}),
        metrics=metrics,
    )
    return app


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _parse_metrics(text: str) -> dict[str, float]:
    """Parse the exposition into ``{series: value}``, keeping label sets in the key."""
    samples: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.rpartition(" ")
        samples[name] = float(value)
    return samples


def test_metrics_reflect_real_traffic() -> None:
    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            for _ in range(3):
                resp = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "tiny-qwen",
                        "messages": [{"role": "user", "content": "hello there"}],
                        "max_tokens": 5,
                    },
                )
                assert resp.status_code == 200

            metrics = await client.get("/metrics")
            assert metrics.status_code == 200
            assert metrics.headers["content-type"].startswith("text/plain")
            text = metrics.text

            # Valid exposition: every metric carries a HELP and TYPE line.
            assert "# HELP llm_infer_requests_total" in text
            assert "# TYPE llm_infer_requests_total counter" in text
            assert "# TYPE llm_infer_kv_utilization_ratio gauge" in text
            assert "# TYPE llm_infer_queue_time_seconds histogram" in text
            assert "# TYPE llm_infer_ttft_seconds histogram" in text
            assert "# TYPE llm_infer_preemptions_total counter" in text

            samples = _parse_metrics(text)
            # Three requests served, each capped at 5 tokens under greedy length-stop.
            assert samples["llm_infer_requests_total"] == 3
            assert samples["llm_infer_requests_admitted_total"] == 3
            assert samples['llm_infer_requests_completed_total{finish_reason="length"}'] == 3
            assert samples["llm_infer_generated_tokens_total"] == 15
            assert samples["llm_infer_stream_tokens_total"] == 15
            # Block total is the real pool size; used/free is a sane partition of it.
            assert samples["llm_infer_kv_blocks_total"] == NUM_BLOCKS
            assert 0 <= samples["llm_infer_kv_blocks_used"] <= NUM_BLOCKS
            assert (
                samples["llm_infer_kv_blocks_used"] + samples["llm_infer_kv_blocks_free"]
                == NUM_BLOCKS
            )
            assert 0.0 <= samples["llm_infer_kv_utilization_ratio"] <= 1.0
            # No KV pressure on the tiny model with a big pool: nothing was preempted.
            assert samples["llm_infer_preemptions_total"] == 0
            # The TTFT/latency histograms observed all three requests.
            assert samples["llm_infer_queue_time_seconds_count"] == 3
            assert samples["llm_infer_ttft_seconds_count"] == 3
            assert samples["llm_infer_request_latency_seconds_count"] == 3

    asyncio.run(go())


def test_stop_truncated_request_is_counted_completed() -> None:
    """A stop-truncated request is counted as completed (finish_reason=stop), not dropped.

    The async engine aborts the stream the moment the handler detects a stop, so it never sees
    that request reach a ``finish_reason`` — its completion is recorded by the handler instead.
    Without that, a stop-truncated request would be served but vanish from the completed counter
    and the latency histogram, under-reporting successful traffic.
    """

    async def go() -> None:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            body = {
                "model": "tiny-qwen",
                "messages": [{"role": "user", "content": "hello there"}],
                "max_tokens": 6,
            }
            first = await client.post("/v1/chat/completions", json=body)
            assert first.status_code == 200
            text = first.json()["choices"][0]["message"]["content"]
            assert len(text) >= 3  # need a substring to stop on, with output preceding it

            # A substring of the deterministic greedy output, so the stop is guaranteed to fire.
            second = await client.post("/v1/chat/completions", json={**body, "stop": text[1:3]})
            assert second.status_code == 200
            assert second.json()["choices"][0]["finish_reason"] == "stop"

            samples = _parse_metrics((await client.get("/metrics")).text)
            assert samples["llm_infer_requests_total"] == 2
            assert samples['llm_infer_requests_completed_total{finish_reason="length"}'] == 1
            assert samples['llm_infer_requests_completed_total{finish_reason="stop"}'] == 1
            # Both completions observed end-to-end latency — the stop one via the handler.
            assert samples["llm_infer_request_latency_seconds_count"] == 2

    asyncio.run(go())


def test_histogram_buckets_are_cumulative() -> None:
    """A histogram's bucket counts are cumulative and the +Inf bucket equals the total count."""
    hist = Histogram("h_seconds", "help", buckets=(0.1, 1.0, 10.0))
    for value in (0.05, 0.2, 0.2, 5.0):
        hist.observe(value)
    registry = Registry()
    registry.register(hist)
    samples = _parse_metrics(registry.render())
    assert samples['h_seconds_bucket{le="0.1"}'] == 1  # only 0.05
    assert samples['h_seconds_bucket{le="1"}'] == 3  # 0.05, 0.2, 0.2
    assert samples['h_seconds_bucket{le="10"}'] == 4  # all four
    assert samples['h_seconds_bucket{le="+Inf"}'] == 4
    assert samples["h_seconds_count"] == 4
