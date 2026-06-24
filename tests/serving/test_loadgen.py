"""The load generator, run against the in-process tiny-model server over ASGI.

``run_load`` takes an injected ``httpx.AsyncClient``, so the test points it at the in-process
app through ``httpx.ASGITransport`` — no real socket, fully deterministic, no GPU. We assert it
drives every request to completion and reports non-zero throughput, for both streaming and
blocking modes, plus that the summary table renders.
"""

from __future__ import annotations

import asyncio

import httpx

from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.server import AsyncInferenceEngine, create_app
from scripts.loadgen import format_summary, run_load, summarize
from tests.correctness.test_chunked_prefill import _tiny_qwen
from tests.serving.test_api import EOS_ID, TinyTokenizer


def _build_app():
    model = _tiny_qwen()
    engine = InferenceEngine(model, block_size=8, num_blocks=64)
    async_engine = AsyncInferenceEngine(engine)
    return create_app(
        async_engine=async_engine,
        tokenizer=TinyTokenizer(),
        model_id="tiny-qwen",
        eos_token_ids=frozenset({EOS_ID}),
    )


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _drive(stream: bool) -> dict[str, object]:
    async def go() -> dict[str, object]:
        app = _build_app()
        async with app.router.lifespan_context(app), _client(app) as client:
            results, wall_s = await run_load(
                client,
                model="tiny-qwen",
                prompt="hello there",
                max_tokens=6,
                concurrency=2,
                num_requests=6,
                stream=stream,
            )
        return summarize(results, wall_s)

    return asyncio.run(go())


def test_loadgen_streaming_reports_nonzero_throughput() -> None:
    summary = _drive(stream=True)
    assert summary["requests"] == 6
    assert summary["ok"] == 6
    assert summary["errors"] == 0
    # Six requests × six tokens each (greedy length cap, tiny model never emits EOS here).
    assert summary["total_output_tokens"] == 36
    assert summary["throughput_tok_s"] > 0
    assert summary["per_request_tok_s_mean"] > 0
    assert summary["latency_p50"] > 0
    # The summary table renders without error.
    table = format_summary(summary, concurrency=2, stream=True)
    assert "throughput" in table


def test_loadgen_blocking_reports_nonzero_throughput() -> None:
    summary = _drive(stream=False)
    assert summary["ok"] == 6
    assert summary["errors"] == 0
    assert summary["total_output_tokens"] == 36
    assert summary["throughput_tok_s"] > 0
