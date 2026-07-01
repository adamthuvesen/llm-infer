"""The load generator, run against the in-process tiny-model server over ASGI.

``run_load`` takes an injected ``httpx.AsyncClient``, so the test points it at the in-process
app through ``httpx.ASGITransport`` — no real socket, fully deterministic, no GPU. We assert it
drives every request to completion and reports non-zero throughput, for both streaming and
blocking modes, plus that the summary table renders.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from llm_infer.model.runtime import load_model_runtime
from llm_infer.serve import build_app_from_runtime
from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.server import AsyncInferenceEngine, create_app
from scripts.loadgen import format_summary, run_load, summarize
from tests.correctness.test_chunked_prefill import _tiny_qwen
from tests.correctness.test_pretrain_bundle import _write_tiny_bundle
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


def _build_esme_app(tmp_path: Path):
    """The real Esme bundle backend, served exactly as ``serve.py --backend esme`` would."""
    runtime = load_model_runtime("esme", bundle_path=_write_tiny_bundle(tmp_path))
    return build_app_from_runtime(runtime, block_size=8, num_blocks=64), runtime.model_id


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
    # Six requests × six deltas each. The tiny tokenizer decodes one token per delta, so deltas
    # equal tokens here — but the loadgen still labels the streaming unit as deltas, not tokens.
    assert summary["total_output"] == 36
    assert summary["throughput_per_s"] > 0
    assert summary["per_request_per_s_mean"] > 0
    assert summary["latency_p50"] > 0
    # The streaming table labels the unit as deltas, never tokens.
    table = format_summary(summary, concurrency=2, stream=True)
    assert "throughput" in table
    assert "delta/s" in table
    assert "tok/s" not in table


def test_loadgen_rejects_nonpositive_concurrency() -> None:
    """concurrency < 1 would build a zero-permit semaphore that hangs every task — reject it."""
    with pytest.raises(ValueError, match="concurrency"):
        asyncio.run(
            run_load(
                None,  # rejected before the client is touched
                model="m",
                prompt="p",
                max_tokens=4,
                concurrency=0,
                num_requests=4,
                stream=True,
            )
        )


def test_loadgen_blocking_reports_nonzero_throughput() -> None:
    summary = _drive(stream=False)
    assert summary["ok"] == 6
    assert summary["errors"] == 0
    # Blocking reads true completion tokens from usage; the table labels them tok/s.
    assert summary["total_output"] == 36
    assert summary["throughput_per_s"] > 0
    table = format_summary(summary, concurrency=2, stream=False)
    assert "tok/s" in table


def test_loadgen_drives_esme_server(tmp_path: Path) -> None:
    """loadgen targets an Esme server by its served model id — the real Esme load path.

    Single-model servers 404 on a mismatched id, so this drives load with
    ``model=runtime.model_id`` to prove loadgen targets Esme by the served model id. Same
    in-process ASGI transport as the tiny-Qwen loadgen tests.
    """

    async def go() -> dict[str, object]:
        app, model_id = _build_esme_app(tmp_path)
        async with app.router.lifespan_context(app), _client(app) as client:
            results, wall_s = await run_load(
                client,
                model=model_id,
                prompt="tok_1 tok_4 tok_7",
                max_tokens=6,
                concurrency=2,
                num_requests=6,
                stream=False,
            )
        return summarize(results, wall_s)

    summary = asyncio.run(go())
    assert summary["requests"] == 6
    assert summary["ok"] == 6
    assert summary["errors"] == 0
    assert summary["total_output"] == 36
    assert summary["throughput_per_s"] > 0
