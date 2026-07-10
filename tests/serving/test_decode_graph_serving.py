"""Serving-level parity for the decode-graph runner: identical output streams.

`tests/model/test_decode_graph.py` pins the runner at the engine-unit level; these tests
prove the *serving* wiring — ``build_app_from_runtime`` → ``AsyncInferenceEngine`` → the
OpenAI streaming endpoint — emits token-for-token identical streams with the runner on and
off. ``mode="eager"`` runs the exact padded static-buffer flow the CUDA capture replays,
so this is CPU-runnable; on the GPU the only remaining delta is the capture itself, which
the Modal reference gate covers.

Also pinned: ``build_app_from_runtime``'s decode-graph default is a no-op off CUDA, so CPU
serving never pays the padded-bucket overhead.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serve import build_app_from_runtime

# Three prompts stream concurrently so decode batches of 3 pad up to the bucket of 4.
_PROMPTS = ("tok_1 tok_4 tok_7", "tok_2 tok_5 tok_3 tok_6 tok_9", "tok_8 tok_1")
_MAX_NEW_TOKENS = 10


def _stream_completions(runtime, **app_kwargs) -> tuple[list[list[str]], str]:
    """Serve every prompt concurrently over SSE; return delta sequences and /metrics text."""
    app = build_app_from_runtime(runtime, block_size=4, num_blocks=64, **app_kwargs)

    async def one(client: httpx.AsyncClient, prompt: str) -> list[str]:
        deltas: list[str] = []
        async with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "tiny-dense",
                "prompt": prompt,
                "max_tokens": _MAX_NEW_TOKENS,
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                text = json.loads(data)["choices"][0].get("text")
                if text:
                    deltas.append(text)
        return deltas

    async def go() -> tuple[list[list[str]], str]:
        transport = httpx.ASGITransport(app=app)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        ):
            deltas = list(await asyncio.gather(*(one(client, p) for p in _PROMPTS)))
            metrics = (await client.get("/metrics")).text
            return deltas, metrics

    return asyncio.run(go())


def test_streams_identical_with_and_without_decode_graph_runner(tmp_path: Path) -> None:
    """The graphs path and the eager path stream identical deltas for every request."""
    runtime = load_model_runtime("esme", bundle_path=write_tiny_pretrain_bundle(tmp_path))
    assert runtime.model.decode_graphs is None  # CPU: the serve default must not enable it

    plain, _ = _stream_completions(runtime)
    runtime.model.enable_decode_graphs(capture_sizes=(4,), mode="eager")
    padded, _ = _stream_completions(runtime)

    assert all(deltas for deltas in plain)  # every request actually streamed tokens
    assert padded == plain


def test_streams_identical_with_grouped_dispatch_and_metrics_count_its_steps(
    tmp_path: Path,
) -> None:
    """The serve-level grouped wiring: identical streams, and /metrics proves real hits.

    Buckets 1-4 cover every batch shape three concurrent requests can produce, so each
    decode window runs through a grouped runner — the exported step counter must be
    positive, pinning that the flag reaches the engine and the runners actually fire.
    """
    runtime = load_model_runtime("esme", bundle_path=write_tiny_pretrain_bundle(tmp_path))
    runtime.model.decode_graphs = None

    plain, plain_metrics = _stream_completions(runtime)
    assert "llm_infer_grouped_decode_steps_total 0" in plain_metrics

    grouped, grouped_metrics = _stream_completions(
        runtime,
        grouped_decode_graphs=True,
        decode_graph_buckets=(1, 2, 3, 4),
    )
    assert grouped == plain
    match = re.search(r"llm_infer_grouped_decode_steps_total (\d+)", grouped_metrics)
    assert match is not None and int(match.group(1)) > 0


def test_build_app_decode_graphs_default_is_noop_on_cpu(tmp_path: Path) -> None:
    """`decode_graphs=True` (the default) must not capture — or pad — on a CPU model."""
    runtime = load_model_runtime("esme", bundle_path=write_tiny_pretrain_bundle(tmp_path))
    build_app_from_runtime(runtime, block_size=4, num_blocks=64, decode_graphs=True)
    assert runtime.model.decode_graphs is None
