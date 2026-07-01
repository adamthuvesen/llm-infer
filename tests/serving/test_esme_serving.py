"""Esme end-to-end through the real serving path.

Qwen's HTTP surface is covered by ``test_api.py`` on a tiny random-weight Qwen; these tests
prove the **Esme** bundle backend is a true peer on the *serving* path, not just at the engine
unit level. Everything runs on the tiny ``llm_pretrain_dense_v1`` bundle (``_write_tiny_bundle``)
so it is fast, deterministic, and CPU-only — the real Esme-214M-Chat weights are exercised
separately by the ``ESME_BUNDLE_PATH``-gated parity tests.

Two things are checked here that the landed engine-unit tests do not:

* Esme serves end-to-end through ``serve.py``'s ``build_app_from_runtime`` →
  ``AsyncInferenceEngine`` → OpenAI app, and the served token ids match the full-recompute
  ``greedy_decode`` reference (the "match the reference before measuring" gate, serving path).
* Prefix caching **and** recompute preemption run through the **async serving loop** under
  concurrent requests — not a single ``engine.run()`` — each verified token-for-token against the
  per-sequence recompute reference, with a real preemption proven to have fired.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from llm_infer.model.decode import greedy_decode
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serve import build_app_from_runtime
from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.request import Request
from llm_infer.serving.server import AsyncInferenceEngine
from llm_infer.tracing import TraceRecorder
from tests.correctness.test_pretrain_bundle import _write_tiny_bundle

# Prompts are bare token ids from the tiny WordLevel vocab (tok_0..tok_10): the bundle tokenizer
# decodes "tok_N" -> N, so a space-joined string of them round-trips to exactly these ids.
_PROMPT_IDS = [1, 4, 7]
_MAX_NEW_TOKENS = 5


def _esme_runtime(tmp_path: Path):
    return load_model_runtime("esme", bundle_path=_write_tiny_bundle(tmp_path))


def _prompt_text(token_ids: list[int]) -> str:
    return " ".join(f"tok_{token_id}" for token_id in token_ids)


def _reference_tokens(runtime, prompt_ids: list[int], max_new_tokens: int) -> list[int]:
    """The full-recompute oracle: direct ``PretrainBundleModel.logits()`` greedy decode."""
    return greedy_decode(
        runtime.model,
        list(prompt_ids),
        max_new_tokens=max_new_tokens,
        eos_token_ids=set(runtime.eos_token_ids),
    )


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_esme_serves_completions_end_to_end_matching_reference(tmp_path: Path) -> None:
    """Esme serves /v1/completions through the async engine; tokens match recompute greedy."""
    runtime = _esme_runtime(tmp_path)
    # No bundle EOS in the tiny manifest, so the reference runs to the length cap, never stopping.
    reference = _reference_tokens(runtime, _PROMPT_IDS, _MAX_NEW_TOKENS)
    app = build_app_from_runtime(runtime, block_size=8, num_blocks=32)

    async def go() -> None:
        async with app.router.lifespan_context(app), _client(app) as client:
            models = await client.get("/v1/models")
            assert models.json()["data"][0]["id"] == "tiny-dense"

            response = await client.post(
                "/v1/completions",
                json={
                    "model": "tiny-dense",
                    "prompt": _prompt_text(_PROMPT_IDS),
                    "max_tokens": _MAX_NEW_TOKENS,
                },
            )
            assert response.status_code == 200
            body = response.json()
            assert body["model"] == "tiny-dense"
            assert body["usage"]["completion_tokens"] == _MAX_NEW_TOKENS
            # The served text decodes back to exactly the reference token ids.
            served_ids = [int(piece[len("tok_") :]) for piece in body["choices"][0]["text"].split()]
            assert served_ids == reference

    asyncio.run(go())


def test_esme_streams_chat_completions_end_to_end(tmp_path: Path) -> None:
    """Esme streams /v1/chat/completions as SSE deltas ending in [DONE], over the async loop."""
    runtime = _esme_runtime(tmp_path)
    app = build_app_from_runtime(runtime, block_size=8, num_blocks=32)

    async def go() -> None:
        async with app.router.lifespan_context(app), _client(app) as client:
            deltas: list[str] = []
            saw_done = False
            async with client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "tiny-dense",
                    "messages": [{"role": "user", "content": _prompt_text(_PROMPT_IDS)}],
                    "max_tokens": _MAX_NEW_TOKENS,
                    "stream": True,
                },
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: ") :]
                    if payload == "[DONE]":
                        saw_done = True
                        break
                    import json

                    chunk = json.loads(payload)
                    content = chunk["choices"][0].get("delta", {}).get("content")
                    if content:
                        deltas.append(content)
            assert saw_done
            assert deltas, "streaming produced no content deltas"

    asyncio.run(go())


def _drive_concurrent(
    runtime,
    requests: list[Request],
    *,
    block_size: int,
    num_blocks: int,
    preemption: bool,
    trace: TraceRecorder | None,
) -> dict[str, list[int]]:
    """Drive several requests concurrently through the real ``AsyncInferenceEngine`` serving loop.

    This is the serving/scheduler path — the same background step loop the HTTP app uses — not a
    single ``engine.run()``. Each request is streamed independently; we collect its emitted ids.
    """
    engine = InferenceEngine(
        runtime.model,
        block_size=block_size,
        num_blocks=num_blocks,
        capabilities=runtime.capabilities,
        preemption=preemption,
        trace=trace,
    )
    async_engine = AsyncInferenceEngine(engine)

    async def go() -> dict[str, list[int]]:
        async_engine.start()
        try:

            async def consume(request: Request) -> tuple[str, list[int]]:
                ids: list[int] = []
                async for item in async_engine.stream(
                    request_id=request.request_id,
                    prompt_ids=request.prompt_ids,
                    max_new_tokens=request.max_new_tokens,
                    eos_token_ids=request.eos_token_ids,
                    sampling=request.sampling,
                ):
                    ids.append(item.token_id)
                return request.request_id, ids

            results = await asyncio.gather(*(consume(request) for request in requests))
            return dict(results)
        finally:
            async_engine.stop()

    return asyncio.run(go())


def test_esme_prefix_caching_through_serving_path_matches_reference(tmp_path: Path) -> None:
    """Concurrent prefix-group siblings served through the async loop match recompute, each.

    The two requests share a prompt via ``prefix_group_id``, so the second reuses the leader's
    cached prefix blocks. Driven through the real serving loop (not ``engine.run()``); both must
    reproduce the per-sequence full-recompute greedy reference token-for-token.
    """
    runtime = _esme_runtime(tmp_path)
    reference = _reference_tokens(runtime, _PROMPT_IDS, _MAX_NEW_TOKENS)
    requests = [
        Request(
            request_id,
            list(_PROMPT_IDS),
            max_new_tokens=_MAX_NEW_TOKENS,
            eos_token_ids=runtime.eos_token_ids,
            prefix_group_id="shared",
        )
        for request_id in ("sib-a", "sib-b")
    ]

    outputs = _drive_concurrent(
        runtime, requests, block_size=8, num_blocks=32, preemption=False, trace=None
    )

    assert outputs["sib-a"] == reference
    assert outputs["sib-b"] == reference


def test_esme_preemption_through_serving_path_matches_reference(tmp_path: Path) -> None:
    """Concurrent Esme requests under a tight KV pool preempt + recompute-resume, staying exact.

    A real preemption is forced (over-committed pool, LIFO eviction) and proven to fire via the
    trace; every served output must equal that request's uninterrupted recompute reference, so the
    recompute-on-resume path is correct on the Esme bundle through the serving loop.
    """
    runtime = _esme_runtime(tmp_path)
    prompts = {
        "a": [1, 4, 7],
        "b": [2, 5, 8],
        "c": [3, 6, 9],
    }
    max_new = 6
    references = {
        request_id: _reference_tokens(runtime, prompt, max_new)
        for request_id, prompt in prompts.items()
    }

    def build_requests() -> list[Request]:
        return [
            Request(request_id, list(prompt), max_new, runtime.eos_token_ids)
            for request_id, prompt in prompts.items()
        ]

    # Roomy reservation pool: worst-case fits all three, nobody is preempted — the control.
    roomy = _drive_concurrent(
        runtime, build_requests(), block_size=4, num_blocks=12, preemption=False, trace=None
    )
    # Tight pool with preemption: each prompt is 3 tokens (footprint 1 block at block_size 4) so
    # three fit on admission, but as they decode the pool (3 blocks) must evict and recompute.
    recorder = TraceRecorder()
    tight = _drive_concurrent(
        runtime, build_requests(), block_size=4, num_blocks=3, preemption=True, trace=recorder
    )

    preemptions = [event for event in recorder.events if event.event == "request_preempted"]
    resumes = [event for event in recorder.events if event.event == "request_resumed"]
    assert preemptions, "tight pool must force at least one real preemption"
    assert resumes, "a preempted request must resume"
    for event in preemptions:
        assert event.preempt_reason == "kv_pressure"

    for request_id, expected in references.items():
        assert roomy[request_id] == expected, f"{request_id}: roomy serving diverged from reference"
        assert tight[request_id] == expected, (
            f"{request_id}: preempted serving output diverged from the recompute reference "
            f"(served={tight[request_id]}, reference={expected})"
        )


def test_esme_serving_rejects_unknown_model_id(tmp_path: Path) -> None:
    """A single-model Esme server 404s any other model id — the contract loadgen relies on."""
    runtime = _esme_runtime(tmp_path)
    app = build_app_from_runtime(runtime, block_size=8, num_blocks=32)

    async def go() -> None:
        async with app.router.lifespan_context(app), _client(app) as client:
            response = await client.post(
                "/v1/completions",
                json={"model": "not-the-served-model", "prompt": _prompt_text(_PROMPT_IDS)},
            )
            assert response.status_code == 404

    asyncio.run(go())
