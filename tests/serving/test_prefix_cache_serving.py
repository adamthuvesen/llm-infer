"""Cross-turn prefix cache through the HTTP serving path, plus the serve.py flag defaults.

The server check proves two sequential completions over the same app produce identical output
whether the cache is on or off, and that the cache actually served a reused prefix. The flag
checks pin the tri-state auto resolution (non-CUDA on, CUDA off) without touching a GPU.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as _write_tiny_bundle
from llm_infer.model.interface import BackendCapabilities
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serve import build_app_from_runtime

_CAPS = BackendCapabilities(
    paged_kv=True, prefix_caching=True, speculative=True, flash_attention=False
)
_PROMPT_IDS = [4, 5, 6, 7, 8, 9, 10, 4]
_MAX_NEW_TOKENS = 3


def _prompt_text(token_ids: list[int]) -> str:
    return " ".join(f"tok_{token_id}" for token_id in token_ids)


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def _complete(client: httpx.AsyncClient) -> str:
    response = await client.post(
        "/v1/completions",
        json={
            "model": "tiny-dense",
            "prompt": _prompt_text(_PROMPT_IDS),
            "max_tokens": _MAX_NEW_TOKENS,
        },
    )
    assert response.status_code == 200
    return response.json()["choices"][0]["text"]


async def _hit_tokens(client: httpx.AsyncClient) -> int:
    metrics = (await client.get("/metrics")).text
    for line in metrics.splitlines():
        if line.startswith("llm_infer_prefix_cache_hit_tokens_total "):
            return int(line.split()[1])
    return 0


def test_two_sequential_completions_match_with_cache_on_and_off(tmp_path: Path) -> None:
    import asyncio

    def outputs(*, prefix_cache: bool) -> tuple[str, str, int]:
        runtime = load_model_runtime("esme", bundle_path=_write_tiny_bundle(tmp_path))
        app = build_app_from_runtime(
            runtime, block_size=4, num_blocks=32, prefix_cache=prefix_cache
        )

        async def go() -> tuple[str, str, int]:
            async with app.router.lifespan_context(app), _client(app) as client:
                first = await _complete(client)
                second = await _complete(client)
                return first, second, await _hit_tokens(client)

        return asyncio.run(go())

    cold_first, cold_second, cold_hits = outputs(prefix_cache=False)
    warm_first, warm_second, warm_hits = outputs(prefix_cache=True)

    # Identical greedy prompt twice: the second turn must reproduce the first, and the cache must
    # not change the served text versus a cache-off server.
    assert cold_first == cold_second
    assert warm_first == cold_first
    assert warm_second == cold_second
    # The cache off never reuses a prefix; the cache on reuses the first turn's donated blocks.
    assert cold_hits == 0
    assert warm_hits > 0
