"""Static chat UI route checks."""

from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from llm_infer.serving.server import register_webui


def test_webui_serves_topic_picker_and_training_grounded_questions() -> None:
    async def get_index() -> httpx.Response:
        app = FastAPI()
        register_webui(app)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/")

    response = asyncio.run(get_index())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert '<link rel="icon" type="image/svg+xml"' in response.text
    assert "fill='%23D6531F'" in response.text
    assert 'id="starter-topic"' in response.text
    assert 'id="quick-question"' in response.text
    assert "const ESME_MARK = `<svg" in response.text
    assert 'aria-label="Close sidebar"' in response.text
    assert 'aria-label="Open sidebar"' in response.text
    assert 'id="max-val">128</span>' in response.text
    assert "maxTokens: 128" in response.text
    assert "Try asking Esme" in response.text
    assert "Why do Earth's seasons occur?" in response.text
    assert "What is an HTTP request?" in response.text
    assert "Likely coverage, not guaranteed recall." in response.text
    assert 'class="caret"' not in response.text
    assert 'id="thread-title"' not in response.text
