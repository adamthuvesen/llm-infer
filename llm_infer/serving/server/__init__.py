"""OpenAI-compatible HTTP serving for the inference engine — transport over the step loop.

The engine, sampler, and KV-cache are unchanged; this package only carries token ids over
HTTP. :class:`AsyncInferenceEngine` runs the synchronous ``step()`` loop on a background
thread and bridges per-request token streams to asyncio; :func:`create_app` wraps it in a
FastAPI app whose handlers speak the OpenAI chat/completions wire format. The app is built
around an injected engine + tokenizer so tests drive the tiny CPU model and production drives
the pinned Qwen through the same code.
"""

from __future__ import annotations

from llm_infer.serving.server.app import create_app
from llm_infer.serving.server.async_engine import AsyncInferenceEngine, TokenStreamItem
from llm_infer.serving.server.detokenizer import IncrementalDetokenizer
from llm_infer.serving.server.metrics import ServerMetrics

__all__ = [
    "AsyncInferenceEngine",
    "IncrementalDetokenizer",
    "ServerMetrics",
    "TokenStreamItem",
    "create_app",
]
