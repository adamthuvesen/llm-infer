"""OpenAI-compatible HTTP serving for the inference engine — transport over the step loop.

The engine and KV-cache are unchanged; this package only carries token ids over HTTP.
:class:`AsyncInferenceEngine` runs the synchronous ``step()`` loop on a background thread and
bridges per-request token streams to asyncio; :func:`create_app` wraps it in a FastAPI app
whose handlers speak the OpenAI chat/completions wire format and map each request's sampling
fields to per-request params. The app is built around an injected engine + tokenizer so tests
and production backends drive the same code.
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
