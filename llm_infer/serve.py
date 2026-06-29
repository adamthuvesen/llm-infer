"""Run the OpenAI-compatible server around a registered model backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from llm_infer.model.interface import ModelRuntime
from llm_infer.model.runtime import available_backends, load_model_runtime
from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.server import (
    AsyncInferenceEngine,
    ServerMetrics,
    create_app,
    register_webui,
)

# A 64-token page and a few hundred pages comfortably hold a handful of concurrent
# chat sessions of the lengths this server is demoed at; tune via the CLI for a real load.
DEFAULT_BLOCK_SIZE = 64
DEFAULT_NUM_BLOCKS = 512


def build_app_from_runtime(
    runtime: ModelRuntime,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    device: str = "cpu",
):
    """Wire a loaded model runtime into the HTTP app."""
    engine = InferenceEngine(
        runtime.model,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        capabilities=runtime.capabilities,
    )
    metrics = ServerMetrics()
    async_engine = AsyncInferenceEngine(engine, metrics=metrics)
    app = create_app(
        async_engine=async_engine,
        tokenizer=runtime.tokenizer,
        model_id=runtime.model_id,
        eos_token_ids=runtime.eos_token_ids,
        metrics=metrics,
    )
    register_webui(app)
    return app


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise argparse.ArgumentTypeError("dtype must be one of: float32, bfloat16, float16")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a registered llm-infer backend.")
    parser.add_argument("--backend", choices=available_backends(), default="qwen")
    env_bundle = os.environ.get("LLM_INFER_BUNDLE")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=Path(env_bundle) if env_bundle else None,
        help="Export bundle path for bundle-backed models (defaults to $LLM_INFER_BUNDLE)",
    )
    parser.add_argument("--model", dest="model_id", help="Optional backend-specific model id")
    parser.add_argument("--revision", help="Optional backend-specific model revision")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", type=_dtype, default=torch.float32)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument(
        "--open", action="store_true", help="Open the chat UI in a browser once the server is up"
    )
    args = parser.parse_args()

    import uvicorn

    runtime = load_model_runtime(
        args.backend,
        dtype=args.dtype,
        device=args.device,
        bundle_path=args.bundle,
        model_id=args.model_id,
        revision=args.revision,
    )
    app = build_app_from_runtime(
        runtime,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
        device=args.device,
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"llm-infer chat UI: {url}")
    if args.open:
        import threading
        import webbrowser

        # uvicorn.run blocks, so open the browser from a timer once the server is listening.
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
