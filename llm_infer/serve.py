"""Run the OpenAI-compatible server around a registered model backend."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

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
from llm_infer.serving.speculative import SpeculativeDecodingConfig

# A 64-token page and a few hundred pages comfortably hold a handful of concurrent
# chat sessions of the lengths this server is demoed at; tune via the CLI for a real load.
DEFAULT_BLOCK_SIZE = 64
DEFAULT_NUM_BLOCKS = 512
DEFAULT_BACKEND = "esme"
BUNDLE_BACKENDS = frozenset({"dense", "esme"})


def build_app_from_runtime(
    runtime: ModelRuntime,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    device: str = "cpu",
    preemption_policy: Literal["off", "recompute"] = "off",
    prefill_chunk_size: int | None = None,
    prompt_lookup_speculative: SpeculativeDecodingConfig | None = None,
):
    """Wire a loaded model runtime into the HTTP app."""
    engine = InferenceEngine(
        runtime.model,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        capabilities=runtime.capabilities,
        preemption=preemption_policy == "recompute",
        prefill_chunk_size=prefill_chunk_size,
        speculative=prompt_lookup_speculative,
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


def _positive_int(name: str) -> Callable[[str], int]:
    def parse(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError(f"{name} must be >= 1")
        return parsed

    return parse


def _bundle_path_for_backend(
    backend: str,
    *,
    explicit_bundle: Path | None,
    env_bundle: str | None,
) -> Path | None:
    if backend not in BUNDLE_BACKENDS:
        if explicit_bundle is not None:
            raise argparse.ArgumentTypeError(f"backend {backend!r} does not accept --bundle")
        return None
    bundle = explicit_bundle or (Path(env_bundle) if env_bundle else None)
    if bundle is None:
        raise argparse.ArgumentTypeError(
            f"--backend {backend} requires --bundle or $ESME_BUNDLE_PATH"
        )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a registered llm-infer backend.")
    parser.add_argument(
        "--backend",
        choices=available_backends(),
        default=DEFAULT_BACKEND,
        help="Backend to serve (defaults to Esme; use qwen only for historical reproduction)",
    )
    # ESME_BUNDLE_PATH is the name the Esme bundle uses everywhere else (tests, Modal
    # harnesses); LLM_INFER_BUNDLE stays as the generic fallback for any bundle backend.
    env_bundle = os.environ.get("ESME_BUNDLE_PATH") or os.environ.get("LLM_INFER_BUNDLE")
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Export bundle path for bundle-backed models; defaults to $ESME_BUNDLE_PATH, "
        "then $LLM_INFER_BUNDLE",
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
        "--preemption-policy",
        choices=("off", "recompute"),
        default="off",
        help="KV pressure policy: off reserves worst-case blocks; recompute evicts and rebuilds.",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=_positive_int("prefill_chunk_size"),
        help="Maximum prompt tokens cached per prefill step; unset caches full prompts.",
    )
    parser.add_argument(
        "--prompt-lookup-speculative",
        action="store_true",
        help="Enable prompt-lookup speculative decode when the backend supports it.",
    )
    parser.add_argument(
        "--prompt-lookup-max-draft-tokens",
        type=_positive_int("prompt_lookup_max_draft_tokens"),
        default=4,
    )
    parser.add_argument(
        "--prompt-lookup-max-ngram-size",
        type=_positive_int("prompt_lookup_max_ngram_size"),
        default=4,
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the chat UI in a browser once the server is up"
    )
    args = parser.parse_args()
    try:
        bundle_path = _bundle_path_for_backend(
            args.backend,
            explicit_bundle=args.bundle,
            env_bundle=env_bundle,
        )
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    import uvicorn

    runtime = load_model_runtime(
        args.backend,
        dtype=args.dtype,
        device=args.device,
        bundle_path=bundle_path,
        model_id=args.model_id,
        revision=args.revision,
    )
    speculative = (
        SpeculativeDecodingConfig(
            max_draft_tokens=args.prompt_lookup_max_draft_tokens,
            max_ngram_size=args.prompt_lookup_max_ngram_size,
        )
        if args.prompt_lookup_speculative
        else None
    )
    try:
        app = build_app_from_runtime(
            runtime,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            device=args.device,
            preemption_policy=args.preemption_policy,
            prefill_chunk_size=args.prefill_chunk_size,
            prompt_lookup_speculative=speculative,
        )
    except ValueError as exc:
        parser.error(str(exc))
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
