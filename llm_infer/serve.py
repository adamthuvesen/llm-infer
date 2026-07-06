"""Run the OpenAI-compatible server around a registered model backend."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import torch

from llm_infer.kernels.base import PagedDecodeAttentionBackend
from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
from llm_infer.model.interface import ModelRuntime
from llm_infer.model.runtime import (
    ATTENTION_BACKEND_CHOICES,
    available_backends,
    load_model_runtime,
)
from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.request import Request
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
DEFAULT_DECODE_WINDOW_SIZE = 8
DEFAULT_BACKEND = "esme"
BUNDLE_BACKENDS = frozenset({"dense", "esme"})
# Decode-graph buckets the server captures at startup. Capture costs roughly 5 s per bucket
# on an A100, so the serve default covers a handful of concurrent chat sessions (~25 s)
# rather than the benchmark harnesses' full spread; a batch above the largest bucket falls
# back to the eager planned window, which is correct, just slower. Tune via
# --decode-graph-buckets.
DEFAULT_DECODE_GRAPH_BUCKETS = (1, 2, 4, 8, 16)


def build_app_from_runtime(
    runtime: ModelRuntime,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    num_blocks: int = DEFAULT_NUM_BLOCKS,
    device: str = "cpu",
    preemption_policy: Literal["off", "recompute"] = "off",
    prefill_chunk_size: int | None = None,
    prompt_lookup_speculative: SpeculativeDecodingConfig | None = None,
    decode_window_size: int = DEFAULT_DECODE_WINDOW_SIZE,
    decode_graphs: bool = True,
    decode_graph_buckets: tuple[int, ...] = DEFAULT_DECODE_GRAPH_BUCKETS,
):
    """Wire a loaded model runtime into the HTTP app.

    Decode graphs are on by default: on a CUDA bundle model the piecewise decode-window
    graphs are captured here, before the engine starts serving, so the capture cost lands
    at startup, never inside a request. On CPU or non-bundle backends this is a no-op.
    """
    warmup_s = _warm_flashinfer_decode_if_needed(
        runtime,
        block_size=block_size,
        device=device,
    )
    if warmup_s is not None:
        print(f"attention backend: {type(runtime.model.backend).__name__}")
        print(f"flashinfer warmup: one-token decode in {warmup_s:.1f} s")
    else:
        print(f"attention backend: {type(runtime.model.backend).__name__}")
    if decode_graphs:
        capture_s = enable_decode_graphs_if_cuda(runtime.model, decode_graph_buckets)
        if capture_s is not None:
            print(f"decode graphs: captured buckets {decode_graph_buckets} in {capture_s:.1f} s")
    engine = InferenceEngine(
        runtime.model,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        capabilities=runtime.capabilities,
        preemption=preemption_policy == "recompute",
        prefill_chunk_size=prefill_chunk_size,
        speculative=prompt_lookup_speculative,
        decode_window_size=decode_window_size,
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


# The warmup runs one short request (1 prompt token + 1 decode token) purely to trigger
# FlashInfer's shape-specific JIT before serving; a couple of blocks always cover it, so it
# never touches the full serving pool.
_WARMUP_NUM_BLOCKS = 4


def _warm_flashinfer_decode_if_needed(
    runtime: ModelRuntime,
    *,
    block_size: int,
    device: str,
) -> float | None:
    if not isinstance(runtime.model.backend, PagedDecodeAttentionBackend):
        return None
    if torch.device(device).type != "cuda":
        return None

    prompt_ids = _warmup_prompt_ids(runtime)
    engine = InferenceEngine(
        runtime.model,
        block_size=block_size,
        num_blocks=_WARMUP_NUM_BLOCKS,
        device=device,
        capabilities=runtime.capabilities,
        decode_window_size=1,
    )
    engine.add_request(Request("_flashinfer_warmup", prompt_ids, 1, frozenset()))
    torch.cuda.synchronize()
    start = time.perf_counter()
    engine.run()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _warmup_prompt_ids(runtime: ModelRuntime) -> list[int]:
    token_ids = runtime.tokenizer.encode("warmup")
    if token_ids:
        return [int(token_ids[0])]
    return [0]


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise argparse.ArgumentTypeError("dtype must be one of: float32, bfloat16, float16")


def _bucket_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part) for part in value.split(",") if part.strip())
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError(
            f"decode-graph-buckets must be positive integers; got {value!r}"
        )
    return sizes


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
        help="Backend to serve (defaults to Esme; use qwen for the Qwen reference backend)",
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
    parser.add_argument(
        "--attention-backend",
        choices=ATTENTION_BACKEND_CHOICES,
        default="auto",
        help="Attention backend selector. auto uses FlashInfer for CUDA bf16/fp16 Esme bundles.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", type=_dtype, default=torch.float32)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--num-blocks", type=int, default=DEFAULT_NUM_BLOCKS)
    parser.add_argument(
        "--decode-window-size",
        type=_positive_int("decode_window_size"),
        default=DEFAULT_DECODE_WINDOW_SIZE,
        help="Decode steps per EOS/stop host sync for all-greedy batches; 1 restores the "
        "classic per-step decode path.",
    )
    parser.add_argument(
        "--no-decode-graphs",
        dest="decode_graphs",
        action="store_false",
        help="Serve on the eager decode window instead of capturing CUDA graphs at startup.",
    )
    parser.add_argument(
        "--decode-graph-buckets",
        type=_bucket_sizes,
        default=DEFAULT_DECODE_GRAPH_BUCKETS,
        help="Comma-separated batch-size buckets to capture (~5 s each on an A100); batches "
        "above the largest bucket decode on the eager window.",
    )
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
        attention_backend_name=args.attention_backend,
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
            decode_window_size=args.decode_window_size,
            decode_graphs=args.decode_graphs,
            decode_graph_buckets=args.decode_graph_buckets,
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
