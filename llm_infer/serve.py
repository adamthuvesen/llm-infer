"""Run the OpenAI-compatible server around a registered model backend."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

import torch

from llm_infer.model.runtime import (
    ATTENTION_BACKEND_CHOICES,
    available_backends,
    load_model_runtime,
)
from llm_infer.serving.server.runtime_app import (
    DEFAULT_BACKEND,
    DEFAULT_BLOCK_SIZE,
    DEFAULT_DECODE_GRAPH_BUCKETS,
    DEFAULT_DECODE_WINDOW_SIZE,
    DEFAULT_NUM_BLOCKS,
    build_app_from_runtime,
    bundle_path_for_backend,
    describe_run_config,
    resolve_engine_defaults,
    resolve_runtime_defaults,
)
from llm_infer.serving.speculative import SpeculativeDecodingConfig


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
        help="Attention backend selector. auto uses FlashInfer for CUDA bf16/fp16 Esme bundles "
        "and torch_sdpa for local (non-CUDA) bundle serving.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. auto resolves to cpu, because mps measured slower on the 214M "
        "bundle. Pass cpu, mps, or cuda to choose explicitly.",
    )
    parser.add_argument(
        "--dtype",
        type=_dtype,
        default=None,
        help="Model dtype. Default is auto: fp16 on cpu and mps, fp32 on cuda. An explicit "
        "value wins.",
    )
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
        "--grouped-decode-graphs",
        dest="grouped_decode_graphs",
        action="store_true",
        default=None,
        help="Force engine-owned grouped-layer decode graphs on. The default enables them "
        "automatically on a CUDA bundle model with a paged attention backend; exact-batch "
        "windows decode through them, everything else uses the piecewise path.",
    )
    parser.add_argument(
        "--no-grouped-decode-graphs",
        dest="grouped_decode_graphs",
        action="store_false",
        help="Serve without grouped decode graphs (skips their startup capture).",
    )
    parser.add_argument(
        "--fused-elementwise",
        dest="fused_elementwise",
        action="store_true",
        default=None,
        help="Force fused RMSNorm and RoPE kernels on (CUDA bundle models only). The default "
        "enables them for CUDA bf16/fp16 bundle models; they add about 40 s of compile time "
        "at startup.",
    )
    parser.add_argument(
        "--no-fused-elementwise",
        dest="fused_elementwise",
        action="store_false",
        help="Serve with the unfused elementwise kernels (skips their startup compile).",
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
        dest="prompt_lookup_speculative",
        action="store_true",
        default=None,
        help="Turn prompt-lookup speculative decode on. Default is auto: on for non-CUDA "
        "devices (a CPU/Mac serving speedup) and off on CUDA, where it would disable the "
        "deferred decode window and grouped graphs.",
    )
    parser.add_argument(
        "--no-prompt-lookup-speculative",
        dest="prompt_lookup_speculative",
        action="store_false",
        help="Turn prompt-lookup speculative decode off (overrides the auto default).",
    )
    parser.add_argument(
        "--prefix-cache",
        dest="prefix_cache",
        action="store_true",
        default=None,
        help="Turn the cross-turn prefix cache on so a follow-up chat turn reuses the previous "
        "turn's prompt KV instead of re-prefilling it. Default is auto: on for non-CUDA bundle "
        "backends that support prefix caching, off on CUDA.",
    )
    parser.add_argument(
        "--no-prefix-cache",
        dest="prefix_cache",
        action="store_false",
        help="Turn the cross-turn prefix cache off (overrides the auto default).",
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
        bundle_path = bundle_path_for_backend(
            args.backend,
            explicit_bundle=args.bundle,
            env_bundle=env_bundle,
        )
    except ValueError as exc:
        parser.error(str(exc))

    import uvicorn

    defaults = resolve_runtime_defaults(
        device=args.device,
        dtype=args.dtype,
        attention_backend=args.attention_backend,
        backend=args.backend,
    )
    device, dtype = defaults.device, defaults.dtype
    print(describe_run_config(device, dtype))
    runtime = load_model_runtime(
        args.backend,
        dtype=dtype,
        device=device,
        bundle_path=bundle_path,
        model_id=args.model_id,
        revision=args.revision,
        attention_backend_name=defaults.attention_backend,
    )
    engine_defaults = resolve_engine_defaults(
        prompt_lookup=args.prompt_lookup_speculative,
        prefix_cache=args.prefix_cache,
        device=device,
        backend=args.backend,
        capabilities=runtime.capabilities,
    )
    speculative = (
        SpeculativeDecodingConfig(
            max_draft_tokens=args.prompt_lookup_max_draft_tokens,
            max_ngram_size=args.prompt_lookup_max_ngram_size,
        )
        if engine_defaults.prompt_lookup
        else None
    )
    prefix_cache = engine_defaults.prefix_cache
    try:
        app = build_app_from_runtime(
            runtime,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
            device=device,
            preemption_policy=args.preemption_policy,
            prefill_chunk_size=args.prefill_chunk_size,
            prompt_lookup_speculative=speculative,
            prefix_cache=prefix_cache,
            decode_window_size=args.decode_window_size,
            decode_graphs=args.decode_graphs,
            decode_graph_buckets=args.decode_graph_buckets,
            grouped_decode_graphs=args.grouped_decode_graphs,
            fused_elementwise=args.fused_elementwise,
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
