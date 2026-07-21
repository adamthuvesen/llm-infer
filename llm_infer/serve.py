"""Run the OpenAI-compatible server around a registered model backend."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from fastapi import FastAPI

from llm_infer.kernels.base import PagedDecodeAttentionBackend
from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
from llm_infer.model.interface import BackendCapabilities, ModelRuntime
from llm_infer.model.runtime import (
    ATTENTION_BACKEND_CHOICES,
    AttentionBackendChoice,
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
    prefix_cache: bool = False,
    decode_window_size: int = DEFAULT_DECODE_WINDOW_SIZE,
    decode_graphs: bool = True,
    decode_graph_buckets: tuple[int, ...] = DEFAULT_DECODE_GRAPH_BUCKETS,
    grouped_decode_graphs: bool | None = None,
    allow_cors: bool = False,
) -> FastAPI:
    """Wire a loaded model runtime into the HTTP app.

    Decode graphs are on by default: on a CUDA bundle model the piecewise decode-window
    graphs are captured here, before the engine starts serving, so the capture cost lands
    at startup, never inside a request. On CPU or non-bundle backends this is a no-op.

    ``grouped_decode_graphs`` additionally captures engine-owned grouped-layer graphs for
    the same buckets; an exact-batch window decodes through them, everything else falls
    back to the piecewise buckets. ``None`` (the serve default) enables it exactly where
    capture is possible — a CUDA bundle model with a native paged attention backend and
    piecewise graphs on — measured at +18% single-request and +7% batch-8 greedy HTTP
    tok/s for roughly 7-10 s of extra startup capture per bucket. Explicit ``True`` forces
    it (CPU engines then run the eager grouped path — the test hook); ``False`` disables.
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
    if grouped_decode_graphs is None:
        # Auto-enable only where grouped steps can actually run: capture needs a CUDA
        # bundle model on a native paged backend, and dispatch happens only inside
        # deferred decode windows — which per-step windows, preemption, and speculative
        # decoding all disable (see ``_window_eligible`` in serving/engine_decode.py).
        # Anything else would pay capture time and memory for a permanently-zero hit
        # counter. Explicit ``True`` still forces it for tests and debugging.
        grouped_decode_graphs = (
            decode_graphs
            and torch.device(device).type == "cuda"
            and runtime.capabilities.planned_decode
            and isinstance(runtime.model.backend, PagedDecodeAttentionBackend)
            and decode_window_size > 1
            and preemption_policy == "off"
            and prompt_lookup_speculative is None
        )
    grouped_start = time.perf_counter()
    engine = InferenceEngine(
        runtime.model,
        block_size=block_size,
        num_blocks=num_blocks,
        device=device,
        capabilities=runtime.capabilities,
        preemption=preemption_policy == "recompute",
        prefill_chunk_size=prefill_chunk_size,
        speculative=prompt_lookup_speculative,
        prefix_cache=prefix_cache,
        decode_window_size=decode_window_size,
        grouped_decode_graphs=grouped_decode_graphs,
        grouped_capture_sizes=decode_graph_buckets,
    )
    if grouped_decode_graphs and torch.device(device).type == "cuda":
        grouped_s = time.perf_counter() - grouped_start
        captured = tuple(sorted(engine.grouped_decode_runners))
        print(f"grouped decode graphs: captured buckets {captured} in {grouped_s:.1f} s")
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
    if allow_cors:
        # Off by default: the local server is same-origin (see register_webui). The Modal
        # deployment turns it on so the *local* webui's throughput bench can point its
        # base-URL field at the *.modal.run origin. The API carries no credentials or
        # user data, so a wildcard read-only surface is acceptable there.
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
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


@dataclass(frozen=True)
class RuntimeDefaults:
    """Resolved pre-load choices: device, dtype, and attention backend selector."""

    device: str
    dtype: torch.dtype
    attention_backend: AttentionBackendChoice


def resolve_runtime_defaults(
    *,
    device: str,
    dtype: torch.dtype | None,
    attention_backend: str,
    backend: str,
) -> RuntimeDefaults:
    """Resolve ``--device``/``--dtype``/``--attention-backend`` autos; explicit values win.

    Measured on the 214M bundle (8-turn chat, 2026-07), which fixes each auto:

    * device ``auto`` -> ``cpu``: mps fp16 served 1.7x *slower* than cpu fp32 (per-op launch
      overhead dominates this model's tiny per-step GEMMs). ``--device mps`` stays a supported
      opt-in for larger bundles where the tradeoff may flip.
    * dtype ``None`` -> fp16 on cpu/mps (~12% faster than fp32, byte-identical on that bench;
      the startup line labels any non-fp32 config experimental and ``--dtype float32``
      restores the reference config), fp32 on CUDA so GPU behavior does not move. bf16
      measured slower than fp32 on this CPU and is never picked automatically.
    * attention ``auto`` -> ``torch_sdpa`` for non-CUDA bundle backends (the fused local
      path); CUDA ``auto`` still means FlashInfer inside the loader, and non-bundle backends
      pass through unchanged.
    """
    resolved_device = device if device != "auto" else "cpu"
    device_type = torch.device(resolved_device).type
    resolved_dtype = dtype
    if resolved_dtype is None:
        resolved_dtype = torch.float16 if device_type in ("cpu", "mps") else torch.float32
    name = attention_backend
    if name == "auto" and backend in BUNDLE_BACKENDS and device_type != "cuda":
        name = "torch_sdpa"
    # argparse gates the passthrough to registered choices and the loader re-validates.
    return RuntimeDefaults(resolved_device, resolved_dtype, cast(AttentionBackendChoice, name))


@dataclass(frozen=True)
class EngineDefaults:
    """Resolved post-load engine features; these need the loaded backend's capabilities."""

    prompt_lookup: bool
    prefix_cache: bool


def resolve_engine_defaults(
    *,
    prompt_lookup: bool | None,
    prefix_cache: bool | None,
    device: str,
    backend: str,
    capabilities: BackendCapabilities,
) -> EngineDefaults:
    """Resolve the tri-state speculative/prefix-cache flags; explicit values win.

    Both default on for non-CUDA serving where the backend supports them, and off on CUDA:
    speculative decode would disable the deferred window and grouped graphs there, and the
    settled GPU serving config must not move. An explicit ``True`` on an unsupported backend
    is left on so the engine raises a clear error rather than silently ignoring the request.
    """
    local = torch.device(device).type != "cuda"
    resolved_lookup = prompt_lookup
    if resolved_lookup is None:
        resolved_lookup = capabilities.speculative and local
    resolved_cache = prefix_cache
    if resolved_cache is None:
        resolved_cache = (
            backend in BUNDLE_BACKENDS
            and capabilities.prefix_caching
            and capabilities.paged_kv
            and local
        )
    return EngineDefaults(resolved_lookup, resolved_cache)


_DTYPE_LABELS = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}


def describe_run_config(device: str, dtype: torch.dtype) -> str:
    """One startup line naming the resolved device/dtype and whether it is the fp32 reference.

    Only ``cpu`` + fp32 reproduces the bundle reference token-for-token; every other config
    (fp16, or MPS, or both) diverges within numerical noise and is labeled experimental so the
    difference is visible per repo policy, never silent.
    """
    dtype_label = _DTYPE_LABELS.get(dtype, str(dtype).removeprefix("torch."))
    config = f"{device} {dtype_label}"
    if device == "cpu" and dtype == torch.float32:
        return f"device/dtype: {config} (cpu fp32 reference config)"
    return f"device/dtype: experimental: {config} — reference is cpu fp32"


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
        help="Attention backend selector. auto uses FlashInfer for CUDA bf16/fp16 Esme bundles "
        "and torch_sdpa for local (non-CUDA) bundle serving.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. auto resolves to mps on Apple silicon, else cpu; cpu/mps/cuda are "
        "explicit.",
    )
    parser.add_argument(
        "--dtype",
        type=_dtype,
        default=None,
        help="Model dtype. Default is auto: fp16 on MPS, fp32 otherwise; an explicit value wins.",
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
        bundle_path = _bundle_path_for_backend(
            args.backend,
            explicit_bundle=args.bundle,
            env_bundle=env_bundle,
        )
    except argparse.ArgumentTypeError as exc:
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
