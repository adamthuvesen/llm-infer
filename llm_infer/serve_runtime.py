"""Build the serving app for a loaded ``ModelRuntime``, and resolve the serve defaults."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from fastapi import FastAPI

from llm_infer.kernels.base import PagedDecodeAttentionBackend
from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
from llm_infer.model.fused_elementwise import use_fused_elementwise
from llm_infer.model.interface import BackendCapabilities, ModelRuntime
from llm_infer.model.pretrain_bundle import PretrainBundleModel
from llm_infer.model.runtime import BUNDLE_BACKENDS, AttentionBackendChoice
from llm_infer.request.request import Request
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
DEFAULT_DECODE_WINDOW_SIZE = 8
DEFAULT_BACKEND = "esme"
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
    fused_elementwise: bool | None = None,
    allow_cors: bool = False,
) -> FastAPI:
    """Wire a loaded model runtime into the HTTP app.

    Decode graphs are on by default: on a CUDA bundle model the piecewise decode-window
    graphs are captured here, before the engine starts serving, so the capture cost lands
    at startup, never inside a request. On CPU or non-bundle backends this is a no-op.

    ``grouped_decode_graphs`` additionally captures engine-owned grouped-layer graphs for
    the same buckets; an exact-batch window decodes through them, everything else falls
    back to the piecewise buckets. The server groups every layer: full depth was 2.9-4.6%
    faster per step than the engine's four-layer default at batch 1-64, with exact parity
    (docs/findings.md entry 21). ``None`` (the serve default) enables it exactly where
    capture is possible — a CUDA bundle model with a native paged attention backend and
    piecewise graphs on. How much grouping pays depends on the host (entry 18). Explicit
    ``True`` forces it (CPU engines then run the eager grouped path — the test hook);
    ``False`` disables.

    ``fused_elementwise`` compiles RMSNorm and RoPE into fused kernels before any warmup or
    capture (``llm_infer/model/fused_elementwise.py``). ``None`` (the serve default) enables it
    for CUDA bundle models in bf16 or fp16: it cut 37-44% of each decode step, passed the
    near-tie gate, and adds about 40 s of compile time to a cold start (docs/findings.md entries
    23 and 24). The fp32 CUDA config stays the unfused reference. Explicit ``True`` needs a CUDA
    bundle model, and ``False`` disables it.
    """
    if fused_elementwise is None:
        fused_elementwise = (
            torch.device(device).type == "cuda"
            and isinstance(runtime.model, PretrainBundleModel)
            and runtime.model.dtype in (torch.bfloat16, torch.float16)
        )
    if fused_elementwise and (
        torch.device(device).type != "cuda" or not isinstance(runtime.model, PretrainBundleModel)
    ):
        raise ValueError("--fused-elementwise needs a CUDA bundle model")
    # The switch is process-wide, so set it either way: an earlier app may have turned it on.
    # Before any warmup or capture, so every graph records the chosen kernels.
    use_fused_elementwise(fused_elementwise)
    if fused_elementwise:
        print("elementwise kernels: fused with torch.compile")
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
        # decoding all disable (see ``_window_eligible`` in serving/engine.py).
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
        grouped_layers=(
            runtime.model.num_layers if isinstance(runtime.model, PretrainBundleModel) else None
        ),
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
      restores the reference config). CUDA keeps the fp32 reference dtype, and GPU runs pass
      ``--dtype bfloat16`` or ``--dtype float16`` to select FlashInfer. bf16 measured slower
      than fp32 on this CPU and is never picked automatically.
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

    Both default on for non-CUDA serving where the backend supports them, and off on CUDA.
    On CUDA a speculative verify step gives up the deferred decode window and grouped graphs,
    and it measured 3.5x to 4.2x slower per request at batch 1 (docs/findings.md entry 22).
    The prefix cache stays opt-in there. An explicit ``True`` on an unsupported backend
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


def bundle_path_for_backend(
    backend: str,
    *,
    explicit_bundle: Path | None,
    env_bundle: str | None,
) -> Path | None:
    if backend not in BUNDLE_BACKENDS:
        if explicit_bundle is not None:
            raise ValueError(f"backend {backend!r} does not accept --bundle")
        return None
    bundle = explicit_bundle or (Path(env_bundle) if env_bundle else None)
    if bundle is None:
        raise ValueError(f"--backend {backend} requires --bundle or $ESME_BUNDLE_PATH")
    return bundle
