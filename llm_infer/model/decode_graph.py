"""Piecewise CUDA-graph capture for the planned decode window (the vLLM shape).

The decode wall at 214M is kernel-launch dispatch: ~1,850 tiny kernels per token-step
after the planned-window work. This module replays most of them from CUDA graphs instead
of relaunching them from Python:

* **Bucketed, padded, captured once.** Graphs are captured for a fixed set of batch-size
  buckets; a real batch is padded up to the nearest bucket and replayed. Static input
  buffers are ``copy_``-ed into, addresses never change, and nothing is ever recaptured per
  batch composition — per-shape recapture is the measured losing pattern from the archived
  Qwen-era slice (docs/internal/decode-graph-plan.md).
* **Piecewise, attention eager.** Each decode step replays captured segments for the layer
  math (norms, projections, QK-norm, RoPE, MLP, logits) while the paged-KV write, the
  packed history gather, and the flash-attn varlen call run eagerly between segments. The
  ragged KV history changes size every step, so keeping it out of the graphs removes all
  shape dynamism; the captured segments see only fixed ``(bucket, ...)`` tensors. The KV
  write also stays eager so no graph ever bakes in a cache tensor address — the runner
  outlives any single engine (and its cache).
* **Padding is contained.** Pad rows flow through the captured segments (row-independent
  math; their garbage never mixes into real rows) and are *skipped* by the eager attention
  — pad K/V is never written and pad queries never attend. Pad logits are dropped by the
  ``[:batch]`` slice. This is strictly less work than attending pad rows to a dummy block.
* **One memory pool.** All buckets' graphs share one capture pool; every tensor that
  crosses a segment boundary (hidden, residual, q/k/v, attention out, logits) lives in a
  static buffer allocated *outside* the pool, so interleaving replays across buckets can
  never alias live state.

``mode="eager"`` runs the exact same segment functions and static-buffer flow without
capture — CPU-runnable, which is what pins padding/copy semantics in unit tests; on the
GPU the only delta left is the capture itself.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from llm_infer.model.layers import linear_projection, swiglu_mlp
from llm_infer.model.rope_utils import rms_norm

if TYPE_CHECKING:
    from llm_infer.kv_cache.paged_kv_cache import KVReadPlan, PagedKVCache
    from llm_infer.model.decode_plan import DecodeWindowPlan
    from llm_infer.model.pretrain_bundle import PretrainBundleModel

DEFAULT_CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)


def enable_decode_graphs_if_cuda(
    model: object, capture_sizes: tuple[int, ...] = DEFAULT_CAPTURE_SIZES
) -> float | None:
    """Capture decode graphs when the model supports them on CUDA; return capture seconds.

    The one call serving and benchmark entry points make at startup. Returns ``None`` (and
    changes nothing) when the backend has no ``enable_decode_graphs`` (Qwen) or the model is
    not on CUDA, where capture is impossible; the eager planned window keeps running there.
    Callers log the returned capture time so the startup cost stays visible.
    """
    enable = getattr(model, "enable_decode_graphs", None)
    device = getattr(model, "device", None)
    if enable is None or device is None or device.type != "cuda":
        return None
    start = time.perf_counter()
    enable(capture_sizes)
    return time.perf_counter() - start


def select_bucket(capture_sizes: tuple[int, ...], batch: int) -> int | None:
    """The smallest capture size that fits ``batch``, or ``None`` when none does."""
    if batch < 1:
        raise ValueError(f"batch must be >= 1; got {batch}")
    for size in capture_sizes:
        if size >= batch:
            return size
    return None


@dataclass
class _BucketState:
    """Static buffers and captured segments for one padded batch size."""

    size: int
    tokens: torch.Tensor  # (size,) long — this step's input token per row
    positions: torch.Tensor  # (size,) long — RoPE position per row; +1 inside the last segment
    hidden: torch.Tensor  # (size, hidden) — layer-loop hidden state
    residual: torch.Tensor  # (size, hidden) — attention residual across the eager boundary
    q: torch.Tensor  # (size, num_heads, head_dim) — pre-attention output for the eager call
    k: torch.Tensor  # (size, num_kv_heads, head_dim)
    v: torch.Tensor  # (size, num_kv_heads, head_dim)
    attn: torch.Tensor  # (size, num_heads, head_dim) — eager attention output, graph input
    logits: torch.Tensor  # (size, vocab)
    run_segments: list[Callable[[], None]]
    graphs: list[torch.cuda.CUDAGraph] | None = None


class DecodeGraphRunner:
    """Bucket-padded piecewise graph replay for ``PretrainBundleModel.decode_window_step``.

    Owned by the model, not an engine: the captured segments reference only model weights,
    the runner's static buffers, and a pinned-down RoPE table — never a ``PagedKVCache``
    tensor — so one capture serves every engine built on the model. All buckets are captured
    up front at construction so no capture cost can leak into a timed region later.
    """

    def __init__(
        self,
        model: PretrainBundleModel,
        *,
        capture_sizes: tuple[int, ...] = DEFAULT_CAPTURE_SIZES,
        max_position: int = 8192,
        mode: str = "graph",
    ) -> None:
        if mode not in ("graph", "eager"):
            raise ValueError(f"mode must be 'graph' or 'eager'; got {mode!r}")
        sizes = tuple(sorted(set(capture_sizes)))
        if not sizes or sizes[0] < 1:
            raise ValueError(f"capture_sizes must be positive; got {capture_sizes!r}")
        if max_position < 1:
            raise ValueError(f"max_position must be >= 1; got {max_position}")
        if mode == "graph" and model.device.type != "cuda":
            raise ValueError("mode='graph' needs a CUDA model; use mode='eager' elsewhere")
        self.model = model
        self.capture_sizes = sizes
        self.mode = mode
        # Pin the RoPE rows the graphs read. The eager path may later *replace* the model's
        # table (``_ensure_rope_rows`` reallocates on growth); these references keep the
        # captured memory alive, and the ``max_position`` bound keeps every graph-path
        # position inside it, where the row values are immutable and correct.
        cos, sin = model._ensure_rope_rows(max_position)
        self._rope_cos = cos
        self._rope_sin = sin
        self.max_position = int(cos.shape[0])
        # The plan currently pipelined through the buckets; a plan that fell back to the
        # eager path at its first step must stay eager for its whole window (the positions
        # buffer is loaded only at step 0).
        self._active_plan: DecodeWindowPlan | None = None
        self._buckets = {size: self._build_bucket(size) for size in sizes}
        if mode == "graph":
            self._capture_all()

    def bucket_for(self, batch: int) -> int | None:
        """The padded size ``batch`` would replay at, or ``None`` (caller stays eager)."""
        return select_bucket(self.capture_sizes, batch)

    @torch.no_grad()
    def window_step(
        self, cache: PagedKVCache, plan: DecodeWindowPlan, token_ids: torch.Tensor
    ) -> torch.Tensor | None:
        """One planned decode step through the padded bucket path.

        Returns ``(batch, vocab)`` logits — a view into the bucket's static buffer, valid
        only until the next step — or ``None`` when this plan cannot run here (batch above
        the largest bucket, window reaching past the pinned RoPE rows, or a window that
        already started on the eager path). ``None`` means the caller runs the eager planned
        step; the plan advances exactly once either way.
        """
        batch = len(plan.tables)
        bucket_size = self.bucket_for(batch)
        if bucket_size is None:
            return None
        state = self._buckets[bucket_size]
        if plan.steps_used == 0:
            if max(plan.base_lengths) + plan.budget > self.max_position:
                return None
            state.positions[:batch].copy_(plan.positions)
            # Pad positions restart at 0 each window so they can never creep past the
            # pinned RoPE rows (they still advance by +1 per step inside the last segment).
            state.positions[batch:].zero_()
            self._active_plan = plan
        elif self._active_plan is not plan:
            return None

        write_slots, read_plan = plan.begin_step()
        state.tokens[:batch].copy_(token_ids.reshape(-1))
        self._run_segment(state, 0)
        for layer in range(self.model.num_layers):
            self._attention_eager(state, layer, cache, batch, write_slots, read_plan)
            self._run_segment(state, layer + 1)
        plan.complete_step()
        return state.logits[:batch]

    def _run_segment(self, state: _BucketState, index: int) -> None:
        if state.graphs is not None:
            state.graphs[index].replay()
        else:
            state.run_segments[index]()

    def _attention_eager(
        self,
        state: _BucketState,
        layer: int,
        cache: PagedKVCache,
        batch: int,
        write_slots: torch.Tensor,
        read_plan: KVReadPlan,
    ) -> None:
        """The per-layer eager slice: KV write, packed history gather, flash attention.

        Identical math to ``PretrainBundleModel._decode_attention_planned`` — same write
        slots, same packed gather, same GQA expansion, same backend call — operating on the
        real ``batch`` rows only. Pad rows are never written and never attend.
        """
        model = self.model
        cache.write_rows(layer, write_slots, state.k[:batch], state.v[:batch])
        k_hist, v_hist = cache.read_many_plan(layer, read_plan)
        k_exp, v_exp = model._expand_kv_token_major(k_hist, v_hist)
        out = model.backend.forward_decode_batch_packed(
            state.q[:batch], k_exp, v_exp, read_plan.cu_seqlens, read_plan.max_len
        )
        state.attn[:batch].copy_(out)

    def _pre_attention(self, state: _BucketState, layer: int) -> None:
        """Input norm, Q/K/V projection, QK-norm, RoPE — into the q/k/v static buffers."""
        model = self.model
        prefix = f"layers.{layer}."
        state.residual.copy_(state.hidden)
        x = rms_norm(state.hidden, model._norm_weight(prefix + "input_norm.weight"), model.rms_eps)
        q, k, v = model._project_heads(x, prefix)
        cos = self._rope_cos.index_select(0, state.positions)
        sin = self._rope_sin.index_select(0, state.positions)
        q, k = model._qk_norm_rope(q, k, cos, sin, prefix)
        state.q.copy_(q.transpose(0, 1))
        state.k.copy_(k.transpose(0, 1))
        state.v.copy_(v.transpose(0, 1))

    def _post_attention(self, state: _BucketState, layer: int) -> None:
        """Output projection + residual, post-attention norm, MLP + residual — into hidden."""
        model = self.model
        prefix = f"layers.{layer}."
        merged = state.attn.reshape(state.size, model.num_heads * model.head_dim)
        hidden = state.residual + linear_projection(
            model.w, merged, prefix + "attn.o_proj", model.dtype
        )
        x = rms_norm(
            hidden, model._norm_weight(prefix + "post_attention_norm.weight"), model.rms_eps
        )
        state.hidden.copy_(hidden + swiglu_mlp(model.w, x, prefix, model.dtype))

    def _build_bucket(self, size: int) -> _BucketState:
        model = self.model
        device = model.device
        hidden_size = int(model.w["embed_tokens.weight"].shape[1])
        vocab_size = model.config.vocab_size

        def buf(*shape: int, dtype: torch.dtype = model.dtype) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype, device=device)

        state = _BucketState(
            size=size,
            tokens=buf(size, dtype=torch.long),
            positions=buf(size, dtype=torch.long),
            hidden=buf(size, hidden_size),
            residual=buf(size, hidden_size),
            q=buf(size, model.num_heads, model.head_dim),
            k=buf(size, model.num_kv_heads, model.head_dim),
            v=buf(size, model.num_kv_heads, model.head_dim),
            attn=buf(size, model.num_heads, model.head_dim),
            logits=buf(size, vocab_size),
            run_segments=[],
        )
        state.run_segments = self._make_segments(state)
        return state

    def _make_segments(self, state: _BucketState) -> list[Callable[[], None]]:
        """The capturable pieces, in replay order: embed+pre(0), post(l-1)+pre(l)…, final."""
        model = self.model

        def first() -> None:
            embedded = model.w["embed_tokens.weight"].index_select(0, state.tokens)
            state.hidden.copy_(embedded.to(model.dtype))
            self._pre_attention(state, 0)

        def mid(layer: int) -> Callable[[], None]:
            def run() -> None:
                self._post_attention(state, layer - 1)
                self._pre_attention(state, layer)

            return run

        def last() -> None:
            self._post_attention(state, model.num_layers - 1)
            x = rms_norm(state.hidden, model._norm_weight("norm.weight"), model.rms_eps)
            state.logits.copy_(model._apply_logit_soft_cap(x @ model._lm_head().T))
            state.positions += 1

        return [first, *(mid(layer) for layer in range(1, model.num_layers)), last]

    @torch.no_grad()
    def _capture_all(self) -> None:
        """Warm up on a side stream, then capture every bucket into one shared pool.

        Warmup and capture touch only static buffers and model weights (KV writes are
        eager-only), so nothing here can corrupt engine state. Buckets are captured largest
        first so the shared pool is sized once.
        """
        torch.cuda.synchronize()
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for size in sorted(self._buckets, reverse=True):
                for _ in range(3):
                    for segment in self._buckets[size].run_segments:
                        segment()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        for size in sorted(self._buckets, reverse=True):
            state = self._buckets[size]
            graphs: list[torch.cuda.CUDAGraph] = []
            for segment in state.run_segments:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    segment()
                graphs.append(graph)
            state.graphs = graphs
        torch.cuda.synchronize()
