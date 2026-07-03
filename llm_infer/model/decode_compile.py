"""torch.compile'd decode-window step with paged attention as an opaque custom op.

The manual piecewise runner (``decode_graph.py``) replays hand-captured CUDA graph
segments. This runner asks the compiler stack for the same job — Inductor fuses the
small-op long tail (norms, RoPE, projections, MLP) and ``mode="reduce-overhead"`` manages
CUDA graphs — while the paged-KV write, the ragged history gather, and the flash-attn
varlen call stay out of the traced graph as one opaque custom op per layer:

* **Custom op boundary.** ``llm_infer::paged_decode_attention`` writes into a
  caller-provided output buffer (``register_fake`` for tracing) and carries the
  ``cudagraph_unsafe`` tag where the torch build has it, so Inductor's graph partitioner
  keeps it out of any CUDA graph — its gather sizes grow every step and must never be
  baked into a replay.
* **Tensors, never Python ints.** Positions cross the step boundary as a per-bucket device
  buffer and sequence lengths live inside the read plan's ``cu_seqlens`` tensor, so Dynamo
  guards see stable Python state and steady-state steps trigger zero recompiles.
* **Bucket padding on the caller side.** Same buckets and padding contract as the manual
  runner: a batch pads up to the nearest capture size, pad rows flow through the
  row-independent math and are skipped by the eager attention slice. ``mark_dynamic`` on
  the batch dim keeps every bucket above 1 on one compiled artifact (bucket 1 specializes,
  as Dynamo always does). The buffers are deliberately *not* address-pinned: pinning adds
  an object-identity guard per bucket, which blows Dynamo's recompile limit under
  ``fullgraph``; compiler-managed CUDA graphs copy the two tiny ``(B,)`` inputs into their
  own placeholders instead.
* **Warmup at enable time, parity-checked.** Every bucket runs enough steps at startup to
  compile, record, and replay, so no compile cost can leak into a timed region — and each
  bucket's steps are compared against the eager planned path on the same synthetic
  workload. A torch build that CUDA-graph-captures *through* the custom op (baking one
  step's gather into the replay) fails loudly here instead of decoding garbage; callers
  can then retry with ``mode=None`` (fusion only, no compiler-managed graphs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch._dynamo
import torch._inductor.config as inductor_config

from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.decode_graph import DEFAULT_CAPTURE_SIZES, select_bucket
from llm_infer.model.layers import linear_projection, swiglu_mlp
from llm_infer.model.rope_utils import rms_norm

if TYPE_CHECKING:
    from llm_infer.kv_cache.paged_kv_cache import KVReadPlan
    from llm_infer.model.decode_plan import DecodeWindowPlan
    from llm_infer.model.pretrain_bundle import PretrainBundleModel

_WARMUP_PROMPT = [1, 2, 3]
_WARMUP_STEPS = 3  # compiled-eager, CUDA-graph record, CUDA-graph replay


@dataclass
class _ActiveStep:
    """The eager-side state the opaque attention op reads for the step in flight.

    The op's traced signature carries only tensors and the layer index; the cache handle,
    the real (unpadded) batch, and the step's write slots / read plan cross the boundary
    through this module-level slot instead, set and cleared around each compiled call.
    """

    model: PretrainBundleModel
    cache: PagedKVCache
    batch: int
    write_slots: torch.Tensor
    read_plan: KVReadPlan


_ACTIVE_STEP: _ActiveStep | None = None

# Keeps Inductor's graph partitioner from capturing through the op on torch builds that
# split at cudagraph_unsafe ops; harmless (empty) elsewhere — the warmup parity check is
# what actually rejects a build that captures through it anyway.
_ATTENTION_OP_TAGS = (torch.Tag.cudagraph_unsafe,) if hasattr(torch.Tag, "cudagraph_unsafe") else ()

torch.library.define(
    "llm_infer::paged_decode_attention",
    "(Tensor q, Tensor k, Tensor v, int layer, Tensor(a!) out) -> ()",
    tags=_ATTENTION_OP_TAGS,
)


@torch.library.impl("llm_infer::paged_decode_attention", "CompositeExplicitAutograd")
def _paged_decode_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: int, out: torch.Tensor
) -> None:
    """The eager per-layer slice: KV write, packed history gather, flash attention.

    Identical math to ``PretrainBundleModel._decode_attention_planned`` on the real batch
    rows only — pad rows are never written and never attend; their ``out`` rows keep
    whatever the buffer held (row-independent, dropped by the caller's ``[:batch]`` slice).
    """
    step = _ACTIVE_STEP
    if step is None:
        raise RuntimeError("paged_decode_attention ran outside CompiledDecodeRunner.window_step")
    batch = step.batch
    step.cache.write_rows(layer, step.write_slots, k[:batch], v[:batch])
    k_hist, v_hist = step.cache.read_many_plan(layer, step.read_plan)
    k_exp, v_exp = step.model._expand_kv_token_major(k_hist, v_hist)
    attn = step.model.backend.forward_decode_batch_packed(
        q[:batch], k_exp, v_exp, step.read_plan.cu_seqlens, step.read_plan.max_len
    )
    out[:batch].copy_(attn)


@torch.library.register_fake("llm_infer::paged_decode_attention")
def _paged_decode_attention_fake(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer: int, out: torch.Tensor
) -> None:
    return None


def dynamo_counters_snapshot() -> dict[str, int]:
    """Recompile/cudagraph counters for the harness audit (deltas must be zero mid-bench)."""
    from torch._dynamo.utils import counters

    return {
        "unique_graphs": int(counters["stats"].get("unique_graphs", 0)),
        "cudagraph_skips": int(counters["inductor"].get("cudagraph_skips", 0)),
    }


@dataclass
class _CompiledBucket:
    """Static input buffers for one padded batch size."""

    size: int
    tokens: torch.Tensor  # (size,) long — this step's input token per row
    positions: torch.Tensor  # (size,) long — RoPE position per row; +1 per step by the caller


class CompiledDecodeRunner:
    """Bucket-padded ``torch.compile`` path for ``PretrainBundleModel.decode_window_step``.

    Same caller contract as :class:`~llm_infer.model.decode_graph.DecodeGraphRunner` (the
    model routes window steps through whichever runner sits in ``model.decode_graphs``):
    ``window_step`` returns padded-and-sliced logits, or ``None`` when the plan must run on
    the eager planned path — batch above the largest bucket, window reaching past the
    pinned RoPE rows, or a window that already started eager.
    """

    def __init__(
        self,
        model: PretrainBundleModel,
        *,
        capture_sizes: tuple[int, ...] = DEFAULT_CAPTURE_SIZES,
        max_position: int = 8192,
        mode: str | None = "reduce-overhead",
        compile_backend: str = "inductor",
        warmup: bool = True,
    ) -> None:
        sizes = tuple(sorted(set(capture_sizes)))
        if not sizes or sizes[0] < 1:
            raise ValueError(f"capture_sizes must be positive; got {capture_sizes!r}")
        if max_position < 1:
            raise ValueError(f"max_position must be >= 1; got {max_position}")
        self.model = model
        self.capture_sizes = sizes
        self.mode = mode
        self.compile_backend = compile_backend
        # Pin the RoPE rows the compiled graphs read, exactly like the manual runner: the
        # eager path may later replace the model's table, so hold strong references and
        # bound every compiled-path position by them.
        cos, sin = model._ensure_rope_rows(max_position)
        self._rope_cos = cos
        self._rope_sin = sin
        self.max_position = int(cos.shape[0])
        self._active_plan: DecodeWindowPlan | None = None
        # Pre-populate the model's fp32 norm-weight cache so tracing never mutates it
        # (a dict write mid-trace is a graph break under fullgraph=True).
        for layer in range(model.num_layers):
            prefix = f"layers.{layer}."
            model._norm_weight(prefix + "input_norm.weight")
            model._norm_weight(prefix + "post_attention_norm.weight")
            if model.qk_norm:
                model._norm_weight(prefix + "attn.q_norm.weight")
                model._norm_weight(prefix + "attn.k_norm.weight")
        model._norm_weight("norm.weight")
        if mode == "reduce-overhead" and hasattr(inductor_config, "graph_partition"):
            # Partition Inductor's CUDA graphs around cudagraph_unsafe ops instead of
            # skipping (or worse, capturing through) the attention boundary.
            inductor_config.graph_partition = True
        self._buckets = {size: self._build_bucket(size) for size in sizes}
        if compile_backend == "inductor":
            self._step = torch.compile(self._decode_step, mode=mode, fullgraph=True)
        else:
            self._step = torch.compile(self._decode_step, backend=compile_backend, fullgraph=True)
        self.warmup_s: float | None = None
        if warmup:
            start = time.perf_counter()
            self._warmup_and_check()
            self.warmup_s = time.perf_counter() - start

    def bucket_for(self, batch: int) -> int | None:
        """The padded size ``batch`` would run at, or ``None`` (caller stays eager)."""
        return select_bucket(self.capture_sizes, batch)

    @torch.no_grad()
    def window_step(
        self, cache: PagedKVCache, plan: DecodeWindowPlan, token_ids: torch.Tensor
    ) -> torch.Tensor | None:
        """One planned decode step through the compiled padded path.

        Returns ``(batch, vocab)`` logits — with compiler-managed CUDA graphs a view into
        graph-pool memory, valid only until the next step (the engine consumes them
        immediately via argmax) — or ``None`` when this plan cannot run here; the caller
        then runs the eager planned step. The plan advances exactly once either way.
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
            # pinned RoPE rows (the caller still advances the whole buffer by +1 per step).
            state.positions[batch:].zero_()
            self._active_plan = plan
        elif self._active_plan is not plan:
            return None

        write_slots, read_plan = plan.begin_step()
        state.tokens[:batch].copy_(token_ids.reshape(-1))
        global _ACTIVE_STEP
        _ACTIVE_STEP = _ActiveStep(
            model=self.model,
            cache=cache,
            batch=batch,
            write_slots=write_slots,
            read_plan=read_plan,
        )
        try:
            logits = self._step(state.tokens, state.positions)
        finally:
            _ACTIVE_STEP = None
        state.positions.add_(1)
        plan.complete_step()
        return logits[:batch]

    def _decode_step(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """The traced whole-step function: embed → layers (opaque attention) → logits.

        Same math as ``decode_window_step``'s eager body, written over the padded bucket.
        Everything here is functional over ``tokens``/``positions`` and the model weights;
        the only side effects live inside the opaque attention op.
        """
        model = self.model
        hidden = model.w["embed_tokens.weight"].index_select(0, tokens).to(model.dtype)
        cos = self._rope_cos.index_select(0, positions)
        sin = self._rope_sin.index_select(0, positions)
        for layer in range(model.num_layers):
            prefix = f"layers.{layer}."
            residual = hidden
            x = rms_norm(hidden, model._norm_weight(prefix + "input_norm.weight"), model.rms_eps)
            q, k, v = model._project_heads(x, prefix)
            q, k = model._qk_norm_rope(q, k, cos, sin, prefix)
            q_rows = q.transpose(0, 1).contiguous()  # (bucket, num_heads, head_dim)
            k_rows = k.transpose(0, 1).contiguous()
            v_rows = v.transpose(0, 1).contiguous()
            # Pad rows keep whatever the fresh buffer holds — row-independent garbage the
            # ``[:batch]`` slices never see.
            attn = torch.empty_like(q_rows)
            torch.ops.llm_infer.paged_decode_attention(q_rows, k_rows, v_rows, layer, attn)
            merged = attn.reshape(attn.shape[0], model.num_heads * model.head_dim)
            hidden = residual + linear_projection(
                model.w, merged, prefix + "attn.o_proj", model.dtype
            )
            x = rms_norm(
                hidden, model._norm_weight(prefix + "post_attention_norm.weight"), model.rms_eps
            )
            hidden = hidden + swiglu_mlp(model.w, x, prefix, model.dtype)
        x = rms_norm(hidden, model._norm_weight("norm.weight"), model.rms_eps)
        return model._apply_logit_soft_cap(x @ model._lm_head().T)

    def _build_bucket(self, size: int) -> _CompiledBucket:
        device = self.model.device
        tokens = torch.zeros(size, dtype=torch.long, device=device)
        positions = torch.zeros(size, dtype=torch.long, device=device)
        # Dynamo specializes size 1 regardless (0/1 specialization); marking it dynamic
        # would raise, so only the >1 buckets share the dynamic-batch artifact. No
        # mark_static_address: its per-tensor identity guard would cost one Dynamo entry
        # per bucket and trip the recompile limit under fullgraph.
        if size > 1:
            torch._dynamo.mark_dynamic(tokens, 0)
            torch._dynamo.mark_dynamic(positions, 0)
        return _CompiledBucket(size=size, tokens=tokens, positions=positions)

    @torch.no_grad()
    def _warmup_and_check(self) -> None:
        """Compile, record, and replay every bucket; pin each against the eager planned path.

        Runs the same tiny synthetic workload through this runner and through the model's
        eager planned window, comparing per-step logits. With compiler-managed CUDA graphs
        the last warmup step is a replay, so a build that captured through the attention op
        (stale gather baked in) diverges here and raises instead of serving wrong tokens.
        """
        for size in self.capture_sizes:
            compiled = self._run_synthetic_window(size, use_runner=True)
            eager = self._run_synthetic_window(size, use_runner=False)
            for step, (got, expected) in enumerate(zip(compiled, eager, strict=True)):
                if not torch.allclose(got, expected, rtol=0.05, atol=0.5):
                    worst = (got - expected).abs().max().item()
                    raise RuntimeError(
                        f"compiled decode step diverged from the eager planned path at "
                        f"bucket {size}, warmup step {step} (max |Δlogit| {worst:.3f}) — "
                        f"this torch build likely CUDA-graph-captures through the paged "
                        f"attention op; retry with mode=None"
                    )
        # Drop the last synthetic plan so its throwaway cache/buffers can be freed and no
        # real window is ever mistaken for a resumed warmup window.
        self._active_plan = None

    def _run_synthetic_window(self, size: int, *, use_runner: bool) -> list[torch.Tensor]:
        """Prefill ``size`` tiny requests on a throwaway cache and decode one window."""
        model = self.model
        cache = PagedKVCache(
            num_layers=model.num_layers,
            num_blocks=size + 4,
            block_size=16,
            num_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
            dtype=model.dtype,
            device=model.device,
        )
        tables = []
        first_tokens = []
        for _ in range(size):
            table = cache.new_request()
            logits = model.prefill(list(_WARMUP_PROMPT), cache, table)
            tables.append(table)
            first_tokens.append(torch.argmax(logits))
        tokens = torch.stack(first_tokens)
        plan = model.open_decode_window(cache, tables, _WARMUP_STEPS)
        if plan is None:
            raise RuntimeError("warmup window unexpectedly not plan-safe")

        saved_runner = model.decode_graphs
        model.decode_graphs = None  # the eager reference must not reroute into any runner
        try:
            steps: list[torch.Tensor] = []
            for _ in range(_WARMUP_STEPS):
                if use_runner:
                    logits = self.window_step(cache, plan, tokens)
                    if logits is None:
                        raise RuntimeError(f"warmup bucket {size} fell back to the eager path")
                else:
                    logits = model.decode_window_step(cache, plan, tokens)
                snapshot = logits.detach().clone().float()
                steps.append(snapshot)
                tokens = torch.argmax(snapshot, dim=-1)
            return steps
        finally:
            model.decode_graphs = saved_runner
