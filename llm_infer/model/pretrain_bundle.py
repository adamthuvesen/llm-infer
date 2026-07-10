"""Loader and reference-checked forward pass for esme-pretrain dense export bundles.

The weights are Esme's own DenseBackbone export; the forward is ours, routed through the
shared serving engine. Two decode paths share one layer stack:

* :meth:`logits` — full recompute over the whole sequence, no cache. The known-good
  reference oracle; left intact in meaning.
* :meth:`prefill` / :meth:`prefill_chunk` / :meth:`decode_one` / :meth:`decode_many` /
  :meth:`decode_tokens` — the real paged-KV path. Prefill writes prompt K/V into the paged
  store; each decode step computes new token(s), appends K/V, and attends against the
  gathered history through the same ``torch_naive`` backend. This is the shared engine
  prefill/decode + scheduler path — not an Esme-only fork.

Esme has two model details that the paged path must respect:

* QK-norm — when configured, the per-head RMSNorm on Q and K is applied **before** RoPE, so
  the K written to the paged store is post-QK-norm, post-RoPE (the rotation is fixed by the
  token's absolute position, computed once at write time).
* Logit soft cap — a ``tanh`` cap applied to the **final** logits only. Every path that forms
  logits (full recompute and cached) applies it, so cached logits match the reference.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import nullcontext
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from llm_infer.kernels.base import (
    AttentionBackend,
    PackedPrefillAttentionBackend,
    PagedDecodeAttentionBackend,
)
from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.kv_cache.paged_kv_cache import KVReadPlan, PagedKVCache
from llm_infer.model.decode_graph import DEFAULT_CAPTURE_SIZES, DecodeGraphRunner
from llm_infer.model.decode_plan import DecodeWindowPlan, build_decode_window_plan
from llm_infer.model.layers import (
    expand_grouped_kv,
    linear_projection,
    merge_attention_heads,
    rope_tables_for_positions,
    swiglu_mlp,
)
from llm_infer.model.pretrain_bundle_loader import (
    BUNDLE_FORMAT,
    PretrainBundleError,
    PretrainDenseConfig,
    normalize_state_dict,
    read_json_object,
    read_weights,
    require_manifest_format,
    require_weight_key_format,
    required_file,
    resolve_tokenizer_path,
)
from llm_infer.model.rope_utils import apply_rope, rms_norm
from llm_infer.profiling import TimingProfiler

if TYPE_CHECKING:
    from llm_infer.model.decode_compile import CompiledDecodeRunner

__all__ = ["BUNDLE_FORMAT", "PretrainBundleError", "PretrainBundleModel"]

# The closure each decoder layer calls for its attention block: (x, prefix, layer) -> out.
_AttentionFn = Callable[[torch.Tensor, str, int], torch.Tensor]


class PretrainBundleModel:
    """Reference-checked paged-KV inference for ``llm_pretrain_dense_v1`` bundles.

    A first-class serving backend for exported Esme/DenseBackbone weights: it writes and
    reads real paged K/V through the shared engine prefill/decode path.
    :meth:`logits` stays the full-recompute reference oracle the paged path is validated
    against.
    """

    def __init__(
        self,
        *,
        weights: Mapping[str, torch.Tensor],
        config: PretrainDenseConfig,
        tokenizer_path: Path,
        backend: AttentionBackend,
        dtype: torch.dtype,
    ) -> None:
        self.w = dict(weights)
        self.config = config
        self.tokenizer_path = tokenizer_path
        self.backend = backend
        # Resolve the native-paged capability once. The Protocol is runtime_checkable, so
        # ``isinstance`` scans attributes on py3.11 — too costly to repeat per layer per decode
        # step. The backend never changes after construction, so cache the narrowed reference.
        self._paged_backend: PagedDecodeAttentionBackend | None = (
            backend if isinstance(backend, PagedDecodeAttentionBackend) else None
        )
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rms_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.tie_word_embeddings = config.tie_word_embeddings
        self.qk_norm = config.qk_norm
        self.device = self.w["embed_tokens.weight"].device
        self.profiler: TimingProfiler | None = None
        # Cached RoPE cos/sin rows in the model dtype, one row per absolute position; grown on
        # demand and indexed by the decode paths so per-step trig and host-to-device position
        # copies disappear. Same values as computing each row directly — the cache is the whole
        # ``rope_tables_for_positions(arange(n))`` table, cast once instead of once per layer.
        self._rope_rows_cos: torch.Tensor | None = None
        self._rope_rows_sin: torch.Tensor | None = None
        # RMSNorm weights pre-cast to fp32, keyed by weight name. rms_norm runs in fp32, so a
        # bf16 model otherwise re-casts every norm weight on every call — 4 per layer per
        # decode step. Same values, cast once. For an fp32 model ``to`` returns the weight
        # itself, so this caches nothing new.
        self._norm_weights_fp32: dict[str, torch.Tensor] = {}
        # Fast-path runner for the planned decode window (None = eager): the piecewise
        # CUDA-graph runner from :meth:`enable_decode_graphs` or the torch.compile runner
        # from :meth:`enable_decode_compile`. Assign None to disable without dropping the
        # capture (keep the runner object around and reassign it to re-enable).
        self.decode_graphs: DecodeGraphRunner | CompiledDecodeRunner | None = None

    @classmethod
    def load(
        cls,
        bundle_path: Path | str,
        *,
        dtype: torch.dtype = torch.float32,
        backend: AttentionBackend | None = None,
        device: torch.device | str = "cpu",
    ) -> PretrainBundleModel:
        """Load and validate an exported esme-pretrain dense bundle."""
        root = Path(bundle_path)
        if not root.is_dir():
            raise PretrainBundleError(f"bundle path must be a directory: {root}")

        manifest_path = required_file(root, "manifest.json")
        config_path = required_file(root, "config.json")
        weights_path = required_file(root, "weights.pt")

        manifest = read_json_object(manifest_path)
        require_manifest_format(manifest, manifest_path)
        tokenizer_path = resolve_tokenizer_path(root, manifest)
        read_json_object(tokenizer_path)

        config = PretrainDenseConfig.from_json(read_json_object(config_path))
        state_dict, metadata = read_weights(weights_path, device)
        require_weight_key_format(metadata, weights_path)
        weights = normalize_state_dict(state_dict, config, dtype=dtype, device=device)

        return cls(
            weights=weights,
            config=config,
            tokenizer_path=tokenizer_path,
            backend=backend or TorchNaiveAttention(),
            dtype=dtype,
        )

    @torch.no_grad()
    def logits(self, token_ids: list[int]) -> torch.Tensor:
        """Next-token logits for every input position. Shape ``(seq_len, vocab_size)``.

        Full recompute over the whole sequence — no cache. The known-good reference oracle.
        """
        self._validate_token_ids(token_ids)
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        hidden = self.w["embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_tables(len(token_ids))
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden, layer, lambda x, p, _lyr: self._attention(x, p, cos, sin)
            )

        with self._profile("logits"):
            hidden = rms_norm(hidden, self._norm_weight("norm.weight"), self.rms_eps)
            return self._apply_logit_soft_cap(hidden @ self._lm_head().T)

    @torch.no_grad()
    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        """Cached prefill: run the prompt, store all K/V, return last-position logits.

        Hidden states match :meth:`logits` (the cache write is a side effect that does not
        touch the values the backend sees). Only the last row's logits are formed, since greedy
        needs only the first generated token. Sets ``table.length``.
        """
        self._validate_token_ids(prompt_ids)
        if table.length != 0:
            raise PretrainBundleError(f"prefill expected an empty table; got length {table.length}")
        seq_len = len(prompt_ids)
        table.reserve(seq_len)
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        hidden = self.w["embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_tables(seq_len)
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._prefill_attention(x, p, cos, sin, lyr, cache, table),
            )
        table.length = seq_len

        with self._profile("logits"):
            last = rms_norm(hidden[-1:], self._norm_weight("norm.weight"), self.rms_eps)
            return self._apply_logit_soft_cap((last @ self._lm_head().T)[-1])

    @torch.no_grad()
    def prefill_many(
        self,
        prompts: list[list[int]],
        cache: PagedKVCache,
        tables: list[BlockTable],
    ) -> torch.Tensor:
        """Prefill ragged full prompts in one packed layer stack.

        Prompt rows are concatenated without padding. Packed causal attention keeps each
        request isolated while the projections, norms, MLPs, and output head run over all
        prompt tokens together. Returns one next-token logit row per request, ``(B, vocab)``.
        """
        if not prompts:
            raise ValueError("prefill_many needs at least one prompt")
        if len(prompts) != len(tables):
            raise ValueError(f"prompts/tables length mismatch: {len(prompts)} vs {len(tables)}")
        for prompt in prompts:
            self._validate_token_ids(prompt)
        nonempty_lengths = [table.length for table in tables if table.length != 0]
        if nonempty_lengths:
            raise PretrainBundleError(
                f"prefill_many expected empty tables; got lengths {nonempty_lengths}"
            )
        backend = self.backend
        if not isinstance(backend, PackedPrefillAttentionBackend):
            raise PretrainBundleError("attention backend does not support packed prefill")

        lengths = [len(prompt) for prompt in prompts]
        for table, length in zip(tables, lengths, strict=True):
            table.reserve(length)

        packed_ids = torch.tensor(
            [token for prompt in prompts for token in prompt],
            dtype=torch.long,
            device=self.device,
        )
        offsets = [0, *accumulate(lengths)]
        total_tokens = offsets[-1]
        cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        request_starts = torch.repeat_interleave(
            cu_seqlens[:-1],
            torch.tensor(lengths, dtype=torch.long, device=self.device),
        )
        positions = torch.arange(
            total_tokens, dtype=torch.float32, device=self.device
        ) - request_starts
        write_slots = torch.as_tensor(
            [
                slot
                for table, length in zip(tables, lengths, strict=True)
                for slot in table.physical_slots(0, length)
            ],
            dtype=torch.long,
            device=self.device,
        )

        hidden = self.w["embed_tokens.weight"][packed_ids].to(self.dtype)
        cos, sin = self._rope_for_positions(positions)
        max_len = max(lengths)
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._prefill_attention_packed(
                    x,
                    p,
                    cos,
                    sin,
                    lyr,
                    cache,
                    write_slots,
                    cu_seqlens,
                    max_len,
                    backend,
                ),
            )

        final_positions = torch.tensor(
            [offset - 1 for offset in offsets[1:]], dtype=torch.long, device=self.device
        )
        with self._profile("logits"):
            last = hidden.index_select(0, final_positions)
            last = rms_norm(last, self._norm_weight("norm.weight"), self.rms_eps)
            logits = self._apply_logit_soft_cap(last @ self._lm_head().T)
        for table, length in zip(tables, lengths, strict=True):
            table.length = length
        return logits

    @torch.no_grad()
    def prefill_chunk(
        self,
        prompt_ids: list[int],
        cache: PagedKVCache,
        table: BlockTable,
        *,
        start_pos: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """Cached prefill for ``prompt_ids[start_pos:end]``.

        Every layer writes the current chunk's K/V at its absolute prompt positions, gathers the
        prefix-plus-current history for that layer, and runs causal attention for the chunk
        queries only. Returns the logits for the last token in the chunk.
        """
        self._validate_token_ids(prompt_ids)
        if not 0 <= start_pos < len(prompt_ids):
            raise ValueError(f"start_pos must be in [0, {len(prompt_ids)}); got {start_pos}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
        if table.length != start_pos:
            raise ValueError(
                f"chunk start {start_pos} must equal cached prompt length {table.length}"
            )

        end_pos = min(len(prompt_ids), start_pos + chunk_size)
        chunk_ids = prompt_ids[start_pos:end_pos]
        table.reserve(len(chunk_ids))
        ids = torch.tensor(chunk_ids, dtype=torch.long, device=self.device)
        hidden = self.w["embed_tokens.weight"][ids].to(self.dtype)

        positions = torch.arange(start_pos, end_pos, dtype=torch.float32, device=self.device)
        cos, sin = self._rope_for_positions(positions)
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._prefill_chunk_attention(
                    x, p, cos, sin, lyr, cache, table, start_pos, end_pos
                ),
            )
        table.length = end_pos

        with self._profile("logits"):
            last = rms_norm(hidden[-1:], self._norm_weight("norm.weight"), self.rms_eps)
            return self._apply_logit_soft_cap((last @ self._lm_head().T)[-1])

    @torch.no_grad()
    def decode_one(
        self, cache: PagedKVCache, table: BlockTable, token_id: int | torch.Tensor
    ) -> torch.Tensor:
        """Cached decode of one token. Returns its next-token logits ``(vocab_size,)``.

        The new token's RoPE position is ``table.length`` — the request's own running length.
        Its K/V is appended to the paged store, then the full history (including this token) is
        gathered and attended through the backend. Advances ``table.length`` by one.
        """
        return self.decode_tokens(cache, table, token_id)[-1]

    @torch.no_grad()
    def decode_many(
        self,
        cache: PagedKVCache,
        tables: list[BlockTable],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Cached decode of one new token for each of ``B`` requests in ONE batched forward.

        The ``B`` new tokens run through the layer stack together (one matmul per projection,
        not ``B``), while each request keeps its **own** RoPE position and its **own** paged
        history. Per-request KV write and history gather are O(B) bookkeeping; the attention
        itself is one batched ragged call. Identical per request to the single-request decode
        path. Returns ``(B, vocab)`` next-token logits and advances each table's length by one.
        """
        if not tables:
            raise ValueError("decode_many needs at least one request")
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).reshape(-1)
        if len(tables) != int(ids.numel()):
            raise ValueError(f"tables/token_ids length mismatch: {len(tables)} vs {ids.numel()}")

        positions = [table.length for table in tables]
        new_lengths = [pos + 1 for pos in positions]
        for table in tables:
            table.reserve(1)
        # COW must run BEFORE plan_read_many, even though write_many re-runs prepare_write per
        # layer: a prefix-shared partial block is copied private here, rebinding table.blocks. The
        # read plan is built once (layer-independent) from those slots, so it has to see the
        # post-COW physical blocks — otherwise it would gather the old shared block while the
        # per-layer write lands in the new private one. Do not fold this pre-pass into write_many.
        for table, pos in zip(tables, positions, strict=True):
            cache.prepare_write(table, pos, 1)
        hidden = self.w["embed_tokens.weight"][ids].to(self.dtype)  # (B, hidden)

        # One RoPE cos/sin row per request, each at the request's own absolute position — from
        # the cached table, already in model dtype, so the per-layer apply_rope cast is a no-op.
        cos, sin = self._rope_rows(positions)
        read_plan = cache.plan_read_many(
            tables, new_lengths, include_pages=self._uses_paged_decode_backend()
        )
        self._prepare_paged_decode(read_plan)
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._decode_attention_batched(
                    x, p, cos, sin, lyr, cache, tables, positions, read_plan
                ),
            )
        for table, new_length in zip(tables, new_lengths, strict=True):
            table.length = new_length

        with self._profile("logits"):
            hidden = rms_norm(hidden, self._norm_weight("norm.weight"), self.rms_eps)
            return self._apply_logit_soft_cap(hidden @ self._lm_head().T)  # (B, vocab)

    def open_decode_window(
        self, cache: PagedKVCache, tables: list[BlockTable], budget: int
    ) -> DecodeWindowPlan | None:
        """Build planned-decode buffers for a stable batch, or ``None`` when not plan-safe.

        Also grows the cached RoPE table to cover every position the window can reach, so
        :meth:`decode_window_step` never allocates or syncs for positions.
        """
        plan = build_decode_window_plan(cache, tables, budget)
        if plan is not None:
            self._ensure_rope_rows(max(plan.base_lengths) + budget)
        return plan

    def enable_decode_graphs(
        self,
        capture_sizes: tuple[int, ...] = DEFAULT_CAPTURE_SIZES,
        *,
        max_position: int = 8192,
        mode: str = "graph",
    ) -> DecodeGraphRunner:
        """Capture the piecewise decode-window graphs now and route window steps through them.

        Captures every bucket up front (so no capture cost can land inside a timed region)
        and returns the runner. Batches above the largest bucket, or windows reaching past
        ``max_position``, fall back to the eager planned path per window. ``mode="eager"``
        skips capture and runs the same segment/buffer flow directly — the CPU test hook.
        """
        self.decode_graphs = DecodeGraphRunner(
            self, capture_sizes=capture_sizes, max_position=max_position, mode=mode
        )
        return self.decode_graphs

    def enable_decode_compile(
        self,
        capture_sizes: tuple[int, ...] = DEFAULT_CAPTURE_SIZES,
        *,
        max_position: int = 8192,
        mode: str | None = "reduce-overhead",
        compile_backend: str = "inductor",
    ) -> CompiledDecodeRunner:
        """Compile the decode-window step now and route window steps through it.

        The torch.compile counterpart to :meth:`enable_decode_graphs` — same slot, same
        window contract, same eager fallbacks. Compiles and warms up every bucket up front
        (parity-checked against the eager planned path), so no compile cost can land inside
        a timed region. ``compile_backend="eager"`` skips Inductor and runs the traced
        graph directly — the CPU test hook.
        """
        from llm_infer.model.decode_compile import CompiledDecodeRunner

        self.decode_graphs = CompiledDecodeRunner(
            self,
            capture_sizes=capture_sizes,
            max_position=max_position,
            mode=mode,
            compile_backend=compile_backend,
        )
        return self.decode_graphs

    @torch.no_grad()
    def decode_window_step(
        self, cache: PagedKVCache, plan: DecodeWindowPlan, token_ids: torch.Tensor
    ) -> torch.Tensor:
        """One planned decode step: :meth:`decode_many` math from preallocated buffers.

        Identical per request to the classic batched decode — same projections, QK-norm, RoPE,
        paged write/gather, packed attention, and logits — differing only in where the
        bookkeeping comes from: write slots, read plan, and RoPE rows are views into the
        window's device buffers instead of per-step Python walks over block tables. Advances
        the plan (and each table's length) by one token.

        With :attr:`decode_graphs` enabled the step replays captured segments instead; the
        returned logits are then a view into a static buffer, valid only until the next step
        (the engine consumes them immediately via argmax).
        """
        if int(token_ids.numel()) != len(plan.tables):
            raise ValueError(
                f"tables/token_ids length mismatch: {len(plan.tables)} vs {token_ids.numel()}"
            )
        if self.decode_graphs is not None:
            logits = self.decode_graphs.window_step(cache, plan, token_ids)
            if logits is not None:
                return logits
        write_slots, read_plan = plan.begin_step(
            include_pages=self._uses_paged_decode_backend()
        )
        self._prepare_paged_decode(read_plan)
        hidden = self.w["embed_tokens.weight"][token_ids.reshape(-1)].to(self.dtype)
        # max_len - 1 is the largest position this step touches; the table already covers the
        # whole window (open_decode_window grew it), so this is a cheap host-side check.
        cos_table, sin_table = self._ensure_rope_rows(plan.max_len)
        cos = cos_table.index_select(0, plan.positions)
        sin = sin_table.index_select(0, plan.positions)
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._decode_attention_planned(
                    x, p, cos, sin, lyr, cache, write_slots, read_plan
                ),
            )
        plan.complete_step()

        with self._profile("logits"):
            hidden = rms_norm(hidden, self._norm_weight("norm.weight"), self.rms_eps)
            return self._apply_logit_soft_cap(hidden @ self._lm_head().T)  # (B, vocab)

    @torch.no_grad()
    def decode_tokens(
        self,
        cache: PagedKVCache,
        table: BlockTable,
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Cached decode of several contiguous tokens for one request.

        Used by speculative verification: the input is ``last_token + draft``. Row ``i`` returns
        next-token logits after token ``i`` has been appended, so draft ids can be checked in one
        forward pass. The caller owns any rollback of ``table.length`` when the draft is rejected.
        """
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).reshape(-1)
        if ids.numel() < 1:
            raise ValueError("decode_tokens needs at least one token")

        start_pos = table.length
        count = int(ids.numel())
        end_pos = start_pos + count
        table.reserve(count)
        hidden = self.w["embed_tokens.weight"][ids].to(self.dtype)

        positions = torch.arange(start_pos, end_pos, dtype=torch.float32, device=self.device)
        cos, sin = self._rope_for_positions(positions)
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._prefill_chunk_attention(
                    x, p, cos, sin, lyr, cache, table, start_pos, end_pos
                ),
            )
        table.length = end_pos

        with self._profile("logits"):
            hidden = rms_norm(hidden, self._norm_weight("norm.weight"), self.rms_eps)
            return self._apply_logit_soft_cap(hidden @ self._lm_head().T)

    def release_table(self, table: BlockTable) -> None:
        """No-op — real K/V lives in the paged cache, not per-table backend state."""
        del table

    def _uses_paged_decode_backend(self) -> bool:
        return self._paged_backend is not None

    def _prepare_paged_decode(self, read_plan: KVReadPlan) -> None:
        backend = self._paged_backend
        if backend is None or read_plan.page_plan is None:
            return
        with self._profile("paged_attention_plan"):
            backend.plan_decode_batch_paged(
                read_plan.page_plan,
                num_qo_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
            )

    def _decode_attention_from_plan(
        self,
        queries: torch.Tensor,
        layer: int,
        cache: PagedKVCache,
        read_plan: KVReadPlan,
    ) -> torch.Tensor:
        backend = self._paged_backend
        if backend is not None and read_plan.page_plan is not None:
            with self._profile("paged_attention"):
                return backend.forward_decode_batch_paged(queries, cache.layer_kv(layer))

        with self._profile("kv_read_gather"):
            k_hist, v_hist = cache.read_many_plan(layer, read_plan)
        k_exp, v_exp = self._expand_kv_token_major(k_hist, v_hist)
        return self.backend.forward_decode_batch_packed(
            queries, k_exp, v_exp, read_plan.cu_seqlens, read_plan.max_len
        )

    def _apply_decoder_layer(
        self, hidden: torch.Tensor, layer: int, attention: _AttentionFn
    ) -> torch.Tensor:
        """One decoder block: pre-norm attention then pre-norm MLP, both with residuals.

        The attention sub-block is supplied as a closure so the full-recompute and cached paths
        share this wrapper while differing only in how attention is run.
        """
        prefix = f"layers.{layer}."
        residual = hidden
        x = rms_norm(hidden, self._norm_weight(prefix + "input_norm.weight"), self.rms_eps)
        hidden = residual + attention(x, prefix, layer)

        residual = hidden
        x = rms_norm(hidden, self._norm_weight(prefix + "post_attention_norm.weight"), self.rms_eps)
        return residual + self._mlp(x, prefix)

    def _attention(
        self, x: torch.Tensor, prefix: str, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Full-recompute attention over the whole sequence (known-good reference path)."""
        q, k, v = self._project_heads(x, prefix)
        q, k = self._qk_norm_rope(q, k, cos, sin, prefix)
        k, v = self._expand_kv(k, v)
        attn = self.backend.forward(q, k, v)
        return self._output_proj(attn, prefix)

    def _prefill_attention(
        self,
        x: torch.Tensor,
        prefix: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: PagedKVCache,
        table: BlockTable,
    ) -> torch.Tensor:
        """Prefill attention: same math as :meth:`_attention`, plus a write of K/V to cache.

        The just-computed K/V (positions ``0 .. L-1``) is exactly what attention needs here, so
        it is used directly; writing it to the paged store seeds the decode steps that follow.
        """
        q, k, v = self._project_heads(x, prefix)
        q, k = self._qk_norm_rope(q, k, cos, sin, prefix)
        # Store pre-GQA K/V as (seq, num_kv_heads, head_dim) at positions 0..seq-1.
        with self._profile("kv_write"):
            cache.write(
                table,
                layer,
                0,
                k.transpose(0, 1).contiguous(),
                v.transpose(0, 1).contiguous(),
            )
        k, v = self._expand_kv(k, v)
        attn = self.backend.forward(q, k, v)
        return self._output_proj(attn, prefix)

    def _prefill_attention_packed(
        self,
        x: torch.Tensor,
        prefix: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: PagedKVCache,
        write_slots: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_len: int,
        backend: PackedPrefillAttentionBackend,
    ) -> torch.Tensor:
        """Packed full-prompt attention with native-GQA cache writes."""
        q, k, v = self._project_heads(x, prefix)
        q, k = self._qk_norm_rope(q, k, cos, sin, prefix)
        k_rows = k.transpose(0, 1).contiguous()
        v_rows = v.transpose(0, 1).contiguous()
        with self._profile("kv_write"):
            cache.write_rows(layer, write_slots, k_rows, v_rows)

        attn = backend.forward_prefill_batch_packed(
            q.transpose(0, 1).contiguous(),
            k_rows,
            v_rows,
            cu_seqlens,
            max_len,
        )
        return self._output_proj(attn.transpose(0, 1).contiguous(), prefix)

    def _prefill_chunk_attention(
        self,
        x: torch.Tensor,
        prefix: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: PagedKVCache,
        table: BlockTable,
        start_pos: int,
        end_pos: int,
    ) -> torch.Tensor:
        """Chunked prefill attention over cached prefix plus the current prompt chunk."""
        q, k, v = self._project_heads(x, prefix)
        q, k = self._qk_norm_rope(q, k, cos, sin, prefix)
        with self._profile("kv_write"):
            cache.write(
                table,
                layer,
                start_pos,
                k.transpose(0, 1).contiguous(),
                v.transpose(0, 1).contiguous(),
            )
        with self._profile("kv_read_gather"):
            k_hist, v_hist = cache.read(table, layer, end_pos)
        k_hist, v_hist = self._expand_kv(k_hist.transpose(0, 1), v_hist.transpose(0, 1))
        attn = self.backend.forward(q, k_hist, v_hist)
        return self._output_proj(attn, prefix)

    def _decode_attention_batched(
        self,
        x: torch.Tensor,
        prefix: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: PagedKVCache,
        tables: list[BlockTable],
        positions: list[int],
        read_plan: KVReadPlan,
    ) -> torch.Tensor:
        """Batched decode attention: same cached decode math as :meth:`decode_tokens`, fused.

        ``x`` is ``(B, hidden)`` — one new token per request. Projection, QK-norm, and RoPE run
        on all ``B`` at once (each row rotated by its own position). Each request's new K/V is
        written to its own paged history and its full history gathered (GQA-expanded) — ragged
        across requests — then one batched attention call returns the ``B`` outputs.
        """
        q, k, v = self._project_heads(x, prefix)
        # q/k/v shapes: (heads, B, hd), (kv_heads, B, hd), (kv_heads, B, hd).
        q, k = self._qk_norm_rope(q, k, cos, sin, prefix)

        with self._profile("kv_write"):
            cache.write_many(
                tables,
                layer,
                positions,
                k.transpose(0, 1).contiguous(),
                v.transpose(0, 1).contiguous(),
            )

        queries = q.transpose(0, 1).contiguous()  # (B, num_heads, head_dim)
        attn = self._decode_attention_from_plan(queries, layer, cache, read_plan)
        return self._output_proj(attn.transpose(0, 1).contiguous(), prefix)  # (B, hidden)

    def _decode_attention_planned(
        self,
        x: torch.Tensor,
        prefix: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: PagedKVCache,
        write_slots: torch.Tensor,
        read_plan: KVReadPlan,
    ) -> torch.Tensor:
        """Batched decode attention fed from planned buffers.

        Same math as :meth:`_decode_attention_batched`; the write lands at precomputed physical
        slots (``write_rows``) and the gather reuses the window's packed read plan, so no block
        table is touched inside the layer loop.
        """
        q, k, v = self._project_heads(x, prefix)
        q, k = self._qk_norm_rope(q, k, cos, sin, prefix)

        with self._profile("kv_write"):
            cache.write_rows(
                layer,
                write_slots,
                k.transpose(0, 1).contiguous(),
                v.transpose(0, 1).contiguous(),
            )

        queries = q.transpose(0, 1).contiguous()  # (B, num_heads, head_dim)
        attn = self._decode_attention_from_plan(queries, layer, cache, read_plan)
        return self._output_proj(attn.transpose(0, 1).contiguous(), prefix)  # (B, hidden)

    def _qk_norm_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        prefix: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional per-head QK-norm, then RoPE — the order the reference oracle uses.

        QK-norm (when configured) is applied before RoPE, so the K written to the paged store is
        post-norm, post-rotation: the rotation is fixed by the token's absolute position and so
        is correct to cache once at write time.
        """
        if self.qk_norm:
            q = rms_norm(q, self._norm_weight(prefix + "attn.q_norm.weight"), self.rms_eps)
            k = rms_norm(k, self._norm_weight(prefix + "attn.k_norm.weight"), self.rms_eps)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)

    def _project_heads(
        self, x: torch.Tensor, prefix: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V projections reshaped to ``(heads, seq, head_dim)`` (KV keeps kv-head count)."""
        seq_len = x.shape[0]
        q = self._linear(x, prefix + "attn.q_proj")
        k = self._linear(x, prefix + "attn.k_proj")
        v = self._linear(x, prefix + "attn.v_proj")
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        return q, k, v

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """GQA: repeat each KV head over its group of query heads (done before the backend)."""
        return expand_grouped_kv(
            k,
            v,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_axis=0,
        )

    def _expand_kv_token_major(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GQA for packed token-major histories: ``(tokens, kv_heads, head_dim)``."""
        return expand_grouped_kv(
            k,
            v,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_axis=1,
        )

    def _output_proj(self, attn: torch.Tensor, prefix: str) -> torch.Tensor:
        """Merge heads ``(heads, seq, head_dim)`` -> ``(seq, hidden)`` and apply o_proj."""
        merged = merge_attention_heads(attn, num_heads=self.num_heads, head_dim=self.head_dim)
        return self._linear(merged, prefix + "attn.o_proj")

    def _mlp(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        return swiglu_mlp(self.w, x, prefix, self.dtype)

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return linear_projection(self.w, x, name, self.dtype)

    def _profile(self, name: str):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.record(name)

    def _norm_weight(self, name: str) -> torch.Tensor:
        """The named RMSNorm weight pre-cast to fp32 (cast once, reused every call)."""
        cached = self._norm_weights_fp32.get(name)
        if cached is None:
            cached = self.w[name].to(torch.float32)
            self._norm_weights_fp32[name] = cached
        return cached

    def _lm_head(self) -> torch.Tensor:
        if self.tie_word_embeddings:
            return self.w["embed_tokens.weight"].to(self.dtype)
        return self.w["lm_head.weight"].to(self.dtype)

    def _rope_tables(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        return self._rope_for_positions(positions)

    def _rope_rows(self, positions: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached RoPE cos/sin rows (model dtype) for host-known integer positions."""
        cos_table, sin_table = self._ensure_rope_rows(max(positions) + 1)
        index = torch.tensor(positions, dtype=torch.long, device=self.device)
        return cos_table.index_select(0, index), sin_table.index_select(0, index)

    def _ensure_rope_rows(self, min_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Grow the cached RoPE table to at least ``min_len`` rows (doubling to limit rebuilds).

        Rows are the full-table form of :meth:`_rope_for_positions` — the same outer-product
        and trig per element, computed once — cast once to the model dtype (the cast
        ``apply_rope`` would otherwise repeat per layer per step).
        """
        cached = 0 if self._rope_rows_cos is None else int(self._rope_rows_cos.shape[0])
        if self._rope_rows_cos is None or self._rope_rows_sin is None or cached < min_len:
            size = max(min_len, 2 * cached, 256)
            cos, sin = self._rope_tables(size)
            self._rope_rows_cos = cos.to(self.dtype)
            self._rope_rows_sin = sin.to(self.dtype)
        return self._rope_rows_cos, self._rope_rows_sin

    def _rope_for_positions(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cos/sin tables for arbitrary absolute positions. Shape ``(len(positions), head_dim)``.

        Decode passes a single per-request position here so each request is rotated by its own
        sequence length, not a batch-row index.
        """
        return rope_tables_for_positions(
            positions, head_dim=self.head_dim, rope_theta=self.rope_theta
        )

    def _validate_token_ids(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        bad = [
            token_id
            for token_id in token_ids
            if isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < self.config.vocab_size
        ]
        if bad:
            raise ValueError(
                f"token_ids must be ints in [0, {self.config.vocab_size}); got {bad[:3]}"
            )

    def _apply_logit_soft_cap(self, logits: torch.Tensor) -> torch.Tensor:
        cap = self.config.logit_soft_cap
        if cap is None or cap <= 0.0:
            return logits
        return cap * torch.tanh(logits / cap)
