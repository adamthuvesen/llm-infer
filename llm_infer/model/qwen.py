"""A from-scratch Qwen2 forward pass that routes attention through an AttentionBackend.

The weights are HuggingFace's (loaded at the pinned revision); the forward is ours,
so the correctness oracle is testing *our* engine — RoPE, GQA expansion, RMSNorm,
the layer stack, and the decode loop — against HF greedy, not HF against itself. The
attention core is delegated to the pluggable backend so a later paged/flash kernel
is a swap validated by the same oracle.

Two decode paths share one layer stack:

* :meth:`logits` — full recompute over the whole sequence, no cache. The Phase A
  reference and the oracle's path; left bit-for-bit intact.
* :meth:`prefill` / :meth:`decode_one` — the Phase B paged/cached path. Prefill writes
  every prompt position's K/V into the paged store; each decode step computes only the
  new token, appends its K/V, and attends against the gathered history through the
  *same* ``torch_naive`` backend (gathered K/V, materialized softmax — no fast kernel).
  RoPE positions advance **per request** (each request's own length), never a batch row.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from llm_infer.kernels.base import AttentionBackend
from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.config import MODEL_ID, MODEL_REVISION
from llm_infer.profiling import TimingProfiler

# The closure each decoder layer calls for its attention block: (x, prefix, layer) -> out.
_AttentionFn = Callable[[torch.Tensor, str, int], torch.Tensor]


class QwenModel:
    """Qwen2.5-Coder-3B forward pass over a single token sequence.

    Holds the HF weight tensors and config; ``logits`` runs the full network for a
    sequence of token ids and returns the next-token logits at every position.
    """

    def __init__(
        self,
        weights: dict[str, torch.Tensor],
        config: object,
        backend: AttentionBackend,
        dtype: torch.dtype,
    ) -> None:
        self.w = weights
        self.backend = backend
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rms_eps = config.rms_norm_eps
        self.rope_theta = _rope_theta(config)
        self.tie_word_embeddings = config.tie_word_embeddings
        # Per-step helper tensors (token ids, RoPE positions) are built on the weight
        # device so a GPU-resident model never silently mixes CPU and CUDA tensors.
        self.device = self.w["model.embed_tokens.weight"].device
        self.profiler: TimingProfiler | None = None

    @classmethod
    def load(
        cls,
        *,
        dtype: torch.dtype = torch.float32,
        backend: AttentionBackend | None = None,
        device: torch.device | str = "cpu",
        model_id: str = MODEL_ID,
        revision: str | None = MODEL_REVISION,
    ) -> QwenModel:
        """Load the model's weights and config from the HF cache (or a local path).

        Defaults to the pinned base on fp32 CPU (where greedy tie-breaks near-vanish) and the
        ``torch_naive`` reference backend. Pass ``device="cuda"`` to place the weights on
        the GPU for the flash-attn backend; the CPU default keeps the reference path
        bit-identical to Phase A/B. ``revision`` is ``None`` for a local ``model_id`` path
        (the Phase E merged grpo-s0 weights), which carries no git revision.
        """
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        hf = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
        hf.eval()
        weights = {
            name: tensor.detach().to(device) for name, tensor in hf.state_dict().items()
        }
        return cls(
            weights=weights,
            config=config,
            backend=backend or TorchNaiveAttention(),
            dtype=dtype,
        )

    @torch.no_grad()
    def logits(self, token_ids: list[int]) -> torch.Tensor:
        """Next-token logits for every position. Shape ``(seq_len, vocab_size)``.

        Full recompute over the whole sequence — no cache. The Phase A reference path.
        """
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        hidden = self.w["model.embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_tables(len(token_ids))
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden, layer, lambda x, p, _lyr: self._attention(x, p, cos, sin)
            )

        with self._profile("logits"):
            hidden = _rms_norm(hidden, self.w["model.norm.weight"], self.rms_eps)
            return hidden @ self._lm_head().T

    @torch.no_grad()
    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        """Cached prefill: run the prompt, store all K/V, return last-position logits.

        Hidden states match :meth:`logits` to within ~1e-5 (the cache write is a side
        effect that does not touch the values the backend sees; the only non-bit-exact op
        is a BLAS reduction-order difference, far below the argmax-flip threshold). Only
        the last row's logits are formed, since greedy needs only the first generated
        token. Sets ``table.length``.
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        seq_len = len(prompt_ids)
        table.reserve(seq_len)
        ids = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        hidden = self.w["model.embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_tables(seq_len)
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._prefill_attention(x, p, cos, sin, lyr, cache, table),
            )
        table.length = seq_len

        with self._profile("logits"):
            last = _rms_norm(hidden[-1:], self.w["model.norm.weight"], self.rms_eps)
            return (last @ self._lm_head().T)[-1]

    @torch.no_grad()
    def decode_one(
        self, cache: PagedKVCache, table: BlockTable, token_id: int | torch.Tensor
    ) -> torch.Tensor:
        """Cached decode of one token. Returns its next-token logits ``(vocab_size,)``.

        The new token's RoPE position is ``table.length`` — the request's own running
        length, never a batch-row index. Its K/V is appended to the paged store, then
        the full history (including this token) is gathered and attended through the
        backend. Advances ``table.length`` by one.
        """
        pos = table.length
        table.reserve(1)
        new_length = pos + 1
        ids = torch.as_tensor(token_id, dtype=torch.long, device=self.device).reshape(1)
        hidden = self.w["model.embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_for_positions(
            torch.tensor([pos], dtype=torch.float32, device=self.device)
        )
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._decode_attention(
                    x, p, cos, sin, lyr, cache, table, pos, new_length
                ),
            )
        table.length = new_length

        with self._profile("logits"):
            hidden = _rms_norm(hidden, self.w["model.norm.weight"], self.rms_eps)
            return (hidden @ self._lm_head().T)[-1]

    @torch.no_grad()
    def decode_many(
        self,
        cache: PagedKVCache,
        tables: list[BlockTable],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Cached decode of one new token for each of ``B`` requests in ONE batched forward.

        The batched counterpart of :meth:`decode_one`: the ``B`` new tokens run through the
        layer stack together (one matmul per projection, not ``B``), while each request keeps
        its **own** RoPE position and its **own** paged history — the new token's position is
        its table's running length, never a batch-row index. Per-request KV write and history
        gather are O(B) bookkeeping; the attention itself is one fused ragged call. Returns
        ``(B, vocab)`` next-token logits and advances each table's length by one.
        """
        if not tables:
            raise ValueError("decode_many needs at least one request")
        if not (len(tables) == len(token_ids)):
            raise ValueError(f"tables/token_ids length mismatch: {len(tables)} vs {len(token_ids)}")

        positions = [table.length for table in tables]
        new_lengths = [pos + 1 for pos in positions]
        for table in tables:
            table.reserve(1)
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        hidden = self.w["model.embed_tokens.weight"][ids].to(self.dtype)  # (B, hidden)

        # One RoPE cos/sin row per request, each at the request's own absolute position.
        cos, sin = self._rope_for_positions(
            torch.tensor(positions, dtype=torch.float32, device=self.device)
        )
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._decode_attention_batched(
                    x, p, cos, sin, lyr, cache, tables, positions, new_lengths
                ),
            )
        for table, new_length in zip(tables, new_lengths, strict=True):
            table.length = new_length

        with self._profile("logits"):
            hidden = _rms_norm(hidden, self.w["model.norm.weight"], self.rms_eps)
            return hidden @ self._lm_head().T  # (B, vocab)

    def _apply_decoder_layer(
        self, hidden: torch.Tensor, layer: int, attention: _AttentionFn
    ) -> torch.Tensor:
        """One decoder block: pre-norm attention then pre-norm MLP, both with residuals.

        The attention sub-block is supplied as a closure so the full-recompute and
        cached paths share this wrapper while differing only in how attention is run.
        """
        p = f"model.layers.{layer}."
        residual = hidden
        x = _rms_norm(hidden, self.w[p + "input_layernorm.weight"], self.rms_eps)
        hidden = residual + attention(x, p, layer)

        residual = hidden
        x = _rms_norm(hidden, self.w[p + "post_attention_layernorm.weight"], self.rms_eps)
        with self._profile("projections_mlp"):
            return residual + self._mlp(x, p)

    def _attention(
        self, x: torch.Tensor, p: str, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Full-recompute attention over the whole sequence (Phase A path)."""
        with self._profile("projections_mlp"):
            q, k, v = self._project_heads(x, p)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        with self._profile("gqa_expand"):
            k, v = self._expand_kv(k, v)
        with self._profile("attention"):
            attn = self.backend.forward(q, k, v)
        with self._profile("projections_mlp"):
            return self._output_proj(attn, p)

    def _prefill_attention(
        self,
        x: torch.Tensor,
        p: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: PagedKVCache,
        table: BlockTable,
    ) -> torch.Tensor:
        """Prefill attention: same math as :meth:`_attention`, plus a write of K/V to cache.

        The just-computed K/V (positions ``0 .. L-1``) is exactly what attention needs
        here, so it is used directly; writing it to the paged store seeds the decode
        steps that follow.
        """
        with self._profile("projections_mlp"):
            q, k, v = self._project_heads(x, p)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # Store pre-GQA K/V as (seq, num_kv_heads, head_dim) at positions 0..seq-1.
        with self._profile("kv_write"):
            cache.write(
                table, layer, 0, k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()
            )
        with self._profile("gqa_expand"):
            k, v = self._expand_kv(k, v)
        with self._profile("attention"):
            attn = self.backend.forward(q, k, v)
        with self._profile("projections_mlp"):
            return self._output_proj(attn, p)

    def _decode_attention(
        self,
        x: torch.Tensor,
        p: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: PagedKVCache,
        table: BlockTable,
        pos: int,
        length: int,
    ) -> torch.Tensor:
        """Decode attention: append the new token's K/V, gather history, attend.

        ``x`` is one row (the new token). Its K/V is written at position ``pos``; the
        gathered history covers positions ``0 .. length-1`` (``length == pos + 1``,
        including this token), so the single query attends over the whole prefix.
        """
        with self._profile("projections_mlp"):
            q, k, v = self._project_heads(x, p)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        with self._profile("kv_write"):
            cache.write(
                table, layer, pos, k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()
            )
        with self._profile("kv_read_gather"):
            k_hist, v_hist = cache.read(table, layer, length)  # (length, num_kv_heads, head_dim)
        with self._profile("gqa_expand"):
            k_hist, v_hist = self._expand_kv(k_hist.transpose(0, 1), v_hist.transpose(0, 1))
        with self._profile("attention"):
            attn = self.backend.forward(q, k_hist, v_hist)
        with self._profile("projections_mlp"):
            return self._output_proj(attn, p)

    def _decode_attention_batched(
        self,
        x: torch.Tensor,
        p: str,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer: int,
        cache: PagedKVCache,
        tables: list[BlockTable],
        positions: list[int],
        new_lengths: list[int],
    ) -> torch.Tensor:
        """Batched decode attention: same per-request math as :meth:`_decode_attention`, fused.

        ``x`` is ``(B, hidden)`` — one new token per request. Projection and RoPE run on all
        ``B`` at once (each row rotated by its own position via the per-request ``cos``/``sin``).
        Each request's new K/V is written to its own paged history and its full history gathered
        (GQA-expanded) — ragged across requests — then one batched attention call returns the
        ``B`` outputs. Identical per request to the single-request decode path.
        """
        with self._profile("projections_mlp"):
            q, k, v = self._project_heads(x, p)
        # q/k/v shapes: (heads, B, hd), (kv_heads, B, hd), (kv_heads, B, hd).
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        with self._profile("kv_write"):
            cache.write_many(
                tables,
                layer,
                positions,
                k.transpose(0, 1).contiguous(),
                v.transpose(0, 1).contiguous(),
            )

        queries = q.transpose(0, 1).contiguous()  # (B, num_heads, head_dim)
        with self._profile("kv_read_gather"):
            k_hist, v_hist, cu_seqlens_k, max_seqlen_k = cache.read_many(tables, layer, new_lengths)
        with self._profile("gqa_expand"):
            k_exp, v_exp = self._expand_kv_token_major(k_hist, v_hist)

        with self._profile("attention"):
            packed_forward = getattr(self.backend, "forward_decode_batch_packed", None)
            if packed_forward is not None:
                attn = packed_forward(queries, k_exp, v_exp, cu_seqlens_k, max_seqlen_k)
            else:
                keys: list[torch.Tensor] = []
                values: list[torch.Tensor] = []
                for k_chunk, v_chunk in zip(
                    k_exp.split(new_lengths), v_exp.split(new_lengths), strict=True
                ):
                    keys.append(k_chunk.transpose(0, 1).contiguous())
                    values.append(v_chunk.transpose(0, 1).contiguous())
                attn = self.backend.forward_decode_batch(queries, keys, values)
        with self._profile("projections_mlp"):
            return self._output_proj(attn.transpose(0, 1).contiguous(), p)  # (B, hidden)

    def _project_heads(
        self, x: torch.Tensor, p: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Q/K/V projections reshaped to ``(heads, seq, head_dim)`` (KV keeps kv-head count)."""
        seq_len = x.shape[0]
        q = self._linear(x, p + "self_attn.q_proj")
        k = self._linear(x, p + "self_attn.k_proj")
        v = self._linear(x, p + "self_attn.v_proj")
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        return q, k, v

    def _expand_kv(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GQA: repeat each KV head over its group of query heads (done before the backend)."""
        repeat = self.num_heads // self.num_kv_heads
        return k.repeat_interleave(repeat, dim=0), v.repeat_interleave(repeat, dim=0)

    def _expand_kv_token_major(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GQA for packed token-major histories: ``(tokens, kv_heads, head_dim)``."""
        repeat = self.num_heads // self.num_kv_heads
        return k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)

    def _output_proj(self, attn: torch.Tensor, p: str) -> torch.Tensor:
        """Merge heads ``(heads, seq, head_dim)`` -> ``(seq, hidden)`` and apply o_proj."""
        seq_len = attn.shape[1]
        merged = attn.transpose(0, 1).reshape(seq_len, self.num_heads * self.head_dim)
        return self._linear(merged, p + "self_attn.o_proj")

    def _mlp(self, x: torch.Tensor, p: str) -> torch.Tensor:
        gate = self._linear(x, p + "mlp.gate_proj")
        up = self._linear(x, p + "mlp.up_proj")
        return self._linear(torch.nn.functional.silu(gate) * up, p + "mlp.down_proj")

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """``x @ Wᵀ (+ b)`` for the weight (and optional bias) stored under ``name``.

        Qwen2 carries a bias on the attention q/k/v projections and none elsewhere; the
        bias is applied only when the weight dict actually holds one.
        """
        d = self.dtype
        out = x @ self.w[name + ".weight"].to(d).T
        bias = self.w.get(name + ".bias")
        if bias is not None:
            out = out + bias.to(d)
        return out

    def _profile(self, name: str):
        if self.profiler is None:
            return _NullTimer()
        return self.profiler.record(name)

    def _lm_head(self) -> torch.Tensor:
        """The output-projection weight (tied to the embedding when configured)."""
        weight = (
            self.w["model.embed_tokens.weight"]
            if self.tie_word_embeddings
            else self.w["lm_head.weight"]
        )
        return weight.to(self.dtype)

    def _rope_tables(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Cos/sin tables for positions ``0 .. seq_len-1``. Shape ``(seq_len, head_dim)``."""
        return self._rope_for_positions(
            torch.arange(seq_len, dtype=torch.float32, device=self.device)
        )

    def _rope_for_positions(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cos/sin tables for arbitrary absolute positions. Shape ``(len(positions), head_dim)``.

        Decode passes a single per-request position here so each request is rotated by
        its own sequence length, not a batch-row index.
        """
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, half, dtype=torch.float32, device=self.device) / half)
        )
        freqs = torch.outer(positions, inv_freq)  # (len, half)
        emb = torch.cat([freqs, freqs], dim=-1)  # (len, head_dim)
        return emb.cos(), emb.sin()


def _rope_theta(config: object) -> float:
    """Read the RoPE base, tolerating the transformers 4.x flat attr and the 5.x nested dict.

    The pinned Qwen2.5-Coder config uses the ``default`` rope type (no scaling), so we
    only need the base; reject anything else loudly rather than silently mis-rotating.
    """
    params = getattr(config, "rope_parameters", None)
    if params is not None:
        rope_type = params.get("rope_type", "default")
        if rope_type != "default":
            raise ValueError(f"unsupported rope_type {rope_type!r}; Phase A handles 'default' only")
        return float(params["rope_theta"])
    return float(config.rope_theta)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32, matching Qwen2 (normalize in fp32, then cast back, then scale)."""
    dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * xf).to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding, the HF Qwen2 (rotate-half) layout.

    ``x`` is ``(heads, seq, head_dim)``; ``cos``/``sin`` are ``(seq, head_dim)`` and
    broadcast across heads.
    """
    cos = cos.to(x.dtype).unsqueeze(0)
    sin = sin.to(x.dtype).unsqueeze(0)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class _NullTimer:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: object) -> None:
        return None
