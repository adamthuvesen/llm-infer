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

    @classmethod
    def load(
        cls,
        *,
        dtype: torch.dtype = torch.float32,
        backend: AttentionBackend | None = None,
        model_id: str = MODEL_ID,
        revision: str = MODEL_REVISION,
    ) -> QwenModel:
        """Load the pinned model's weights and config from the HF cache.

        Defaults to fp32 (where greedy tie-breaks near-vanish) and the
        ``torch_naive`` reference backend.
        """
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        hf = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype)
        hf.eval()
        weights = {name: tensor.detach() for name, tensor in hf.state_dict().items()}
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
        ids = torch.tensor(token_ids, dtype=torch.long)
        hidden = self.w["model.embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_tables(len(token_ids))
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden, layer, lambda x, p, _lyr: self._attention(x, p, cos, sin)
            )

        hidden = _rms_norm(hidden, self.w["model.norm.weight"], self.rms_eps)
        return hidden @ self._lm_head().T

    @torch.no_grad()
    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        """Cached prefill: run the prompt, store all K/V, return last-position logits.

        Hidden states match :meth:`logits` bit-for-bit (the cache write is a side effect
        that does not touch the values the backend sees); only the last row's logits are
        formed, since greedy needs only the first generated token. Sets ``table.length``.
        """
        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        seq_len = len(prompt_ids)
        table.reserve(seq_len)
        ids = torch.tensor(prompt_ids, dtype=torch.long)
        hidden = self.w["model.embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_tables(seq_len)
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._prefill_attention(x, p, cos, sin, lyr, cache, table),
            )
        table.length = seq_len

        last = _rms_norm(hidden[-1:], self.w["model.norm.weight"], self.rms_eps)
        return (last @ self._lm_head().T)[-1]

    @torch.no_grad()
    def decode_one(self, cache: PagedKVCache, table: BlockTable, token_id: int) -> torch.Tensor:
        """Cached decode of one token. Returns its next-token logits ``(vocab_size,)``.

        The new token's RoPE position is ``table.length`` — the request's own running
        length, never a batch-row index. Its K/V is appended to the paged store, then
        the full history (including this token) is gathered and attended through the
        backend. Advances ``table.length`` by one.
        """
        pos = table.length
        table.reserve(1)
        new_length = pos + 1
        ids = torch.tensor([token_id], dtype=torch.long)
        hidden = self.w["model.embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_for_positions(torch.tensor([pos], dtype=torch.float32))
        for layer in range(self.num_layers):
            hidden = self._apply_decoder_layer(
                hidden,
                layer,
                lambda x, p, lyr: self._decode_attention(
                    x, p, cos, sin, lyr, cache, table, pos, new_length
                ),
            )
        table.length = new_length

        hidden = _rms_norm(hidden, self.w["model.norm.weight"], self.rms_eps)
        return (hidden @ self._lm_head().T)[-1]

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
        return residual + self._mlp(x, p)

    def _attention(
        self, x: torch.Tensor, p: str, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Full-recompute attention over the whole sequence (Phase A path)."""
        q, k, v = self._project_heads(x, p)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        k, v = self._expand_kv(k, v)
        attn = self.backend.forward(q, k, v)
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
        q, k, v = self._project_heads(x, p)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # Store pre-GQA K/V as (seq, num_kv_heads, head_dim) at positions 0..seq-1.
        cache.write(table, layer, 0, k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous())
        k, v = self._expand_kv(k, v)
        attn = self.backend.forward(q, k, v)
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
        q, k, v = self._project_heads(x, p)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        cache.write(
            table, layer, pos, k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()
        )
        k_hist, v_hist = cache.read(table, layer, length)  # (length, num_kv_heads, head_dim)
        k_hist, v_hist = self._expand_kv(k_hist.transpose(0, 1), v_hist.transpose(0, 1))
        attn = self.backend.forward(q, k_hist, v_hist)
        return self._output_proj(attn, p)

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
        return self._rope_for_positions(torch.arange(seq_len, dtype=torch.float32))

    def _rope_for_positions(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cos/sin tables for arbitrary absolute positions. Shape ``(len(positions), head_dim)``.

        Decode passes a single per-request position here so each request is rotated by
        its own sequence length, not a batch-row index.
        """
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
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
