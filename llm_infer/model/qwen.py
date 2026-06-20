"""A from-scratch Qwen2 forward pass that routes attention through an AttentionBackend.

The weights are HuggingFace's (loaded at the pinned revision); the forward is ours,
so the correctness oracle is testing *our* engine — RoPE, GQA expansion, RMSNorm,
the layer stack, and the decode loop — against HF greedy, not HF against itself. The
attention core is delegated to the pluggable backend so a later paged/flash kernel
is a swap validated by the same oracle.

Single request only (no batch dimension), full recompute over the whole sequence on
every call. A KV-cache lives in Phase B; Phase A keeps the forward dead simple and
leans on the oracle to prove it exact.
"""

from __future__ import annotations

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from llm_infer.kernels.base import AttentionBackend
from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.model.config import MODEL_ID, MODEL_REVISION


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
        """Next-token logits for every position. Shape ``(seq_len, vocab_size)``."""
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        ids = torch.tensor(token_ids, dtype=torch.long)
        hidden = self.w["model.embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_tables(len(token_ids))
        for layer in range(self.num_layers):
            hidden = self._decoder_layer(hidden, layer, cos, sin)

        hidden = _rms_norm(hidden, self.w["model.norm.weight"], self.rms_eps)
        lm_head = (
            self.w["model.embed_tokens.weight"]
            if self.tie_word_embeddings
            else self.w["lm_head.weight"]
        )
        return hidden @ lm_head.to(self.dtype).T

    def _decoder_layer(
        self, hidden: torch.Tensor, layer: int, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        p = f"model.layers.{layer}."
        residual = hidden
        x = _rms_norm(hidden, self.w[p + "input_layernorm.weight"], self.rms_eps)
        hidden = residual + self._attention(x, p, cos, sin)

        residual = hidden
        x = _rms_norm(hidden, self.w[p + "post_attention_layernorm.weight"], self.rms_eps)
        return residual + self._mlp(x, p)

    def _attention(
        self, x: torch.Tensor, p: str, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        seq_len = x.shape[0]

        q = self._linear(x, p + "self_attn.q_proj")
        k = self._linear(x, p + "self_attn.k_proj")
        v = self._linear(x, p + "self_attn.v_proj")

        # (seq, heads, head_dim) -> (heads, seq, head_dim)
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)

        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # GQA: repeat each KV head to cover its group of query heads (done here so the
        # backend only sees a uniform head count, per the AttentionBackend contract).
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=0)
        v = v.repeat_interleave(repeat, dim=0)

        attn = self.backend.forward(q, k, v)  # (heads, seq, head_dim)
        attn = attn.transpose(0, 1).reshape(seq_len, self.num_heads * self.head_dim)
        return self._linear(attn, p + "self_attn.o_proj")

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

    def _rope_tables(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Cos/sin tables for positions ``0 .. seq_len-1``. Shape ``(seq_len, head_dim)``."""
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, half)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
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
