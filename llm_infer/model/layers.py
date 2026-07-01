"""Shared transformer-layer primitives used by model backends."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def linear_projection(
    weights: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    name: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply ``x @ W.T`` and an optional bias from ``weights``."""
    out = x @ weights[name + ".weight"].to(dtype).T
    bias = weights.get(name + ".bias")
    if bias is not None:
        out = out + bias.to(dtype)
    return out


def swiglu_mlp(
    weights: Mapping[str, torch.Tensor],
    x: torch.Tensor,
    prefix: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """SwiGLU feed-forward block shared by Qwen and dense bundle exports."""
    gate = linear_projection(weights, x, prefix + "mlp.gate_proj", dtype)
    up = linear_projection(weights, x, prefix + "mlp.up_proj", dtype)
    hidden = torch.nn.functional.silu(gate) * up
    return linear_projection(weights, hidden, prefix + "mlp.down_proj", dtype)


def expand_grouped_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    num_heads: int,
    num_kv_heads: int,
    head_axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat KV heads over query-head groups before attention."""
    repeat = num_heads // num_kv_heads
    return k.repeat_interleave(repeat, dim=head_axis), v.repeat_interleave(repeat, dim=head_axis)


def merge_attention_heads(
    attn: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Merge ``(heads, seq, head_dim)`` attention output into ``(seq, hidden)``."""
    seq_len = attn.shape[1]
    return attn.transpose(0, 1).reshape(seq_len, num_heads * head_dim)


def rope_tables_for_positions(
    positions: torch.Tensor,
    *,
    head_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE cos/sin tables for arbitrary absolute positions."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, half, dtype=torch.float32, device=positions.device) / half)
    )
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()
