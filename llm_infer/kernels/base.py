"""The narrow attention interface every backend implements.

A backend computes scaled dot-product attention for one request: given the query
rows for the current step (prefill: all prompt positions; decode: the single new
position) and the full key/value history for that request, it returns the attention
output. Position encoding (RoPE) and GQA head expansion happen *before* the backend
is called — keys and values arrive already rotated and already repeated to the query
head count — so a backend only owns the `softmax(QKᵀ / √d) V` causal core.

Keeping the interface this narrow is the point: each backend is a drop-in swap validated
against ``torch_naive`` by the reference check, not a rewrite of the model forward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from llm_infer.kv_cache.paged_kv_cache import KVPagePlan


@runtime_checkable
class AttentionBackend(Protocol):
    """Causal scaled-dot-product attention for a single request.

    Shapes (single request, no batch dimension):
        query: ``(num_heads, q_len, head_dim)`` — q_len is the prompt length on
            prefill and 1 on each decode step.
        key, value: ``(num_heads, kv_len, head_dim)`` — the full history including
            the current step's positions. Already RoPE-rotated and already expanded
            from KV heads to ``num_heads`` (GQA repeat done by the caller).

    Returns:
        ``(num_heads, q_len, head_dim)`` attention output, ready for the output
        projection.

    Causality: query position ``i`` (absolute index ``kv_len - q_len + i``) may
    attend to key positions ``0 .. kv_len - q_len + i`` inclusive.
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor: ...

    def forward_decode_batch_packed(
        self,
        queries: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Batched single-token decode over token-major packed K/V histories.

        Every running request advances by exactly one token, but each has its own ragged KV
        history, so the requests cannot be stacked into a dense tensor. ``queries`` is
        ``(B, num_heads, head_dim)`` — one decode query per request; ``key``/``value`` are
        ``(total_kv_tokens, num_heads, head_dim)`` and ``cu_seqlens_k`` indexes each request's
        history (already RoPE'd and GQA-expanded). The reference splits these per request and
        loops; the fast backend runs one ragged kernel over the pack.

        Returns ``(B, num_heads, head_dim)`` — row ``i`` is request ``i``'s output, identical
        to attending ``queries[i]`` over its own history alone.
        """
        ...


@runtime_checkable
class PagedDecodeAttentionBackend(AttentionBackend, Protocol):
    """Optional decode backend that consumes the cache's page table directly."""

    def plan_decode_batch_paged(
        self,
        page_plan: KVPagePlan,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """Prepare layer-reusable metadata for one batched decode step."""
        ...

    def forward_decode_batch_paged(
        self, queries: torch.Tensor, paged_kv_cache: torch.Tensor
    ) -> torch.Tensor:
        """Batched single-token decode over a layer's native paged KV cache."""
        ...
