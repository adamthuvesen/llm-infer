"""The narrow attention interface every backend implements.

A backend computes scaled dot-product attention for one request: given the query
rows for the current step (prefill: all prompt positions; decode: the single new
position) and the full key/value history for that request, it returns the attention
output. Position encoding (RoPE) and GQA head expansion happen *before* the narrow
single-request backend is called. The optional packed-prefill interface keeps K/V at
their native GQA head count so fused kernels do not materialize repeated heads.

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
class PackedPrefillAttentionBackend(AttentionBackend, Protocol):
    """Optional causal attention over a packed batch of complete prompts.

    All tensors are token-major. ``query`` is
    ``(total_tokens, num_qo_heads, head_dim)`` while ``key`` and ``value`` are
    ``(total_tokens, num_kv_heads, head_dim)``. Native GQA is allowed when
    ``num_qo_heads`` is a multiple of ``num_kv_heads``. ``cu_seqlens`` indexes the
    same request boundaries for Q, K, and V, so no request may attend across a
    boundary.

    Returns ``(total_tokens, num_qo_heads, head_dim)`` in the query's dtype.
    """

    def forward_prefill_batch_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor: ...


@runtime_checkable
class GroupedDecodeAttentionBackend(AttentionBackend, Protocol):
    """Optional batched decode that expands GQA KV heads inside the kernel.

    The base :meth:`~AttentionBackend.forward_decode_batch_packed` contract takes K/V already
    repeated to ``num_qo_heads``; a caller with grouped-query attention must therefore
    materialize the whole history at query-head count every step. A backend that broadcasts KV
    heads natively (SDPA's ``enable_gqa``) implements this method instead, so the caller passes
    the packed history at its native ``num_kv_heads`` and skips that per-step expansion.
    """

    def forward_decode_batch_packed_grouped(
        self,
        queries: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Batched single-token decode over native-KV-head packed histories.

        ``queries`` is ``(B, num_qo_heads, head_dim)``; ``key``/``value`` are
        ``(total_kv_tokens, num_kv_heads, head_dim)`` with ``num_qo_heads`` a multiple of
        ``num_kv_heads`` — the kernel repeats each KV head over its query-head group. Returns
        ``(B, num_qo_heads, head_dim)``, identical to expanding the KV heads first and calling
        :meth:`~AttentionBackend.forward_decode_batch_packed`.
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


def validate_packed_prefill_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
) -> None:
    """Check packed-prefill tensor structure without reading device-side offsets."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError(
            "packed prefill query/key/value must all be rank 3; "
            f"got {query.ndim}, {key.ndim}, {value.ndim}"
        )
    if key.shape != value.shape:
        raise ValueError(
            f"packed prefill key/value shapes must match; got {tuple(key.shape)} and "
            f"{tuple(value.shape)}"
        )
    if query.shape[0] != key.shape[0]:
        raise ValueError(
            "packed prefill query/key/value token counts must match; "
            f"got query {query.shape[0]}, key/value {key.shape[0]}"
        )
    if query.shape[2] != key.shape[2]:
        raise ValueError(
            "packed prefill query/key/value head dimensions must match; "
            f"got query {query.shape[2]}, key/value {key.shape[2]}"
        )
    num_qo_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_qo_heads == 0 or num_kv_heads == 0 or num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "packed prefill query head count must be divisible by the KV head count; "
            f"got query {num_qo_heads}, KV {num_kv_heads}"
        )
    if query.device != key.device or query.device != value.device:
        raise ValueError(
            "packed prefill query/key/value must be on the same device; "
            f"got {query.device}, {key.device}, {value.device}"
        )
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
        raise ValueError(
            "packed prefill cu_seqlens must be rank 1 with at least two offsets; "
            f"got shape {tuple(cu_seqlens.shape)}"
        )
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"packed prefill cu_seqlens must have dtype int32; got {cu_seqlens.dtype}")
    if cu_seqlens.device != query.device:
        raise ValueError(
            "packed prefill cu_seqlens must be on the same device as query/key/value; "
            f"got {cu_seqlens.device} and {query.device}"
        )
    if max_seqlen <= 0:
        raise ValueError(f"packed prefill max_seqlen must be positive; got {max_seqlen}")


def packed_prefill_lengths(
    cu_seqlens: torch.Tensor,
    *,
    total_tokens: int,
    max_seqlen: int,
) -> list[int]:
    """Read and check packed request boundaries, returning their sequence lengths."""
    offsets = cu_seqlens.tolist()
    if offsets[0] != 0:
        raise ValueError(f"packed prefill first offset must be 0; got {offsets[0]}")
    if offsets[-1] != total_tokens:
        raise ValueError(
            "packed prefill final offset must equal the token count; "
            f"got {offsets[-1]} and {total_tokens}"
        )
    lengths = [end - start for start, end in zip(offsets, offsets[1:], strict=False)]
    if any(length <= 0 for length in lengths):
        raise ValueError(
            "packed prefill sequence lengths must be positive and offsets strictly increasing; "
            f"got {lengths}"
        )
    actual_max = max(lengths)
    if max_seqlen != actual_max:
        raise ValueError(
            f"packed prefill max_seqlen must equal the longest sequence ({actual_max}); "
            f"got {max_seqlen}"
        )
    return lengths
