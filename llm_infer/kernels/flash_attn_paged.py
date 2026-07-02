"""The fast attention backend: flash-attn's fused varlen kernel behind the protocol.

This is the flash-attn speed swap. It implements the same
:class:`~llm_infer.kernels.base.AttentionBackend` protocol as ``torch_naive`` — same
per-request, already-RoPE'd, already-GQA-expanded ``(num_heads, q_len, head_dim)``
tensors in, same attention output out — so the model forward, scheduler, and paged
KV-cache do not change at all. The engine still owns paging and gathers each request's
K/V history before calling here; this backend only swaps the ``softmax(QKᵀ/√d) V`` core
for flash-attn's fused kernel.

Two layout/dtype facts bridge the protocol to flash-attn:

* **Layout.** The protocol is head-major ``(num_heads, seq, head_dim)``; flash-attn's
  varlen API is token-major ``(total_tokens, num_heads, head_dim)``. We transpose in
  and out. One request at a time means the varlen packing is a single sequence, so
  ``cu_seqlens`` is just ``[0, len]``.
* **dtype.** flash-attn runs in fp16/bf16, not fp32. The model on the target GPU loads
  in bf16, so queries already arrive bf16 and we run the kernel in bf16. If queries
  arrive fp32 (e.g. the fp32 reference check dtype), we run in bf16 and cast the output back —
  the resulting fp32-reduction-order difference vs ``torch_naive`` is exactly the
  genuine-tie effect the reference check's tie-tolerance rule accounts for (docs/fixture-format.md).

Causality matches the protocol contract under ``causal=True``: flash-attn aligns the
query block to the *bottom-right* of the key block, so query position ``i`` attends to
key positions ``0 .. kv_len - q_len + i`` — lower-triangular on prefill
(``q_len == kv_len``) and full-history on a decode step (``q_len == 1``).
"""

from __future__ import annotations

import math

import torch

# flash-attn only builds/imports on CUDA. Importing this module on a CPU-only host (the
# dev Mac) must not explode at import time — only at construction, with a clear message.
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:  # pragma: no cover - exercised only on the CPU dev host
    flash_attn_varlen_func = None

# The kernel runs in bf16; fp32 query input is downcast for the kernel and the output
# is returned in the query's original dtype.
_KERNEL_DTYPE = torch.bfloat16


class FlashAttnPagedAttention:
    """Fused causal attention for one request via flash-attn's varlen kernel.

    Implements the :class:`~llm_infer.kernels.base.AttentionBackend` protocol. A
    drop-in swap for ``torch_naive``: the engine gathers the request's full K/V history
    and GQA-expands it before calling, identical to the reference path.
    """

    def __init__(self) -> None:
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "flash-attn is not available; FlashAttnPagedAttention needs a CUDA build "
                "of flash-attn. Run it on the target GPU (Modal A100), not the CPU host."
            )
        # Decode-batch cu_seqlens_q is always 0..B — cache the arange instead of launching a
        # fresh one per layer per decode step. Sliced views serve any batch up to the cached
        # size; the cache regrows (and re-pins its device) when a bigger batch arrives.
        self._decode_cu_seqlens_q: torch.Tensor | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        num_heads, q_len, head_dim = query.shape
        kv_len = key.shape[1]
        if key.shape[0] != num_heads or value.shape[0] != num_heads:
            raise ValueError(
                f"key/value head count must match query ({num_heads}); "
                f"got key {key.shape[0]}, value {value.shape[0]}"
            )
        if kv_len < q_len:
            raise ValueError(f"kv_len ({kv_len}) must be >= q_len ({q_len})")

        out_dtype = query.dtype
        # Head-major (heads, seq, head_dim) -> token-major (seq, heads, head_dim), bf16.
        q = query.transpose(0, 1).contiguous().to(_KERNEL_DTYPE)
        k = key.transpose(0, 1).contiguous().to(_KERNEL_DTYPE)
        v = value.transpose(0, 1).contiguous().to(_KERNEL_DTYPE)

        cu_seqlens_q = torch.tensor([0, q_len], dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.tensor([0, kv_len], dtype=torch.int32, device=k.device)

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=True,
        )
        # Back to head-major in the query's original dtype.
        return out.transpose(0, 1).contiguous().to(out_dtype)

    def forward_decode_batch_packed(
        self,
        queries: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Fused batched decode over already-packed token-major histories.

        ``key``/``value`` are ``(total_kv_tokens, num_heads, head_dim)`` and
        ``cu_seqlens_k`` describes each request's ragged history.
        """
        batch, _num_heads, head_dim = queries.shape
        out_dtype = queries.dtype
        q = queries.contiguous().to(_KERNEL_DTYPE)
        k = key.contiguous().to(_KERNEL_DTYPE)
        v = value.contiguous().to(_KERNEL_DTYPE)

        cu_seqlens_q = self._decode_arange(batch + 1, q.device)
        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=True,
        )
        return out.to(out_dtype)

    def _decode_arange(self, size: int, device: torch.device) -> torch.Tensor:
        """The cached 0..size-1 int32 arange for decode ``cu_seqlens_q`` (a sliced view)."""
        cached = self._decode_cu_seqlens_q
        if cached is None or cached.numel() < size or cached.device != device:
            cached = torch.arange(max(size, 2 * (0 if cached is None else cached.numel())),
                                  dtype=torch.int32, device=device)
            self._decode_cu_seqlens_q = cached
        return cached[:size]
