"""The fast attention backend: flash-attn's fused varlen kernel behind the protocol.

This is the Phase C speed swap. It implements the same
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
  arrive fp32 (e.g. the fp32 oracle dtype), we run in bf16 and cast the output back —
  the resulting fp32-reduction-order difference vs ``torch_naive`` is exactly the
  genuine-tie effect the oracle's tie-tolerance bar accounts for (docs/fixture-format.md).

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

    def forward_decode_batch(
        self,
        queries: torch.Tensor,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> torch.Tensor:
        """Fused batched decode: one ragged varlen kernel call over all ``B`` requests.

        This is where batched continuous-batching decode pays off — ``B`` single-token
        queries and their ragged histories are packed into one ``flash_attn_varlen_func``
        call instead of ``B`` separate ones. Each request contributes exactly one query token
        (``cu_seqlens_q`` is ``[0, 1, 2, ..., B]``) and ``kv_len_i`` key tokens
        (``cu_seqlens_k`` is the cumulative history lengths), so ``causal=True`` lets each
        query attend over its own full history and nothing else.
        """
        if not (len(keys) == len(values) == queries.shape[0]):
            raise ValueError(
                f"queries/keys/values count mismatch: "
                f"{queries.shape[0]}, {len(keys)}, {len(values)}"
            )
        batch, _num_heads, head_dim = queries.shape
        out_dtype = queries.dtype
        # queries is already token-major: B query tokens, one per request -> (B, heads, head_dim).
        q = queries.contiguous().to(_KERNEL_DTYPE)
        # Each history is head-major (heads, kv_len_i, head_dim); to token-major and concat.
        kv_lens = [k.shape[1] for k in keys]
        k = torch.cat([h.transpose(0, 1) for h in keys], dim=0).contiguous().to(_KERNEL_DTYPE)
        v = torch.cat([h.transpose(0, 1) for h in values], dim=0).contiguous().to(_KERNEL_DTYPE)

        cu_seqlens_q = torch.arange(batch + 1, dtype=torch.int32, device=q.device)
        cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device=k.device)
        cu_seqlens_k[1:] = torch.tensor(kv_lens, dtype=torch.int32, device=k.device).cumsum(0)

        out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=1,
            max_seqlen_k=max(kv_lens),
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(head_dim),
            causal=True,
        )
        # out is (B, num_heads, head_dim) — one query token per request.
        return out.to(out_dtype)
