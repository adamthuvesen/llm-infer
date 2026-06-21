"""The narrow attention interface every backend implements.

A backend computes scaled dot-product attention for one request: given the query
rows for the current step (prefill: all prompt positions; decode: the single new
position) and the full key/value history for that request, it returns the attention
output. Position encoding (RoPE) and GQA head expansion happen *before* the backend
is called — keys and values arrive already rotated and already repeated to the query
head count — so a backend only owns the `softmax(QKᵀ / √d) V` causal core.

Keeping the interface this narrow is the point: a later paged/flash backend is a
drop-in swap validated against `torch_naive` by the correctness oracle, not a
rewrite of the model forward.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


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

    def forward_decode_batch(
        self,
        queries: torch.Tensor,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> torch.Tensor:
        """Batched single-token decode for ``B`` independent requests in one call.

        This is the fused-batch decode path: every running request advances by exactly one
        token, but each has its own (ragged) KV history, so the requests cannot be stacked
        into a dense tensor. The fast backend packs them into one ragged kernel call; the
        reference loops. The win over calling :meth:`forward` per request is a single kernel
        launch / matmul instead of ``B`` of them.

        Shapes:
            queries: ``(B, num_heads, head_dim)`` — one decode query per request.
            keys, values: a length-``B`` list, each ``(num_heads, kv_len_i, head_dim)`` — the
                request's full history including this step, already RoPE'd and GQA-expanded.
                ``kv_len_i`` varies per request.

        Returns:
            ``(B, num_heads, head_dim)`` — one attention output per request, row ``i`` being
            request ``i``'s output, identical to ``forward(queries[i:i+1]..., keys[i], values[i])``.
        """
        ...
