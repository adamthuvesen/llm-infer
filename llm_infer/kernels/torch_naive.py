"""The reference attention backend: slow, explicit, and correct by construction.

This is the truth every other backend is measured against. It favours legibility
over speed — the causal mask is materialized, the softmax is done in fp32, and there
is no fusion or paging. Nothing here should ever be "optimized"; its only job is to
be obviously right.
"""

from __future__ import annotations

import math

import torch


class TorchNaiveAttention:
    """Materialized-mask causal attention for one request.

    Implements the :class:`~llm_infer.kernels.base.AttentionBackend` protocol.
    """

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

        scale = 1.0 / math.sqrt(head_dim)
        # (num_heads, q_len, kv_len) raw scores, computed and softmaxed in fp32.
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
        scores = scores + _causal_mask(q_len, kv_len, scores.device)
        weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(weights, value.float())
        return out.to(query.dtype)

    def forward_decode_batch(
        self,
        queries: torch.Tensor,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
    ) -> torch.Tensor:
        """Reference batched decode: loop over requests, reusing the single-request path.

        No fusion — each request is attended exactly as :meth:`forward` would alone, so the
        batched result is identical-by-construction to running each request serially. That is
        the whole point of the reference: the fast backend's fused ragged kernel is validated
        against this loop.
        """
        if not (len(keys) == len(values) == queries.shape[0]):
            raise ValueError(
                f"queries/keys/values count mismatch: "
                f"{queries.shape[0]}, {len(keys)}, {len(values)}"
            )
        outs = [
            self.forward(q.unsqueeze(1), k, v).squeeze(1)
            for q, k, v in zip(queries, keys, values, strict=True)
        ]
        return torch.stack(outs, dim=0)


def _causal_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    """Additive mask: 0 where attention is allowed, -inf where it is forbidden.

    Query row ``i`` is the absolute position ``kv_len - q_len + i`` and may attend to
    key columns ``0 .. kv_len - q_len + i``. On a decode step (``q_len == 1``) every
    column is visible; during prefill this is the standard lower-triangular block.
    """
    offset = kv_len - q_len
    col = torch.arange(kv_len, device=device).view(1, kv_len)
    row = torch.arange(q_len, device=device).view(q_len, 1) + offset
    allowed = col <= row
    mask = torch.zeros(q_len, kv_len, dtype=torch.float32, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask
