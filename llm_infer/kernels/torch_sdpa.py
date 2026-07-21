"""The local fast path: PyTorch's fused SDPA behind the attention protocol.

``torch.nn.functional.scaled_dot_product_attention`` (SDPA) is a fused attention kernel
that ships with PyTorch and runs on CPU and MPS without a CUDA build. This backend swaps
``torch_naive``'s materialized-mask fp32 loop for that fused call while keeping the exact
same per-request, already-RoPE'd, already-GQA-expanded protocol tensors in and out — so it
is a drop-in speed swap for local (non-CUDA) serving, validated token-for-token against the
reference.

Two facts bridge the protocol to SDPA:

* **Causal alignment.** SDPA's ``is_causal=True`` uses a *top-left* aligned triangular mask
  (query row ``i`` attends key cols ``0..i``), while the protocol aligns to the bottom-right
  (query row ``i`` attends cols ``0..(kv_len - q_len) + i``). The two agree only when
  ``q_len == kv_len``. So we use ``is_causal=True`` for a full prefill, no mask at all for a
  single decode step (every key visible), and an explicit boolean offset mask for a
  chunked-prefill step where ``1 < q_len < kv_len``.
* **dtype.** ``torch_naive`` upcasts to fp32; SDPA runs in the input dtype on purpose — that
  is the point of the fused path (fp32 on CPU today, fp16 on MPS later). The output is
  returned in the query's dtype, as the protocol requires.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from llm_infer.kernels.base import packed_prefill_lengths, validate_packed_prefill_inputs


def _supports_enable_gqa() -> bool:
    """Probe once whether the installed SDPA accepts ``enable_gqa`` on CPU.

    ``enable_gqa`` lets SDPA consume native KV-head tensors without materializing repeated
    heads. It landed in torch 2.5; probing beats guessing from a version string because the
    kwarg's device coverage is what actually matters.
    """
    query = torch.zeros(1, 2, 1, 1)
    key = torch.zeros(1, 1, 1, 1)
    try:
        F.scaled_dot_product_attention(query, key, key, enable_gqa=True)
    except (RuntimeError, TypeError):
        return False
    return True


_ENABLE_GQA = _supports_enable_gqa()


class TorchSdpaAttention:
    """Fused causal attention for one request via PyTorch SDPA.

    Implements the :class:`~llm_infer.kernels.base.PackedPrefillAttentionBackend` protocol
    with the same semantics — and the same input-validation errors — as
    :class:`~llm_infer.kernels.torch_naive.TorchNaiveAttention`.
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
        out = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=self._offset_mask(q_len, kv_len, query.device),
            # is_causal only matches the protocol's bottom-right alignment when q_len == kv_len;
            # every other case supplies an explicit mask (or none for a single decode step).
            is_causal=q_len == kv_len,
            scale=scale,
        )
        return out.to(query.dtype)

    def forward_decode_batch_packed(
        self,
        queries: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Batched single-token decode over token-major packed histories.

        A decode query sees its whole history (``q_len == 1``, no mask). The batch is usually 1
        locally and the per-request SDPA loop measured faster than a padded single call there, so
        it is the kept path — identical by construction to attending each request alone.
        """
        del max_seqlen_k
        batch = queries.shape[0]
        offsets = cu_seqlens_k.tolist()
        scale = 1.0 / math.sqrt(queries.shape[-1])
        outs = []
        for index in range(batch):
            start, end = offsets[index], offsets[index + 1]
            # (num_heads, 1, head_dim) query over this request's (num_heads, seq, head_dim) history.
            query = queries[index].unsqueeze(1)
            key_hist = key[start:end].transpose(0, 1)
            value_hist = value[start:end].transpose(0, 1)
            out = F.scaled_dot_product_attention(query, key_hist, value_hist, scale=scale)
            outs.append(out.squeeze(1))
        return torch.stack(outs, dim=0).to(queries.dtype)

    def forward_decode_batch_packed_grouped(
        self,
        queries: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Batched single-token decode over native-KV-head packed histories.

        Same per-request SDPA loop as :meth:`forward_decode_batch_packed`, but the history stays
        at ``num_kv_heads`` and SDPA broadcasts it over the query-head groups (``enable_gqa``).
        The caller keeps the full history one third the size for a 3:1 GQA ratio, skipping the
        per-step ``repeat_interleave`` over the whole context. Falls back to repeating the KV
        heads when the installed SDPA lacks ``enable_gqa`` — the same fallback the prefill path
        uses — so the result is identical either way.
        """
        del max_seqlen_k
        batch = queries.shape[0]
        num_qo_heads = queries.shape[1]
        num_kv_heads = key.shape[1]
        if num_qo_heads == 0 or num_kv_heads == 0 or num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                "grouped decode query head count must be divisible by the KV head count; "
                f"got query {num_qo_heads}, KV {num_kv_heads}"
            )
        gqa_repeats = num_qo_heads // num_kv_heads
        fused_gqa = gqa_repeats != 1 and _ENABLE_GQA
        offsets = cu_seqlens_k.tolist()
        scale = 1.0 / math.sqrt(queries.shape[-1])
        outs = []
        for index in range(batch):
            start, end = offsets[index], offsets[index + 1]
            query = queries[index].unsqueeze(1)  # (num_qo_heads, 1, head_dim)
            key_hist = key[start:end].transpose(0, 1)  # (num_kv_heads, seq, head_dim)
            value_hist = value[start:end].transpose(0, 1)
            if fused_gqa:
                out = F.scaled_dot_product_attention(
                    query, key_hist, value_hist, scale=scale, enable_gqa=True
                )
            else:
                if gqa_repeats != 1:
                    key_hist = key_hist.repeat_interleave(gqa_repeats, dim=0)
                    value_hist = value_hist.repeat_interleave(gqa_repeats, dim=0)
                out = F.scaled_dot_product_attention(query, key_hist, value_hist, scale=scale)
            outs.append(out.squeeze(1))
        return torch.stack(outs, dim=0).to(queries.dtype)

    def forward_prefill_batch_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Run per-request causal SDPA over packed prompts with native GQA K/V heads."""
        validate_packed_prefill_inputs(query, key, value, cu_seqlens, max_seqlen)
        lengths = packed_prefill_lengths(
            cu_seqlens,
            total_tokens=query.shape[0],
            max_seqlen=max_seqlen,
        )
        gqa_repeats = query.shape[1] // key.shape[1]
        scale = 1.0 / math.sqrt(query.shape[2])
        # Prefer SDPA's native GQA; fall back to repeating KV heads like the reference when the
        # installed SDPA does not accept ``enable_gqa``.
        fused_gqa = gqa_repeats != 1 and _ENABLE_GQA
        outputs = []
        for query_chunk, key_chunk, value_chunk in zip(
            query.split(lengths),
            key.split(lengths),
            value.split(lengths),
            strict=True,
        ):
            # (num_qo_heads, seq, head_dim) query; K/V start at native KV-head count.
            q = query_chunk.transpose(0, 1)
            k = key_chunk.transpose(0, 1)
            v = value_chunk.transpose(0, 1)
            if fused_gqa:
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale, enable_gqa=True
                )
            else:
                if gqa_repeats != 1:
                    k = k.repeat_interleave(gqa_repeats, dim=0)
                    v = v.repeat_interleave(gqa_repeats, dim=0)
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale)
            outputs.append(out.transpose(0, 1))
        return torch.cat(outputs, dim=0).to(query.dtype)

    def _offset_mask(self, q_len: int, kv_len: int, device: torch.device) -> torch.Tensor | None:
        """Boolean SDPA mask (True = allowed) for the chunked-prefill offset case.

        ``None`` for a full prefill (``is_causal`` handles it) and for a single decode step
        (every key visible). For ``1 < q_len < kv_len`` — a chunked prefill step attending its
        cached prefix — query row ``i`` (absolute position ``kv_len - q_len + i``) may attend
        key cols ``0 .. kv_len - q_len + i``, the same rule as ``torch_naive``'s ``_causal_mask``.
        """
        if q_len == kv_len or q_len == 1:
            return None
        offset = kv_len - q_len
        col = torch.arange(kv_len, device=device).view(1, kv_len)
        row = torch.arange(q_len, device=device).view(q_len, 1) + offset
        return col <= row
