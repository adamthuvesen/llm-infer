"""FlashInfer backend that reads the KV cache's pages directly.

The default CUDA fp16/bf16 decode path for Esme bundles: it has passed the fp32 reference
gate with zero non-tie divergences and backs the headline benchmark. The flash-attn backend
receives packed, GQA-expanded K/V histories gathered by PyTorch; this backend keeps flash-attn
for prefill/full-recompute compatibility, but routes batched decode through FlashInfer's
``BatchDecodeWithPagedKVCacheWrapper`` using the model's native page-table metadata.
"""

from __future__ import annotations

import importlib
import math
from types import ModuleType

import torch

from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
from llm_infer.kv_cache.paged_kv_cache import KVPagePlan

_WORKSPACE_BYTES = 128 * 1024 * 1024


class FlashInferPagedAttention:
    """FlashInfer paged decode, with flash-attn as the non-decode fallback."""

    def __init__(
        self,
        *,
        workspace_bytes: int = _WORKSPACE_BYTES,
        use_tensor_cores: bool = False,
        backend: str = "auto",
    ) -> None:
        if workspace_bytes <= 0:
            raise ValueError(f"workspace_bytes must be positive; got {workspace_bytes}")
        self._packed = FlashAttnPagedAttention()
        self._flashinfer = self._import_flashinfer()
        self.workspace_bytes = workspace_bytes
        self.use_tensor_cores = use_tensor_cores
        self.backend = backend
        self._workspace: torch.Tensor | None = None
        self._wrapper: object | None = None
        # The (batch, num_qo_heads, head_dim) the live wrapper plan was built for, or None
        # before the first plan. FlashInfer's wrapper holds shape-specific metadata, so a decode
        # run against a different query shape without re-planning would read stale layout.
        self._planned_shape: tuple[int, int, int] | None = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return self._packed.forward(query, key, value)

    def forward_decode_batch_packed(
        self,
        queries: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        return self._packed.forward_decode_batch_packed(
            queries, key, value, cu_seqlens_k, max_seqlen_k
        )

    def forward_prefill_batch_packed(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        return self._packed.forward_prefill_batch_packed(query, key, value, cu_seqlens, max_seqlen)

    def plan_decode_batch_paged(
        self,
        page_plan: KVPagePlan,
        *,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        if dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"FlashInferPagedAttention needs a fp16/bf16 paged KV cache; got {dtype}"
            )
        wrapper = self._ensure_wrapper(page_plan.indptr.device)
        wrapper.plan(
            page_plan.indptr,
            page_plan.indices,
            page_plan.last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_plan.page_size,
            pos_encoding_mode="NONE",
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            sm_scale=1.0 / math.sqrt(head_dim),
        )
        batch = int(page_plan.indptr.numel()) - 1
        self._planned_shape = (batch, num_qo_heads, head_dim)

    def forward_decode_batch_paged(
        self, queries: torch.Tensor, paged_kv_cache: torch.Tensor
    ) -> torch.Tensor:
        if self._planned_shape is None or self._wrapper is None:
            raise RuntimeError("call plan_decode_batch_paged before paged decode")
        if tuple(queries.shape) != self._planned_shape:
            raise RuntimeError(
                "query shape "
                f"{tuple(queries.shape)} does not match the planned "
                f"(batch, num_qo_heads, head_dim) {self._planned_shape}; "
                "call plan_decode_batch_paged for this batch before paged decode"
            )
        out_dtype = queries.dtype
        out = self._wrapper.run(queries.contiguous(), paged_kv_cache)
        if not isinstance(out, torch.Tensor):
            raise RuntimeError("FlashInfer returned an unexpected non-tensor decode result")
        return out.to(out_dtype)

    def _ensure_wrapper(self, device: torch.device) -> object:
        workspace = self._workspace
        wrapper = self._wrapper
        if workspace is not None and workspace.device == device and wrapper is not None:
            return wrapper

        workspace = torch.zeros(self.workspace_bytes, dtype=torch.uint8, device=device)
        wrapper_cls = self._decode_wrapper_class(self._flashinfer)
        wrapper = wrapper_cls(
            workspace,
            "NHD",
            use_tensor_cores=self.use_tensor_cores,
            backend=self.backend,
        )
        self._workspace = workspace
        self._wrapper = wrapper
        return wrapper

    @staticmethod
    def _import_flashinfer() -> ModuleType:
        try:
            return importlib.import_module("flashinfer")
        except ImportError as exc:
            raise RuntimeError(
                "flashinfer-python is not installed, but it is the default attention backend "
                "for CUDA fp16/bf16 Esme bundles. Install the GPU extra (`uv sync --extra gpu`) "
                "or pick another backend with `--attention-backend torch_naive` or "
                "`--attention-backend flash_attn`."
            ) from exc

    @staticmethod
    def _decode_wrapper_class(flashinfer: ModuleType):
        wrapper_cls = getattr(flashinfer, "BatchDecodeWithPagedKVCacheWrapper", None)
        if wrapper_cls is not None:
            return wrapper_cls
        decode_module = getattr(flashinfer, "decode", None)
        if decode_module is not None:
            wrapper_cls = getattr(decode_module, "BatchDecodeWithPagedKVCacheWrapper", None)
            if wrapper_cls is not None:
                return wrapper_cls
        raise RuntimeError("FlashInfer does not expose BatchDecodeWithPagedKVCacheWrapper")


__all__ = ["FlashInferPagedAttention"]
