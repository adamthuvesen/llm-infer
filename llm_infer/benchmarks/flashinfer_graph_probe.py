"""Small host-side helpers for the benchmark-only FlashInfer CUDA-graph probe."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlashInferProbeShape:
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    dtype: torch.dtype


@dataclass(frozen=True)
class FlashInferPageMetadata:
    indptr: torch.Tensor
    indices: torch.Tensor
    last_page_len: torch.Tensor


def page_metadata_for_lengths(
    lengths: list[int], *, page_size: int, device: torch.device | str
) -> FlashInferPageMetadata:
    """Build request-major page metadata with private sequential pages."""
    if not lengths or any(length < 1 for length in lengths):
        raise ValueError(f"lengths must be positive and nonempty; got {lengths}")
    if page_size < 1:
        raise ValueError(f"page_size must be positive; got {page_size}")
    page_counts = [-(-length // page_size) for length in lengths]
    indptr = [0]
    for count in page_counts:
        indptr.append(indptr[-1] + count)
    return FlashInferPageMetadata(
        indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        indices=torch.arange(indptr[-1], dtype=torch.int32, device=device),
        last_page_len=torch.tensor(
            [((length - 1) % page_size) + 1 for length in lengths],
            dtype=torch.int32,
            device=device,
        ),
    )


def build_graph_wrapper(
    wrapper_class,
    workspace: torch.Tensor,
    fixed_metadata: FlashInferPageMetadata,
):
    """Construct the pinned FlashInfer wrapper with caller-owned graph buffers."""
    return wrapper_class(
        workspace,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=fixed_metadata.indptr,
        paged_kv_indices_buffer=fixed_metadata.indices,
        paged_kv_last_page_len_buffer=fixed_metadata.last_page_len,
        use_tensor_cores=False,
        backend="auto",
    )


def plan_wrapper(
    wrapper,
    metadata: FlashInferPageMetadata,
    shape: FlashInferProbeShape,
) -> None:
    """Plan one token step outside capture; the wrapper copies into its fixed buffers."""
    wrapper.plan(
        metadata.indptr,
        metadata.indices,
        metadata.last_page_len,
        shape.num_qo_heads,
        shape.num_kv_heads,
        shape.head_dim,
        shape.page_size,
        pos_encoding_mode="NONE",
        q_data_type=shape.dtype,
        kv_data_type=shape.dtype,
        o_data_type=shape.dtype,
        sm_scale=shape.head_dim**-0.5,
    )


def run_wrapper_into(
    wrapper, query: torch.Tensor, paged_kv: torch.Tensor, output: torch.Tensor
) -> None:
    """Run into a caller-owned output so capture replays keep its address fixed."""
    result = wrapper.run(query, paged_kv, out=output)
    if result is not None and result is not output:
        output.copy_(result)
