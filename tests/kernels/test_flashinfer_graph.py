"""CPU contracts for the FlashInfer CUDA-graph adapter."""

from __future__ import annotations

import torch

from llm_infer.kernels.flashinfer_graph import (
    FlashInferDecodeShape,
    build_graph_wrapper,
    page_metadata_for_lengths,
    plan_wrapper,
    run_wrapper_into,
)


class _FakeWrapper:
    def __init__(
        self,
        workspace: torch.Tensor,
        kv_layout: str,
        *,
        use_tensor_cores: bool,
        backend: str,
        use_cuda_graph: bool = False,
        paged_kv_indptr_buffer: torch.Tensor | None = None,
        paged_kv_indices_buffer: torch.Tensor | None = None,
        paged_kv_last_page_len_buffer: torch.Tensor | None = None,
    ) -> None:
        self.init_args = (workspace, kv_layout)
        self.init_kwargs: dict[str, object] = {
            "use_tensor_cores": use_tensor_cores,
            "backend": backend,
            "use_cuda_graph": use_cuda_graph,
            "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
            "paged_kv_indices_buffer": paged_kv_indices_buffer,
            "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
        }
        self.plan_args: tuple[object, ...] | None = None

    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *,
        pos_encoding_mode: str,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        o_data_type: torch.dtype,
        sm_scale: float,
    ) -> None:
        self.plan_args = (
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
        )

    def run(
        self, query: torch.Tensor, paged_kv: torch.Tensor, *, out: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        del paged_kv
        assert out is not None
        out.copy_(query + 1)
        return out


def test_graph_wrapper_uses_caller_owned_fixed_buffers() -> None:
    fixed = page_metadata_for_lengths([129, 129], page_size=64, device="cpu")
    workspace = torch.empty(16, dtype=torch.uint8)
    wrapper = build_graph_wrapper(_FakeWrapper, workspace, fixed)
    assert isinstance(wrapper, _FakeWrapper)

    assert wrapper.init_args == (workspace, "NHD")
    assert wrapper.init_kwargs["use_cuda_graph"] is True
    assert wrapper.init_kwargs["paged_kv_indptr_buffer"] is fixed.indptr
    assert wrapper.init_kwargs["paged_kv_indices_buffer"] is fixed.indices
    assert wrapper.init_kwargs["paged_kv_last_page_len_buffer"] is fixed.last_page_len


def test_plan_and_run_keep_exact_metadata_and_output_contract() -> None:
    fixed = page_metadata_for_lengths([129], page_size=64, device="cpu")
    wrapper = build_graph_wrapper(_FakeWrapper, torch.empty(16, dtype=torch.uint8), fixed)
    metadata = page_metadata_for_lengths([65], page_size=64, device="cpu")
    shape = FlashInferDecodeShape(8, 2, 64, 64, torch.bfloat16)
    plan_wrapper(wrapper, metadata, shape)
    assert isinstance(wrapper, _FakeWrapper)
    assert wrapper.plan_args is not None

    assert wrapper.plan_args[:3] == (
        metadata.indptr,
        metadata.indices,
        metadata.last_page_len,
    )
    assert wrapper.plan_args[3:7] == (8, 2, 64, 64)
    query = torch.zeros(1, 8, 64)
    output = torch.empty_like(query)
    run_wrapper_into(wrapper, query, torch.empty(1), output)
    assert torch.equal(output, torch.ones_like(query))


def test_page_metadata_crosses_64_and_128_boundaries() -> None:
    metadata = page_metadata_for_lengths([63, 64, 65, 127, 128, 129], page_size=64, device="cpu")

    assert metadata.indptr.tolist() == [0, 1, 2, 4, 6, 8, 11]
    assert metadata.last_page_len.tolist() == [63, 64, 1, 63, 64, 1]
    assert metadata.indices.tolist() == list(range(11))
