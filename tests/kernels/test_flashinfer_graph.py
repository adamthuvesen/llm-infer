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
    def __init__(self, *args, **kwargs) -> None:
        self.init_args = args
        self.init_kwargs = kwargs
        self.plan_args = None
        self.plan_kwargs = None

    def plan(self, *args, **kwargs) -> None:
        self.plan_args = args
        self.plan_kwargs = kwargs

    def run(self, query, paged_kv, *, out):
        del paged_kv
        out.copy_(query + 1)
        return out


def test_graph_wrapper_uses_caller_owned_fixed_buffers() -> None:
    fixed = page_metadata_for_lengths([129, 129], page_size=64, device="cpu")
    workspace = torch.empty(16, dtype=torch.uint8)
    wrapper = build_graph_wrapper(_FakeWrapper, workspace, fixed)

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
