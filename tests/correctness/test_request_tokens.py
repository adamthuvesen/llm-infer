"""Request token state keeps tensors internally and materializes ints at boundaries."""

from __future__ import annotations

import torch

from llm_infer.serving.request import Request


def test_request_records_tensor_tokens_and_materializes_generated_ids() -> None:
    request = Request("r", [1, 2, 3], max_new_tokens=3, eos_token_ids=frozenset({9}))

    request.record(torch.tensor(4), is_eos=False)
    request.record(torch.tensor(5), is_eos=False)

    assert torch.equal(request.last_token_tensor, torch.tensor(5))
    assert request.generated == [4, 5]
    assert request.last_token == 5
    assert not request.finished


def test_request_honors_eos_flag_without_rechecking_token_id() -> None:
    request = Request("r", [1, 2, 3], max_new_tokens=3, eos_token_ids=frozenset({9}))

    request.record(torch.tensor(4), is_eos=True)

    assert request.generated == [4]
    assert request.finished
