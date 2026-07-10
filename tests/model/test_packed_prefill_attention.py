from __future__ import annotations

from types import ModuleType

import pytest
import torch

from llm_infer.kernels.base import PackedPrefillAttentionBackend
from llm_infer.kernels.torch_naive import TorchNaiveAttention


def _packed_inputs() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(5, 4, 8, generator=generator)
    key = torch.randn(5, 2, 8, generator=generator)
    value = torch.randn(5, 2, 8, generator=generator)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
    return query, key, value, cu_seqlens


def test_torch_naive_packed_prefill_matches_isolated_gqa_sequences() -> None:
    backend = TorchNaiveAttention()
    query, key, value, cu_seqlens = _packed_inputs()

    actual = backend.forward_prefill_batch_packed(
        query, key, value, cu_seqlens, max_seqlen=3
    )

    expected_sequences = []
    for start, end in ((0, 2), (2, 5)):
        expanded_key = key[start:end].repeat_interleave(2, dim=1)
        expanded_value = value[start:end].repeat_interleave(2, dim=1)
        expected = backend.forward(
            query[start:end].transpose(0, 1),
            expanded_key.transpose(0, 1),
            expanded_value.transpose(0, 1),
        )
        expected_sequences.append(expected.transpose(0, 1))

    assert isinstance(backend, PackedPrefillAttentionBackend)
    torch.testing.assert_close(actual, torch.cat(expected_sequences), rtol=0, atol=0)


def test_torch_naive_packed_prefill_does_not_leak_between_sequences() -> None:
    backend = TorchNaiveAttention()
    query, key, value, cu_seqlens = _packed_inputs()
    baseline = backend.forward_prefill_batch_packed(
        query, key, value, cu_seqlens, max_seqlen=3
    )

    changed_key = key.clone()
    changed_value = value.clone()
    changed_key[2:] += 1000
    changed_value[2:] -= 1000
    changed = backend.forward_prefill_batch_packed(
        query, changed_key, changed_value, cu_seqlens, max_seqlen=3
    )

    torch.testing.assert_close(changed[:2], baseline[:2], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("query", "key", "value", "cu_seqlens", "max_seqlen", "message"),
    [
        (
            torch.zeros(3, 4, 8),
            torch.zeros(2, 2, 8),
            torch.zeros(2, 2, 8),
            torch.tensor([0, 3], dtype=torch.int32),
            3,
            "token counts",
        ),
        (
            torch.zeros(3, 3, 8),
            torch.zeros(3, 2, 8),
            torch.zeros(3, 2, 8),
            torch.tensor([0, 3], dtype=torch.int32),
            3,
            "divisible",
        ),
        (
            torch.zeros(3, 4, 8),
            torch.zeros(3, 2, 8),
            torch.zeros(3, 2, 8),
            torch.tensor([0, 2], dtype=torch.int32),
            3,
            "final offset",
        ),
        (
            torch.zeros(3, 4, 8),
            torch.zeros(3, 2, 8),
            torch.zeros(3, 2, 8),
            torch.tensor([0, 1, 3], dtype=torch.int32),
            1,
            "max_seqlen",
        ),
    ],
)
def test_torch_naive_packed_prefill_rejects_invalid_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TorchNaiveAttention().forward_prefill_batch_packed(
            query, key, value, cu_seqlens, max_seqlen
        )


def test_flash_attention_packed_prefill_uses_one_native_gqa_varlen_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_infer.kernels import flash_attn_paged

    calls: list[dict[str, object]] = []

    def fake_varlen(query, key, value, **kwargs):
        calls.append({"query": query, "key": key, "value": value, **kwargs})
        return torch.zeros_like(query)

    monkeypatch.setattr(flash_attn_paged, "flash_attn_varlen_func", fake_varlen)
    backend = flash_attn_paged.FlashAttnPagedAttention()
    query, key, value, cu_seqlens = _packed_inputs()

    actual = backend.forward_prefill_batch_packed(
        query, key, value, cu_seqlens, max_seqlen=3
    )

    assert actual.shape == query.shape
    assert actual.dtype == query.dtype
    assert len(calls) == 1
    assert calls[0]["query"].shape == (5, 4, 8)
    assert calls[0]["key"].shape == (5, 2, 8)
    assert calls[0]["value"].shape == (5, 2, 8)
    assert calls[0]["cu_seqlens_q"] is cu_seqlens
    assert calls[0]["cu_seqlens_k"] is cu_seqlens
    assert calls[0]["max_seqlen_q"] == 3
    assert calls[0]["max_seqlen_k"] == 3
    assert calls[0]["causal"] is True


def test_flashinfer_packed_prefill_delegates_to_flash_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llm_infer.kernels import flashinfer_paged

    calls: list[tuple[object, ...]] = []

    class FakePacked:
        def forward_prefill_batch_packed(self, *args):
            calls.append(args)
            return torch.ones_like(args[0])

    monkeypatch.setattr(flashinfer_paged, "FlashAttnPagedAttention", FakePacked)
    monkeypatch.setattr(
        flashinfer_paged.importlib,
        "import_module",
        lambda name: ModuleType("flashinfer"),
    )
    backend = flashinfer_paged.FlashInferPagedAttention()
    query, key, value, cu_seqlens = _packed_inputs()

    actual = backend.forward_prefill_batch_packed(query, key, value, cu_seqlens, 3)

    assert len(calls) == 1
    assert calls[0] == (query, key, value, cu_seqlens, 3)
    torch.testing.assert_close(actual, torch.ones_like(query))
