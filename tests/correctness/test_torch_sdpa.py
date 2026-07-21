"""torch_sdpa parity: the fused SDPA backend reproduces the torch_naive reference exactly.

The SDPA backend is a drop-in local speed swap for ``torch_naive``. On fp32/CPU the two run
the same math, so parity here is tight (``rtol/atol ~1e-5``), not a tie-tolerance rule — SDPA
is not a bf16 kernel. The kernel tests pin the three ``forward`` causal cases (full prefill,
chunked-prefill offset, decode), the ragged packed decode, and the GQA packed prefill. The
engine tests pin exact greedy token ids on the tiny bundle, including continuous batching and
chunked prefill. The serve tests pin the ``auto`` -> ``torch_sdpa`` local-default resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as _write_tiny_bundle
from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.kernels.torch_sdpa import TorchSdpaAttention
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serve import resolve_attention_backend_default
from llm_infer.serving import InferenceEngine, Request

RTOL = 1e-5
ATOL = 1e-5


def _cu_seqlens(lengths: list[int]) -> torch.Tensor:
    """Cumulative int32 offsets ``[0, l0, l0+l1, ...]`` for packed varlen inputs."""
    offsets = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    offsets[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)
    return offsets


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def test_forward_full_prefill_matches_reference() -> None:
    """q_len == kv_len: the is_causal path."""
    num_heads, seq, head_dim = 4, 6, 8
    query = torch.randn(num_heads, seq, head_dim)
    key = torch.randn(num_heads, seq, head_dim)
    value = torch.randn(num_heads, seq, head_dim)

    sdpa = TorchSdpaAttention().forward(query, key, value)
    reference = TorchNaiveAttention().forward(query, key, value)

    torch.testing.assert_close(sdpa, reference, rtol=RTOL, atol=ATOL)


def test_forward_chunked_prefill_offset_mask_matches_reference() -> None:
    """1 < q_len < kv_len: the explicit offset-mask path (the chunked-prefill trap)."""
    num_heads, q_len, kv_len, head_dim = 4, 3, 7, 8
    query = torch.randn(num_heads, q_len, head_dim)
    key = torch.randn(num_heads, kv_len, head_dim)
    value = torch.randn(num_heads, kv_len, head_dim)

    sdpa = TorchSdpaAttention().forward(query, key, value)
    reference = TorchNaiveAttention().forward(query, key, value)

    torch.testing.assert_close(sdpa, reference, rtol=RTOL, atol=ATOL)


def test_forward_decode_matches_reference() -> None:
    """q_len == 1: every key visible, no mask."""
    num_heads, kv_len, head_dim = 4, 9, 8
    query = torch.randn(num_heads, 1, head_dim)
    key = torch.randn(num_heads, kv_len, head_dim)
    value = torch.randn(num_heads, kv_len, head_dim)

    sdpa = TorchSdpaAttention().forward(query, key, value)
    reference = TorchNaiveAttention().forward(query, key, value)

    torch.testing.assert_close(sdpa, reference, rtol=RTOL, atol=ATOL)


def test_forward_rejects_kv_shorter_than_query() -> None:
    """Same loud input validation as the reference."""
    query = torch.randn(4, 5, 8)
    key = torch.randn(4, 3, 8)
    with pytest.raises(ValueError, match="kv_len"):
        TorchSdpaAttention().forward(query, key, key)


def test_forward_rejects_head_count_mismatch() -> None:
    query = torch.randn(4, 5, 8)
    key = torch.randn(2, 5, 8)
    with pytest.raises(ValueError, match="head count"):
        TorchSdpaAttention().forward(query, key, key)


def test_decode_batch_packed_ragged_matches_reference() -> None:
    """Uneven histories, single-token queries: the packed ragged decode path."""
    num_heads, head_dim = 4, 8
    lengths = [3, 7, 1, 5]
    queries = torch.randn(len(lengths), num_heads, head_dim)
    total = sum(lengths)
    key = torch.randn(total, num_heads, head_dim)
    value = torch.randn(total, num_heads, head_dim)
    cu_seqlens_k = _cu_seqlens(lengths)
    max_len = max(lengths)

    sdpa = TorchSdpaAttention().forward_decode_batch_packed(
        queries, key, value, cu_seqlens_k, max_len
    )
    reference = TorchNaiveAttention().forward_decode_batch_packed(
        queries, key, value, cu_seqlens_k, max_len
    )

    torch.testing.assert_close(sdpa, reference, rtol=RTOL, atol=ATOL)


def test_prefill_batch_packed_gqa_uneven_matches_reference() -> None:
    """Native-GQA K/V (kv_heads < q_heads) over uneven prompts: the packed prefill path."""
    num_qo_heads, num_kv_heads, head_dim = 6, 2, 8
    lengths = [4, 2, 5]
    total = sum(lengths)
    query = torch.randn(total, num_qo_heads, head_dim)
    key = torch.randn(total, num_kv_heads, head_dim)
    value = torch.randn(total, num_kv_heads, head_dim)
    cu_seqlens = _cu_seqlens(lengths)
    max_len = max(lengths)

    sdpa = TorchSdpaAttention().forward_prefill_batch_packed(query, key, value, cu_seqlens, max_len)
    reference = TorchNaiveAttention().forward_prefill_batch_packed(
        query, key, value, cu_seqlens, max_len
    )

    torch.testing.assert_close(sdpa, reference, rtol=RTOL, atol=ATOL)


def _greedy(
    bundle: Path,
    backend_name: str,
    requests: list[Request],
    *,
    prefill_chunk_size: int | None = None,
) -> dict[str, list[int]]:
    """Run greedy generation on the tiny bundle under a named attention backend."""
    runtime = load_model_runtime("esme", bundle_path=bundle, attention_backend_name=backend_name)
    engine = InferenceEngine(
        runtime.model,
        block_size=4,
        num_blocks=64,
        capabilities=runtime.capabilities,
        prefill_chunk_size=prefill_chunk_size,
    )
    for request in requests:
        engine.add_request(request)
    return engine.run()


def test_engine_greedy_single_request_matches_reference(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)

    def build() -> list[Request]:
        # Fresh requests per run: a Request carries generation state, so it must not be reused.
        return [Request("only", [1, 4, 7, 2, 9], 12, frozenset())]

    naive = _greedy(bundle, "torch_naive", build())
    sdpa = _greedy(bundle, "torch_sdpa", build())

    assert sdpa["only"] == naive["only"]


def test_engine_greedy_continuous_batching_matches_reference(tmp_path: Path) -> None:
    """Several ragged requests decoded together must match the reference token-for-token."""
    bundle = _write_tiny_bundle(tmp_path)

    def build() -> list[Request]:
        return [
            Request("a", [1, 4, 7], 10, frozenset()),
            Request("b", [2, 9], 10, frozenset()),
            Request("c", [3, 5, 8, 6], 10, frozenset()),
        ]

    naive = _greedy(bundle, "torch_naive", build())
    sdpa = _greedy(bundle, "torch_sdpa", build())

    assert sdpa == naive


def test_engine_greedy_chunked_prefill_matches_reference(tmp_path: Path) -> None:
    """A prompt chunk-prefilled in pieces (the offset-mask forward path) must still match."""
    bundle = _write_tiny_bundle(tmp_path)

    def build() -> list[Request]:
        return [Request("only", [1, 4, 7, 2, 9, 5, 8], 10, frozenset())]

    naive = _greedy(bundle, "torch_naive", build(), prefill_chunk_size=2)
    sdpa = _greedy(bundle, "torch_sdpa", build(), prefill_chunk_size=2)

    assert sdpa["only"] == naive["only"]


def test_resolve_attention_backend_default_local_bundle_picks_sdpa() -> None:
    assert resolve_attention_backend_default("auto", device="cpu", backend="esme") == "torch_sdpa"
    assert resolve_attention_backend_default("auto", device="cpu", backend="dense") == "torch_sdpa"


def test_resolve_attention_backend_default_cuda_and_qwen_stay_auto() -> None:
    # CUDA auto behavior (FlashInfer) must not change.
    assert resolve_attention_backend_default("auto", device="cuda", backend="esme") == "auto"
    # The non-bundle reference backend keeps auto even on CPU.
    assert resolve_attention_backend_default("auto", device="cpu", backend="qwen") == "auto"


def test_resolve_attention_backend_default_explicit_passes_through() -> None:
    for name in ("torch_naive", "torch_sdpa", "flash_attn", "flashinfer"):
        assert resolve_attention_backend_default(name, device="cpu", backend="esme") == name
