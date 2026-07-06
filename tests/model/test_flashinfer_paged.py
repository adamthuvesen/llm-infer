"""CPU coverage for FlashInfer host-side logic and the serve warmup hook.

FlashInfer only builds on CUDA, so these tests never import the real package: they fake
the module (and the flash-attn packed fallback) with monkeypatch, exactly like
``test_auto_bundle_cuda_low_precision_selects_flashinfer`` fakes runtime resolution. That
keeps the import guard, wrapper-class lookup, workspace/dtype validation, and the plan/shape
guard runnable on the dev host.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from llm_infer import serve
from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.kernels import flashinfer_paged
from llm_infer.kernels.flashinfer_paged import FlashInferPagedAttention
from llm_infer.kv_cache.paged_kv_cache import KVPagePlan
from llm_infer.model.runtime import load_model_runtime


class _FakeWrapper:
    """Stand-in for BatchDecodeWithPagedKVCacheWrapper: records plan, echoes zeros on run."""

    def __init__(self, workspace, layout, *, use_tensor_cores, backend) -> None:
        self.workspace = workspace
        self.layout = layout
        self.use_tensor_cores = use_tensor_cores
        self.backend = backend
        self.plan_calls = 0
        self.run_calls = 0

    def plan(self, *args, **kwargs) -> None:
        self.plan_calls += 1

    def run(self, queries: torch.Tensor, paged_kv_cache: torch.Tensor) -> torch.Tensor:
        self.run_calls += 1
        return torch.zeros_like(queries)


def _fake_flashinfer_module() -> ModuleType:
    module = ModuleType("flashinfer")
    module.BatchDecodeWithPagedKVCacheWrapper = _FakeWrapper
    return module


@pytest.fixture()
def cpu_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``FlashInferPagedAttention()`` constructible on the CPU dev host."""
    monkeypatch.setattr(flashinfer_paged, "FlashAttnPagedAttention", lambda: object())
    monkeypatch.setattr(
        flashinfer_paged.importlib, "import_module", lambda name: _fake_flashinfer_module()
    )


def test_import_guard_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(name: str):
        raise ImportError(f"no module named {name}")

    monkeypatch.setattr(flashinfer_paged.importlib, "import_module", _missing)

    with pytest.raises(RuntimeError) as excinfo:
        FlashInferPagedAttention._import_flashinfer()

    message = str(excinfo.value)
    assert "uv sync --extra gpu" in message
    assert "--attention-backend torch_naive" in message


def test_decode_wrapper_class_prefers_top_level() -> None:
    module = ModuleType("flashinfer")
    module.BatchDecodeWithPagedKVCacheWrapper = _FakeWrapper

    assert FlashInferPagedAttention._decode_wrapper_class(module) is _FakeWrapper


def test_decode_wrapper_class_falls_back_to_submodule() -> None:
    module = ModuleType("flashinfer")
    module.decode = SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=_FakeWrapper)

    assert FlashInferPagedAttention._decode_wrapper_class(module) is _FakeWrapper


def test_decode_wrapper_class_missing_raises() -> None:
    module = ModuleType("flashinfer")

    with pytest.raises(RuntimeError, match="BatchDecodeWithPagedKVCacheWrapper"):
        FlashInferPagedAttention._decode_wrapper_class(module)


def test_non_positive_workspace_bytes_rejected() -> None:
    with pytest.raises(ValueError, match="workspace_bytes must be positive"):
        FlashInferPagedAttention(workspace_bytes=0)


def test_plan_rejects_fp32(cpu_flashinfer: None) -> None:
    backend = FlashInferPagedAttention()
    page_plan = KVPagePlan(
        indptr=torch.tensor([0, 1], dtype=torch.int32),
        indices=torch.tensor([0], dtype=torch.int32),
        last_page_len=torch.tensor([1], dtype=torch.int32),
        page_size=4,
    )

    with pytest.raises(RuntimeError, match="fp16/bf16 paged KV cache"):
        backend.plan_decode_batch_paged(
            page_plan,
            num_qo_heads=2,
            num_kv_heads=1,
            head_dim=8,
            dtype=torch.float32,
        )


def test_decode_before_plan_raises(cpu_flashinfer: None) -> None:
    backend = FlashInferPagedAttention()
    queries = torch.zeros(2, 2, 8, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="call plan_decode_batch_paged before paged decode"):
        backend.forward_decode_batch_paged(queries, torch.zeros(1))


def test_decode_rejects_shape_mismatch_after_plan(cpu_flashinfer: None) -> None:
    backend = FlashInferPagedAttention()
    # indptr length B+1 == 3 -> planned batch of 2.
    page_plan = KVPagePlan(
        indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        indices=torch.tensor([0, 1], dtype=torch.int32),
        last_page_len=torch.tensor([1, 1], dtype=torch.int32),
        page_size=4,
    )
    backend.plan_decode_batch_paged(
        page_plan,
        num_qo_heads=2,
        num_kv_heads=1,
        head_dim=8,
        dtype=torch.float16,
    )
    paged_kv_cache = torch.zeros(1)

    matched = backend.forward_decode_batch_paged(
        torch.zeros(2, 2, 8, dtype=torch.float16), paged_kv_cache
    )
    assert matched.shape == (2, 2, 8)

    with pytest.raises(RuntimeError, match="does not match the planned"):
        backend.forward_decode_batch_paged(
            torch.zeros(3, 2, 8, dtype=torch.float16), paged_kv_cache
        )


class _MinimalPagedBackend:
    """Satisfies the ``PagedDecodeAttentionBackend`` protocol without a real kernel."""

    def forward(self, query, key, value):  # pragma: no cover - never called in warmup test
        raise NotImplementedError

    def forward_decode_batch_packed(self, queries, key, value, cu_seqlens_k, max_seqlen_k):
        raise NotImplementedError  # pragma: no cover

    def plan_decode_batch_paged(self, page_plan, *, num_qo_heads, num_kv_heads, head_dim, dtype):
        raise NotImplementedError  # pragma: no cover

    def forward_decode_batch_paged(self, queries, paged_kv_cache):
        raise NotImplementedError  # pragma: no cover


def test_warm_flashinfer_skips_non_paged_backend(tmp_path: Path) -> None:
    runtime = load_model_runtime("esme", bundle_path=write_tiny_pretrain_bundle(tmp_path))

    # Default CPU backend is TorchNaiveAttention (not a paged backend), so warmup is skipped
    # even when the caller claims a CUDA device.
    assert serve._warm_flashinfer_decode_if_needed(runtime, block_size=64, device="cuda") is None


def test_warm_flashinfer_uses_small_pool_for_paged_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = load_model_runtime(
        "esme",
        bundle_path=write_tiny_pretrain_bundle(tmp_path),
        attention_backend=_MinimalPagedBackend(),
    )

    recorded: dict[str, int] = {}

    class _FakeEngine:
        def __init__(self, model, *, num_blocks, **kwargs) -> None:
            recorded["num_blocks"] = num_blocks

        def add_request(self, request) -> None:
            pass

        def run(self) -> None:
            pass

    monkeypatch.setattr(serve, "InferenceEngine", _FakeEngine)
    monkeypatch.setattr(serve.torch.cuda, "synchronize", lambda: None)

    warmup_s = serve._warm_flashinfer_decode_if_needed(runtime, block_size=64, device="cuda")

    assert warmup_s is not None
    assert recorded["num_blocks"] == serve._WARMUP_NUM_BLOCKS
    assert serve._WARMUP_NUM_BLOCKS < serve.DEFAULT_NUM_BLOCKS
