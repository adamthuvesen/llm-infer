"""serve.py auto-default resolution, without needing an accelerator or a loaded model.

Two resolvers cover every serve auto: ``resolve_runtime_defaults`` (device, dtype, attention
backend — pre-load) and ``resolve_engine_defaults`` (speculative decode, prefix cache — they
need the loaded backend's capabilities). An explicit value always wins over an auto.
"""

from __future__ import annotations

import torch

from llm_infer.model.interface import BackendCapabilities
from llm_infer.serve import (
    describe_run_config,
    resolve_engine_defaults,
    resolve_runtime_defaults,
)

_CAPS = BackendCapabilities(
    paged_kv=True, prefix_caching=True, speculative=True, flash_attention=False
)


def _runtime(**overrides: object) -> object:
    kwargs: dict = {
        "device": "auto",
        "dtype": None,
        "attention_backend": "auto",
        "backend": "esme",
    }
    kwargs.update(overrides)
    return resolve_runtime_defaults(**kwargs)


def test_runtime_auto_is_cpu_fp16_sdpa_for_local_bundles() -> None:
    resolved = _runtime()
    # Deliberate even on an MPS box: mps measured slower than CPU at this model size.
    assert resolved.device == "cpu"
    assert resolved.dtype is torch.float16
    assert resolved.attention_backend == "torch_sdpa"


def test_runtime_explicit_values_win() -> None:
    assert _runtime(device="cuda").device == "cuda"
    assert _runtime(device="mps").device == "mps"
    assert _runtime(dtype=torch.float32).dtype is torch.float32
    assert _runtime(dtype=torch.bfloat16).dtype is torch.bfloat16
    for name in ("torch_naive", "torch_sdpa", "flash_attn", "flashinfer"):
        assert _runtime(attention_backend=name).attention_backend == name


def test_runtime_cuda_keeps_fp32_and_flashinfer_auto() -> None:
    resolved = _runtime(device="cuda")
    assert resolved.dtype is torch.float32
    # CUDA auto behavior (FlashInfer inside the loader) must not change.
    assert resolved.attention_backend == "auto"


def test_runtime_mps_opt_in_gets_fp16_and_sdpa() -> None:
    resolved = _runtime(device="mps")
    assert resolved.dtype is torch.float16
    assert resolved.attention_backend == "torch_sdpa"


def test_runtime_non_bundle_backend_keeps_auto_attention() -> None:
    assert _runtime(backend="qwen").attention_backend == "auto"


def _engine(**overrides: object) -> object:
    kwargs: dict = {
        "prompt_lookup": None,
        "prefix_cache": None,
        "device": "cpu",
        "backend": "esme",
        "capabilities": _CAPS,
    }
    kwargs.update(overrides)
    return resolve_engine_defaults(**kwargs)


def test_engine_auto_is_on_locally_off_on_cuda() -> None:
    local = _engine()
    assert local.prompt_lookup is True
    assert local.prefix_cache is True
    cuda = _engine(device="cuda")
    assert cuda.prompt_lookup is False
    assert cuda.prefix_cache is False


def test_engine_explicit_values_win() -> None:
    assert _engine(prompt_lookup=False).prompt_lookup is False
    assert _engine(prompt_lookup=True, device="cuda").prompt_lookup is True
    assert _engine(prefix_cache=True, device="cuda").prefix_cache is True


def test_engine_auto_respects_capabilities_and_backend() -> None:
    no_spec = BackendCapabilities(
        paged_kv=True, prefix_caching=True, speculative=False, flash_attention=False
    )
    assert _engine(capabilities=no_spec).prompt_lookup is False
    # Auto prefix cache is off for the non-bundle reference backend even on CPU.
    assert _engine(backend="qwen").prefix_cache is False


def test_describe_run_config_labels_reference_and_experimental() -> None:
    reference = describe_run_config("cpu", torch.float32)
    assert "reference" in reference
    assert "experimental" not in reference

    for device, dtype in (
        ("mps", torch.float16),
        ("mps", torch.float32),
        ("cpu", torch.float16),
    ):
        line = describe_run_config(device, dtype)
        assert "experimental" in line
        assert "reference is cpu fp32" in line

    assert "mps fp16" in describe_run_config("mps", torch.float16)
