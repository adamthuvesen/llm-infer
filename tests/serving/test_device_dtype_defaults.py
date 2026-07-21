"""Device/dtype auto-resolution for serve.py, without needing a real accelerator.

``auto`` resolves to CPU + fp32 — MPS measured slower than CPU for the 214M bundle, so it is
explicit opt-in only (fp16 then follows the MPS device). An explicit value always wins. The
actual on-device run lives in test_mps_serving.py.
"""

from __future__ import annotations

import torch

from llm_infer.serve import (
    describe_run_config,
    resolve_device_default,
    resolve_dtype_default,
)


def test_device_auto_resolves_to_cpu() -> None:
    # Deliberate even on an MPS box: measured slower than CPU at this model size.
    assert resolve_device_default("auto") == "cpu"


def test_explicit_device_wins_over_auto() -> None:
    assert resolve_device_default("cpu") == "cpu"
    assert resolve_device_default("cuda") == "cuda"
    assert resolve_device_default("mps") == "mps"


def test_dtype_auto_is_fp16_locally_fp32_on_cuda() -> None:
    assert resolve_dtype_default(None, device="mps") is torch.float16
    assert resolve_dtype_default(None, device="cpu") is torch.float16
    assert resolve_dtype_default(None, device="cuda") is torch.float32


def test_explicit_dtype_wins() -> None:
    assert resolve_dtype_default(torch.float32, device="mps") is torch.float32
    assert resolve_dtype_default(torch.bfloat16, device="cpu") is torch.bfloat16


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
