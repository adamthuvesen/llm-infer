"""The fused-elementwise switch routes the bundle model through compiled kernels and back."""

from __future__ import annotations

from pathlib import Path

import pytest

import llm_infer.model.decode_graph as decode_graph_module
import llm_infer.model.pretrain_bundle as bundle_module
from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.model import rope_utils
from llm_infer.model.fused_elementwise import use_fused_elementwise
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serve import build_app_from_runtime


def test_switch_swaps_every_call_site_and_restores_the_originals() -> None:
    try:
        use_fused_elementwise(True)
        assert bundle_module.rms_norm is not rope_utils.rms_norm
        assert bundle_module.apply_rope is not rope_utils.apply_rope
        assert decode_graph_module.rms_norm is bundle_module.rms_norm
    finally:
        use_fused_elementwise(False)
    assert bundle_module.rms_norm is rope_utils.rms_norm
    assert bundle_module.apply_rope is rope_utils.apply_rope
    assert decode_graph_module.rms_norm is rope_utils.rms_norm


def test_server_rejects_fused_elementwise_off_cuda(tmp_path: Path) -> None:
    runtime = load_model_runtime("esme", bundle_path=write_tiny_pretrain_bundle(tmp_path))
    with pytest.raises(ValueError, match="CUDA bundle model"):
        build_app_from_runtime(runtime, block_size=4, num_blocks=64, fused_elementwise=True)
    assert bundle_module.rms_norm is rope_utils.rms_norm
