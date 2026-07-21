"""On-device MPS smoke checks: the Esme bundle actually generates on mps/fp16.

Skipped unless Apple-silicon MPS is present. fp16-on-MPS output is not expected to match the
fp32 CPU reference token-for-token (that is the local chat path's documented, labeled tradeoff),
so these assert on what must hold on the device itself: generation completes with valid ids, no
NaN logits, two identical runs agree (per-config determinism), and the paged engine paths
(batching, prefix cache donate/reuse) run without breaking on MPS.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving.engine import InferenceEngine
from llm_infer.serving.request import Request

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires Apple-silicon MPS"
)

_EOS: frozenset[int] = frozenset()
_PROMPT = [4, 5, 6, 7, 8, 9, 10, 4]


def _runtime(tmp_path: Path):
    return load_model_runtime(
        "esme",
        dtype=torch.float16,
        device="mps",
        bundle_path=write_tiny_pretrain_bundle(tmp_path),
        attention_backend_name="torch_sdpa",
    )


def _engine(runtime, **kwargs) -> InferenceEngine:
    return InferenceEngine(
        runtime.model,
        block_size=4,
        num_blocks=64,
        device="mps",
        capabilities=runtime.capabilities,
        **kwargs,
    )


def test_generation_completes_with_valid_ids(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    assert str(runtime.model.device) == "mps:0"
    engine = _engine(runtime, decode_window_size=8)
    engine.add_request(Request("r", _PROMPT, 12, _EOS))
    out = engine.run()["r"]
    assert len(out) == 12
    vocab_size = runtime.model.config.vocab_size
    assert all(isinstance(t, int) and 0 <= t < vocab_size for t in out)


def test_two_identical_runs_match(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    def run() -> list[int]:
        engine = _engine(runtime, decode_window_size=8)
        engine.add_request(Request("r", _PROMPT, 12, _EOS))
        return engine.run()["r"]

    assert run() == run()


def test_logits_are_finite(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    logits = runtime.model.logits(_PROMPT)
    assert logits.dtype is torch.float16
    assert bool(torch.isfinite(logits).all())


def test_batched_decode_runs_on_mps(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    engine = _engine(runtime, decode_window_size=8)
    engine.add_request(Request("a", _PROMPT, 10, _EOS))
    engine.add_request(Request("b", [9, 8, 7, 6, 5, 4], 10, _EOS))
    out = engine.run()
    assert len(out["a"]) == 10
    assert len(out["b"]) == 10


def test_prefix_cache_donate_and_reuse_on_mps(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    engine = _engine(runtime, prefix_cache=True, decode_window_size=8)
    engine.add_request(Request("t1", _PROMPT, 8, _EOS))
    engine.run()
    engine.add_request(Request("t2", _PROMPT + [3], 8, _EOS))
    engine.run()
    # The first turn donated its block-aligned prompt prefix; the second reuses it.
    assert engine.prefix_cache_hit_tokens_total > 0
