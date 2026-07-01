"""Esme speculative-decode and preemption parity against the full-recompute oracle.

``test_esme_paged_parity.py`` gates paged/batched/greedy decode on the real bundle. These add the
two remaining serving techniques to that real-bundle gate: prompt-lookup **speculative decode**
and recompute **preemption** must each reproduce the exact greedy token ids that direct
``PretrainBundleModel.logits()`` greedy decode produces, on the same prompt — the repo's "match
the reference before measuring" rule applied to Esme's last two techniques.

Speculative decoding uses a prompt-lookup draft (no separate draft model), so it needs nothing
the Esme bundle does not already provide; the greedy verifier guarantees its output is identical
to the non-speculative greedy path, which is what these tests pin. The tiny synthetic bundle runs
everywhere; the real ``Esme-214M-Chat`` weights are exercised when ``ESME_BUNDLE_PATH`` is set.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as _write_tiny_bundle
from llm_infer.model.decode import greedy_decode
from llm_infer.model.pretrain_bundle import PretrainBundleModel
from llm_infer.serving import InferenceEngine, Request
from llm_infer.serving.speculative import SpeculativeDecodingConfig
from llm_infer.tracing import TraceRecorder

# A prompt with a repeated suffix so prompt-lookup actually finds a draft to verify (otherwise
# speculation degrades to the normal path and the test would not exercise verification).
_REPEATED_PROMPT = [1, 5, 9, 1, 5, 9]
_MAX_NEW = 16


def _greedy_reference(model: PretrainBundleModel, prompt: list[int], max_new: int) -> list[int]:
    return greedy_decode(model, list(prompt), max_new_tokens=max_new, eos_token_ids=set())


def _decode(
    model: PretrainBundleModel,
    prompt: list[int],
    max_new: int,
    *,
    block_size: int,
    num_blocks: int,
    speculative: SpeculativeDecodingConfig | None = None,
    preemption: bool = False,
) -> list[int]:
    engine = InferenceEngine(
        model,
        block_size=block_size,
        num_blocks=num_blocks,
        speculative=speculative,
        preemption=preemption,
    )
    engine.add_request(Request("r", list(prompt), max_new, frozenset()))
    return engine.run()["r"]


def _require_real_esme() -> PretrainBundleModel:
    bundle = os.environ.get("ESME_BUNDLE_PATH")
    if bundle is None:
        pytest.skip("set ESME_BUNDLE_PATH to run the real Esme speculative/preempt parity")
    return PretrainBundleModel.load(Path(bundle), dtype=torch.float32)


def test_esme_speculative_matches_greedy_reference_tiny(tmp_path: Path) -> None:
    model = PretrainBundleModel.load(_write_tiny_bundle(tmp_path))
    reference = _greedy_reference(model, _REPEATED_PROMPT, _MAX_NEW)
    speculative = _decode(
        model,
        _REPEATED_PROMPT,
        _MAX_NEW,
        block_size=8,
        num_blocks=64,
        speculative=SpeculativeDecodingConfig(max_draft_tokens=3, max_ngram_size=3),
    )
    assert speculative == reference


def test_esme_speculative_matches_greedy_reference_real_bundle() -> None:
    model = _require_real_esme()
    reference = _greedy_reference(model, _REPEATED_PROMPT, _MAX_NEW)
    speculative = _decode(
        model,
        _REPEATED_PROMPT,
        _MAX_NEW,
        block_size=16,
        num_blocks=16,
        speculative=SpeculativeDecodingConfig(max_draft_tokens=3, max_ngram_size=3),
    )
    assert speculative == reference


def test_esme_preemption_matches_greedy_reference_real_bundle() -> None:
    """Recompute-on-resume under a tight pool reproduces every request's uninterrupted tokens.

    Three concurrent requests with footprint 1 block each are all admitted into a 3-block pool,
    then must evict and recompute as they decode past block boundaries — a real preemption on the
    real weights. Each preempted output must still equal its uninterrupted greedy reference.
    """
    model = _require_real_esme()
    prompts = {"a": [1, 5, 9, 13], "b": [2, 6, 10, 14], "c": [3, 7, 11, 15]}
    max_new = 8
    references = {rid: _greedy_reference(model, p, max_new) for rid, p in prompts.items()}

    recorder = TraceRecorder()
    engine = InferenceEngine(model, block_size=4, num_blocks=3, preemption=True, trace=recorder)
    for rid, prompt in prompts.items():
        engine.add_request(Request(rid, list(prompt), max_new, frozenset()))
    outputs = engine.run()

    # Prove the tight pool actually preempted — otherwise the parity check is vacuous.
    assert any(event.event == "request_preempted" for event in recorder.events), (
        "tight pool must force a real preemption for this parity check to mean anything"
    )
    for rid, expected in references.items():
        assert outputs[rid] == expected, f"{rid}: preempted output diverged from greedy reference"
