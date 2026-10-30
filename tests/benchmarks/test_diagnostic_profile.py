"""Exercise owner-provided window diagnostics against a real CPU engine."""

from __future__ import annotations

from pathlib import Path

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle
from llm_infer.model.decode import greedy_decode
from llm_infer.model.runtime import load_model_runtime
from llm_infer.profiling import TimingProfiler
from llm_infer.serving import InferenceEngine, Request


def test_window_diagnostics_preserve_output_and_record_both_phases(
    tmp_path: Path,
) -> None:
    runtime = load_model_runtime("esme", bundle_path=write_tiny_pretrain_bundle(tmp_path))
    prompt = [1, 4, 7]
    reference = greedy_decode(runtime.model, prompt, max_new_tokens=10, eos_token_ids=set())
    profiler = TimingProfiler("cpu")
    engine = InferenceEngine(
        runtime.model,
        block_size=8,
        num_blocks=16,
        capabilities=runtime.capabilities,
        decode_window_size=4,
        profiler=profiler,
    )
    engine.add_request(Request("r0", prompt, 10, frozenset()))
    assert engine.run() == {"r0": reference}
    bucket = next(
        bucket for bucket in profiler.summary().buckets if bucket.name == "window_flushing"
    )
    assert bucket.calls == 6  # Three windows, each staged and consumed once.
