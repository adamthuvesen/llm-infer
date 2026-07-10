"""Benchmark runner profiling stays outside headline timing."""

from __future__ import annotations

import torch

from llm_infer.benchmarks.runners import run_llm_infer
from llm_infer.benchmarks.workload import BenchRequest, Workload
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.profiling import TimingProfiler, attach_host_method_profile
from tests.support.fake_causal_lm import FakeCausalLMBase


class TinyModel(FakeCausalLMBase):
    """Small stand-in for QwenModel that exposes whether profiling is attached."""

    def __init__(self) -> None:
        self.num_layers = 1
        self.num_kv_heads = 1
        self.head_dim = 1
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.backend = object()
        self.profiler = None
        self.profile_flags: list[bool] = []

    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        self.profile_flags.append(self.profiler is not None)
        table.reserve(len(prompt_ids))
        table.length = len(prompt_ids)
        return _logits_for(1)

    def decode_many(
        self,
        cache: PagedKVCache,
        tables: list[BlockTable],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        self.profile_flags.append(self.profiler is not None)
        for table in tables:
            table.reserve(1)
            table.length += 1
        return _logits_for(2).unsqueeze(0).repeat(len(tables), 1)


def test_llm_infer_profile_runs_after_timed_iterations() -> None:
    model = TinyModel()
    workload = Workload(
        requests=(BenchRequest("req-0", (7, 8), "case"),),
        max_new_tokens=2,
        eos_token_ids=frozenset({99}),
        model_id="tiny",
        model_revision=None,
        source="unit",
    )

    result = run_llm_infer(
        model,
        workload,
        num_blocks=4,
        warmup=1,
        iters=2,
        device="cpu",
        collect_profile=True,
    )

    assert result.outputs == {"req-0": [1, 2]}
    assert len(result.per_iter_seconds) == 2
    assert len(result.profiles) == 1
    # warmup + 2 measured runs are unprofiled; the single extra diagnostic run is profiled.
    assert model.profile_flags == [False, False, False, False, False, False, True, True]


def test_profile_summary_projects_raw_buckets_to_phase_records() -> None:
    profiler = TimingProfiler("cpu")

    with profiler.record("prefill"):
        pass
    with profiler.record("paged_attention_plan"):
        pass
    with profiler.record("paged_attention"):
        pass
    with profiler.record("sampling"):
        pass

    summary = profiler.summary().as_dict()
    phases = summary["phases"]

    assert isinstance(phases, dict)
    assert phases["prefill"]["available"] is True
    assert phases["page_planning"]["source_buckets"] == ["paged_attention_plan"]
    assert phases["attention"]["source_buckets"] == ["paged_attention"]
    assert phases["sampling"]["calls"] == 1
    assert phases["sampling"]["mean_ms_per_call"] is not None
    assert phases["window_flushing"]["available"] is False
    assert phases["window_flushing"]["mean_ms_per_call"] is None
    assert summary["phase_timings_are_nested"] is True


def test_attach_host_method_profile_records_private_diagnostic_calls() -> None:
    class Target:
        def stage(self, value: int) -> int:
            return value + 1

        def consume(self, value: int) -> int:
            return value - 1

    target = Target()
    profiler = TimingProfiler("cpu")
    attach_host_method_profile(profiler, target, "window_flushing", ("stage", "consume"))

    assert target.stage(4) == 5
    assert target.consume(4) == 3
    phases = profiler.summary().as_dict()["phases"]

    assert isinstance(phases, dict)
    assert phases["window_flushing"]["available"] is True
    assert phases["window_flushing"]["calls"] == 2


def _logits_for(token_id: int) -> torch.Tensor:
    logits = torch.zeros(4)
    logits[token_id] = 1.0
    return logits
