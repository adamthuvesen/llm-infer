"""Serving-eval harness tests on the tiny Esme export bundle.

These are harness checks, not new model-math checks: the tiny bundle keeps them CPU-fast while
exercising the same local engine and ASGI HTTP paths the CLI uses for Esme-214M-Chat.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as _write_tiny_bundle
from llm_infer.model.runtime import load_model_runtime
from scripts.esme_serving_eval import (
    EngineRequestSpec,
    EngineWorkload,
    WorkloadConfig,
    default_engine_workloads,
    default_http_workloads,
    phase0_http_workloads,
    run_asgi_http_workload,
    run_engine_workload,
    run_network_http_workload,
)


def _runtime(tmp_path: Path):
    return load_model_runtime("esme", bundle_path=_write_tiny_bundle(tmp_path))


def _engine_workload(name: str):
    return next(workload for workload in default_engine_workloads() if workload.name == name)


def _http_workload(name: str):
    return next(workload for workload in default_http_workloads() if workload.name == name)


def test_tight_kv_workload_reports_reference_gated_preemption(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = asyncio.run(
        run_engine_workload(runtime, _engine_workload("tight-kv-preemption"), device="cpu")
    )

    metrics = result["metrics"]
    assert result["surface"] == "engine"
    assert result["reference"]["status"] == "pass"
    assert result["policy_version"] == 2
    assert result["reference_status"] == "exact"
    assert result["parity_status"] == "not_applicable"
    assert result["headline_eligible"] is True
    assert metrics["requests_completed"] == 3
    assert metrics["requests_failed"] == 0
    assert metrics["preemption_count"] > 0
    assert metrics["throughput_tokens_per_s"] is not None
    assert metrics["raw_throughput_tokens_per_s"] is not None
    assert metrics["throughput_status"] == "reported"
    assert metrics["queue_time_p50_s"] is not None


def test_shared_prefix_workload_engages_prefix_cache(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = asyncio.run(
        run_engine_workload(runtime, _engine_workload("shared-prefix-greedy"), device="cpu")
    )

    assert result["reference"]["status"] == "pass"
    assert result["metrics"]["throughput_status"] == "reported"
    # One shared prompt group should prefill once, then fork the cached block table to siblings.
    # If prefix_group_id is dropped before Request construction, this count becomes 4.
    assert result["trace"]["by_event"]["prefill_chunk_started"] == 1


def test_asgi_streaming_workload_records_http_and_reference_metrics(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = asyncio.run(
        run_asgi_http_workload(runtime, _http_workload("api-streaming-mixed"), device="cpu")
    )

    metrics = result["metrics"]
    trace = result["trace"]
    assert result["surface"] == "asgi-http"
    assert result["reference"]["status"] == "pass"
    assert metrics["requests_completed"] == 3
    assert metrics["requests_failed"] == 0
    assert metrics["output_tokens"] == 18
    assert metrics["public_stream_tokens"] == 18
    assert metrics["public_requests_admitted"] == 3
    assert metrics["public_queue_time_count"] == 3
    assert metrics["public_queue_time_avg_s"] is not None
    assert metrics["throughput_tokens_per_s"] is not None
    assert metrics["ttft_p50_s"] is not None
    assert metrics["ttft_p95_s"] is not None
    assert metrics["itl_p50_s"] is not None
    assert metrics["itl_p95_s"] is not None
    assert metrics["queue_time_p50_s"] is not None
    assert metrics["queue_time_p95_s"] is not None
    assert result["timing_scope"]["engine_build_s"] >= 0
    assert result["timing_scope"]["server_start_s"] >= 0
    assert result["timing_scope"]["steady_state_wall_s"] == result["wall_s"]
    assert trace["available"] is True
    assert trace["by_event"]["decode_step"] >= 3


def test_network_streaming_workload_uses_uvicorn_and_keeps_reference_gate(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    workload, _ = phase0_http_workloads(1, max_new_tokens=4, block_size=8, num_blocks=16)

    result = asyncio.run(
        run_network_http_workload(runtime, workload, device="cpu", warmup_runs=1, measured_runs=3)
    )

    assert result["surface"] == "network-http"
    assert result["reference"]["status"] == "pass"
    assert result["reference"]["passed"] == 3
    assert result["metrics"]["requests_total"] == 3
    assert result["metrics"]["public_requests_admitted"] == 3
    assert result["metrics"]["throughput_tokens_per_s"] is not None
    assert result["metrics"]["ttft_p50_s"] < result["metrics"]["latency_p50_s"]
    assert result["timing_scope"]["transport"] == "localhost Uvicorn TCP"
    assert result["timing_scope"]["warmup_runs"] == 1
    assert result["timing_scope"]["measured_runs"] == 3
    assert result["timing_scope"]["max_connections"] == 1
    assert result["trace"] == {
        "available": False,
        "reason": "disabled because tracing changes deferred decode-window behavior",
    }


def test_sampled_http_workload_is_reference_gated(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = asyncio.run(
        run_asgi_http_workload(runtime, _http_workload("api-blocking-sampled"), device="cpu")
    )

    metrics = result["metrics"]
    reference = result["reference"]
    assert reference["status"] == "pass"
    assert reference["passed"] == 4
    assert reference["skipped_sampled"] == 0
    assert metrics["requests_completed"] == 4
    assert metrics["requests_failed"] == 0
    assert metrics["output_tokens"] == 24
    assert metrics["throughput_tokens_per_s"] is not None
    assert metrics["throughput_status"] == "reported"
    assert metrics["observed_output_tokens_per_s"] is not None


def test_failed_submission_suppresses_throughput(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    workload = EngineWorkload(
        name="failed-submission",
        description="One valid request plus one request too large for the KV pool.",
        config=WorkloadConfig(block_size=4, num_blocks=2),
        requests=(
            EngineRequestSpec("ok", (1, 4, 7), 2),
            EngineRequestSpec("too-large", (1, 2, 3, 4, 5, 6, 7, 8, 9), 4),
        ),
    )

    result = asyncio.run(run_engine_workload(runtime, workload, device="cpu"))

    metrics = result["metrics"]
    assert result["reference"]["status"] == "pass"
    assert metrics["requests_completed"] == 1
    assert metrics["requests_failed"] == 1
    assert metrics["observed_requests"] == 1
    assert metrics["unobserved_requests"] == 1
    assert metrics["throughput_tokens_per_s"] is None
    assert metrics["raw_throughput_tokens_per_s"] is not None
    assert result["headline_eligible"] is False
    assert metrics["throughput_status"] == "not_reported_requests_failed"


def test_phase0_http_workloads_pair_identical_greedy_and_sampled_shapes() -> None:
    greedy, sampled = phase0_http_workloads(8, max_new_tokens=32)

    assert len(greedy.requests) == len(sampled.requests) == 8
    assert all(request.stream for request in (*greedy.requests, *sampled.requests))
    assert all(request.sampling.is_greedy for request in greedy.requests)
    assert all(not request.sampling.is_greedy for request in sampled.requests)
    assert {request.prompt for request in greedy.requests} == {
        request.prompt for request in sampled.requests
    }


def _sampled_record(runtime, request_id: str, token_ids: list[int]):
    from llm_infer.serving.sampler import SamplingParams
    from scripts.esme_serving_eval import ObservedEngineRequest, RequestSignature

    sampling = SamplingParams(temperature=0.8, top_p=0.95, seed=17)
    prompt_ids = runtime.tokenizer.encode("replay gate prompt")
    record = ObservedEngineRequest(
        request_id=request_id,
        signature=RequestSignature.from_parts(prompt_ids, 4, sampling),
        sampling=sampling,
        arrival_s=0.0,
    )
    record.token_ids = list(token_ids)
    return record


def test_sampled_replay_consistent_but_anchor_divergent_is_accepted(tmp_path: Path) -> None:
    """Identical sampled records that differ from the single-request anchor still pass.

    This is the batched-sampled case: bf16 batch-shape numerics fork a seeded continuation,
    so the gate is replay consistency; the anchor divergence is recorded as evidence.
    """
    from scripts.esme_serving_eval import _reference_summary

    runtime = _runtime(tmp_path)
    stream = [5, 9, 2, 4]  # deliberately not what the anchor engine produces
    records = [_sampled_record(runtime, f"req-{index}", stream) for index in range(3)]

    summary = _reference_summary(runtime, records, block_size=8)

    assert summary["status"] == "pass"
    assert summary["passed"] == 3
    assert summary["sampled_replay_divergent_groups"] == 0
    statuses = {detail.get("status") for detail in summary["details"]}
    assert "pass_replay" in statuses
    assert "replay_anchor_note" in statuses


def test_sampled_replay_divergence_sends_row_to_review(tmp_path: Path) -> None:
    """Sampled records of one signature that disagree with each other need review."""
    from scripts.esme_serving_eval import _reference_summary, _speed_status

    runtime = _runtime(tmp_path)
    records = [
        _sampled_record(runtime, "req-0", [5, 9, 2, 4]),
        _sampled_record(runtime, "req-1", [5, 9, 2, 4]),
        _sampled_record(runtime, "req-2", [5, 9, 7, 1]),
    ]

    summary = _reference_summary(runtime, records, block_size=8)

    assert summary["status"] == "review"
    assert summary["sampled_replay_divergent_groups"] == 1
    divergent = next(d for d in summary["details"] if d.get("status") == "replay_divergent")
    assert divergent["distinct_streams"] == 2
    assert (
        _speed_status(summary["status"], requests_total=3, requests_completed=3, observed_count=3)
        == "not_reported_reference_review_required"
    )
