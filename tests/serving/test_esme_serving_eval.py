"""Serving-eval harness tests on the tiny Esme export bundle.

These are harness checks, not new model-math checks: the tiny bundle keeps them CPU-fast while
exercising the same local engine and ASGI HTTP paths the CLI uses for Esme-214M-Chat.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm_infer.model.runtime import load_model_runtime
from scripts.esme_serving_eval import (
    EngineRequestSpec,
    EngineWorkload,
    WorkloadConfig,
    default_engine_workloads,
    default_http_workloads,
    run_asgi_http_workload,
    run_engine_workload,
)
from tests.correctness.test_pretrain_bundle import _write_tiny_bundle


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
    assert metrics["requests_completed"] == 3
    assert metrics["requests_failed"] == 0
    assert metrics["preemption_count"] > 0
    assert metrics["throughput_tokens_per_s"] is not None
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
    assert metrics["queue_time_p50_s"] is not None
    assert trace["available"] is True
    assert trace["by_event"]["decode_step"] >= 3


def test_sampled_http_workload_records_tokens_without_speed_claim(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    result = asyncio.run(
        run_asgi_http_workload(runtime, _http_workload("api-blocking-sampled"), device="cpu")
    )

    metrics = result["metrics"]
    reference = result["reference"]
    assert reference["status"] == "partial"
    assert reference["passed"] == 2
    assert reference["skipped_sampled"] == 2
    assert metrics["requests_completed"] == 4
    assert metrics["requests_failed"] == 0
    assert metrics["output_tokens"] == 24
    assert metrics["throughput_tokens_per_s"] is None
    assert metrics["throughput_status"] == "not_reported_contains_sampled_requests"
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
    assert metrics["throughput_status"] == "not_reported_requests_failed"
