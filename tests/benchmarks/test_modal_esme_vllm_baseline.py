from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_infer.benchmarks.esme_paged import EsmeBenchRequest
from llm_infer.benchmarks.esme_three_way import EsmeAgreement, vllm_decode_closure
from scripts.modal_esme_vllm_baseline import (
    VLLM_PYTHON,
    _command_config,
    build_context_workloads,
    finalize_row,
    gate_outputs,
    validate_child_identity,
    worker_command,
)


class _Tokenizer:
    def apply_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool, tokenize: bool
    ) -> list[int]:
        assert messages and add_generation_prompt and tokenize
        return [11, 12, 13]


def test_full_matrix_builds_exact_context_lengths_and_batches() -> None:
    workloads = build_context_workloads(_Tokenizer(), (1, 8, 64, 256), (32, 256, 768))

    assert len(workloads) == 12
    assert {(row["batch_size"], row["context_tokens"]) for row in workloads} == {
        (batch, context) for batch in (1, 8, 64, 256) for context in (32, 256, 768)
    }
    for workload in workloads:
        assert len(workload["requests"]) == workload["batch_size"]
        assert all(
            len(request["prompt_ids"]) == workload["context_tokens"]
            for request in workload["requests"]
        )
        assert (
            len({request["request_id"] for request in workload["requests"]})
            == workload["batch_size"]
        )


def test_full_command_pins_phase_zero_protocol() -> None:
    batches, contexts, max_new_tokens, warmup, iters = _command_config("full")

    assert batches == (1, 8, 64, 256)
    assert contexts == (32, 256, 768)
    assert max_new_tokens == 128
    assert warmup == 2
    assert iters == 10


def test_vllm_ignore_eos_does_not_keep_eos_as_custom_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=SamplingParams))
    requests = [EsmeBenchRequest("r0", "prompt", (1, 2, 3))]

    vllm_decode_closure(
        object(),
        requests,
        max_new_tokens=8,
        eos_token_ids=frozenset({2}),
        ignore_eos=True,
    )

    assert captured["ignore_eos"] is True
    assert captured["stop_token_ids"] == []


def test_worker_command_launches_this_script_in_a_fresh_python_process(tmp_path: Path) -> None:
    command = worker_command("vllm", tmp_path / "input.json", tmp_path / "output.json")

    assert command[0] == VLLM_PYTHON
    assert Path(command[1]).name == "modal_esme_vllm_baseline.py"
    assert command[2:] == [
        "--worker",
        "vllm",
        "--input",
        str(tmp_path / "input.json"),
        "--output",
        str(tmp_path / "output.json"),
    ]

    oracle_command = worker_command("oracle", tmp_path / "input.json", tmp_path / "output.json")
    assert oracle_command[0] == sys.executable


def test_same_host_validation_requires_distinct_pid_and_matching_gpu() -> None:
    parent = {"pid": 10, "hostname": "worker-a", "gpu_uuid": "GPU-123"}
    validate_child_identity(
        parent,
        {"pid": 11, "hostname": "worker-a", "gpu_uuid": "GPU-123"},
        "vllm",
    )

    with pytest.raises(AssertionError, match="child process"):
        validate_child_identity(parent, dict(parent), "vllm")
    with pytest.raises(AssertionError, match="reserved host"):
        validate_child_identity(
            parent,
            {"pid": 11, "hostname": "worker-b", "gpu_uuid": "GPU-123"},
            "vllm",
        )
    with pytest.raises(AssertionError, match="reserved host"):
        validate_child_identity(
            parent,
            {"pid": 11, "hostname": "worker-a", "gpu_uuid": "GPU-456"},
            "vllm",
        )


def _typed_agreement(*, exact: int = 0, tie: int = 0, failed: int = 0) -> EsmeAgreement:
    return EsmeAgreement(
        exact=exact,
        tie=tie,
        nontie=failed,
        total=exact + tie + failed,
        ties_sample=[],
        divergences_sample=[],
        failed=failed,
    )


def test_failed_reference_gate_suppresses_throughput() -> None:
    worker_row = {
        "batch_size": 1,
        "context_tokens": 32,
        "per_iter_seconds": [1.0, 2.0, 3.0],
        "outputs": {"r0": [4, 5]},
    }
    passed = finalize_row(
        worker_row,
        system="vllm",
        agreement=_typed_agreement(exact=1),
        total_output_tokens=6,
    )
    failed = finalize_row(
        worker_row,
        system="vllm",
        agreement=_typed_agreement(failed=1),
        total_output_tokens=6,
    )

    assert passed["median_seconds"] == 2.0
    assert passed["p95_seconds"] == 3.0
    assert passed["policy_version"] == 2
    assert passed["reference_status"] == "exact"
    assert passed["parity_status"] == "not_applicable"
    assert passed["raw_tokens_per_second"] == 3.0
    assert passed["headline_eligible"] is True
    assert passed["tokens_per_second"] == 3.0
    assert passed["agreement"]["exact"] == 1
    assert failed["reference_status"] == "failed"
    assert failed["raw_tokens_per_second"] == 3.0
    assert failed["headline_eligible"] is False
    assert failed["tokens_per_second"] is None


def test_serialized_fp32_gate_accepts_only_recorded_near_max_tokens() -> None:
    request = {"request_id": "r0", "prompt": "prompt", "prompt_ids": [3, 4]}
    oracle_case = {
        "context_tokens": 2,
        "output_tokens": [7, 2],
        "tie_tolerance": 0.1,
        "steps": [
            {
                "token_id": 7,
                "max_logit": 10.0,
                "top2_gap": 0.02,
                "near_token_logits": {"7": 10.0, "9": 9.98},
            },
            {
                "token_id": 2,
                "max_logit": 8.0,
                "top2_gap": 1.0,
                "near_token_logits": {"2": 8.0},
            },
        ],
    }

    exact = gate_outputs([request], {"r0": [7, 2]}, oracle_case)
    tie = gate_outputs([request], {"r0": [9, 4]}, oracle_case)
    nontie = gate_outputs([request], {"r0": [8, 4]}, oracle_case)

    assert (exact.exact, exact.tie, exact.nontie) == (1, 0, 0)
    assert (tie.exact, tie.tie, tie.nontie) == (0, 1, 0)
    assert (nontie.exact, nontie.tie, nontie.nontie) == (0, 0, 1)
    assert nontie.review_required == 1
    assert nontie.failed == 0

    tie_row = finalize_row(
        {
            "batch_size": 1,
            "context_tokens": 32,
            "per_iter_seconds": [2.0],
            "outputs": {"r0": [9, 4]},
        },
        system="vllm",
        agreement=tie,
        total_output_tokens=2,
    )
    assert tie_row["reference_status"] == "accepted_numerical"
    assert tie_row["headline_eligible"] is True
    assert tie_row["tokens_per_second"] == 1.0

    reviewed_row = finalize_row(
        {
            "batch_size": 1,
            "context_tokens": 32,
            "per_iter_seconds": [2.0],
            "outputs": {"r0": [8, 4]},
        },
        system="vllm",
        agreement=nontie,
        total_output_tokens=2,
    )
    assert reviewed_row["reference_status"] == "review_required"
    assert reviewed_row["raw_tokens_per_second"] == 1.0
    assert reviewed_row["tokens_per_second"] is None
