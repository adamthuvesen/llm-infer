"""Same-host A100 comparison: llm_infer and vLLM on the converted Esme checkpoint.

The older Esme comparison starts two Modal GPU functions, so its vLLM and ``llm_infer`` rows
can land on different A100 hosts. This Phase 0 harness reserves one A100 and launches each
system as a fresh child process inside that function. Process exit releases vLLM's CUDA memory,
while the parent keeps the host reservation. Hostname and GPU UUID are recorded for the parent,
the fp32 oracle, and both engine children and must match before a result is accepted.

Every speed row is gated against the fp32 ``PretrainBundleModel.logits()`` oracle. Exact output
matches pass. A first divergence passes only when the fp32 top-two gap is within the audited
Esme bf16 tolerance used by the existing three-way benchmark. A non-tie divergence leaves
``tokens_per_second`` unset.

The full command runs the Phase 0 matrix: batch 1/8/64/256, context 32/256/768, 128 decode
tokens, two warmups, and ten measured iterations. ``focused`` keeps the full timing protocol at
batch 1 and 8; ``smoke`` only checks the remote wiring.

    modal run scripts/modal_esme_vllm_baseline.py --command smoke
    modal run scripts/modal_esme_vllm_baseline.py --command focused
    modal run scripts/modal_esme_vllm_baseline.py --command full
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, TypedDict

import modal

if TYPE_CHECKING:
    from llm_infer.benchmarks.esme_three_way import EsmeAgreement

from scripts.modal_esme_bundle import (
    ESME_BUNDLE_MOUNT,
    REMOTE_BUNDLE_PATH,
    VOLUME_NAME,
    local_bundle_path,
    stage_bundle,
)
from scripts.modal_flash_image import FLASH_IMAGE, REMOTE_ROOT, REPO_ROOT

ESME_HF_DIR = "esme-214m-chat-hf-phase0"
REMOTE_HF_PATH = f"{ESME_BUNDLE_MOUNT}/{ESME_HF_DIR}"
CAPTURE_SIZES = (1, 8, 64, 256)
BLOCK_SIZE = 128
FULL_BATCH_SIZES = (1, 8, 64, 256)
CONTEXT_LENGTHS = (32, 256, 768)
MAX_NEW_TOKENS = 128
WARMUPS = 2
MEASURED_ITERS = 10
VLLM_PYTHON = "/opt/vllm/bin/python"


class RequestSpec(TypedDict):
    request_id: str
    prompt: str
    prompt_ids: list[int]


class WorkloadSpec(TypedDict):
    batch_size: int
    context_tokens: int
    requests: list[RequestSpec]


class WorkerInput(TypedDict):
    bundle_path: str
    hf_checkpoint: str
    workloads: list[WorkloadSpec]
    max_new_tokens: int
    warmup: int
    iters: int
    block_size: int
    capture_sizes: list[int]
    ignore_eos: bool


class WorkerRow(TypedDict):
    batch_size: int
    context_tokens: int
    per_iter_seconds: list[float]
    outputs: dict[str, list[int]]


class ProcessIdentity(TypedDict):
    pid: int
    hostname: str
    gpu_uuid: str


class WorkerResult(TypedDict):
    system: str
    identity: ProcessIdentity
    rows: list[WorkerRow]
    versions: dict[str, str]
    model_startup_seconds: float
    engine_startup_seconds: float | None
    graph_capture_seconds: float | None
    attention_backend: str | None
    vllm_config: NotRequired[dict[str, object]]


class OracleStep(TypedDict):
    token_id: int
    max_logit: float
    top2_gap: float
    near_token_logits: dict[str, float]


class OracleCase(TypedDict):
    context_tokens: int
    output_tokens: list[int]
    steps: list[OracleStep]
    tie_tolerance: float


class OracleResult(TypedDict):
    system: str
    identity: ProcessIdentity
    cases: dict[str, OracleCase]
    model_startup_seconds: float
    generation_seconds: float


app = modal.App("llm-infer-esme-vllm-baseline")

# Keep two Python environments on one host. The base environment is the exact pinned
# llm_infer/FlashInfer stack used by the headline. The vLLM environment installs the current
# release and its own pinned Torch/CUDA stack. Sequential child processes select the matching
# interpreter, so neither engine rewrites the other's dependencies or retains CUDA memory.
comparison_image = FLASH_IMAGE.run_commands(
    f"python -m venv {VLLM_PYTHON.rsplit('/', 2)[0]}",
    f"{VLLM_PYTHON} -m pip install vllm modal==1.5.0",
    f"{VLLM_PYTHON} -m pip install --no-deps -e {REMOTE_ROOT}",
).env(
    {
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    }
)

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def build_context_workloads(
    tokenizer: object,
    batch_sizes: tuple[int, ...],
    context_lengths: tuple[int, ...],
) -> list[WorkloadSpec]:
    """Build deterministic requests with exactly the requested cached-token lengths.

    Each context length uses one chat prompt repeated across its batch. Prefix caching is disabled
    in both systems, so repetition does not introduce a cache hit. Keeping one unique prompt per
    context also limits the expensive fp32 oracle to three continuations for the full matrix.
    """
    from llm_infer.benchmarks.esme_paged import HEADLINE_PROMPTS

    if not batch_sizes or any(batch < 1 for batch in batch_sizes):
        raise ValueError(f"batch sizes must be positive; got {batch_sizes!r}")
    if not context_lengths or any(length < 1 for length in context_lengths):
        raise ValueError(f"context lengths must be positive; got {context_lengths!r}")

    prompts_by_length: dict[int, tuple[str, list[int]]] = {}
    for index, context_tokens in enumerate(context_lengths):
        prompt = HEADLINE_PROMPTS[index % len(HEADLINE_PROMPTS)]
        tokenized = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if not isinstance(tokenized, list) or not tokenized:
            raise ValueError(f"tokenizer returned invalid ids for context {context_tokens}")
        base_ids = [int(token_id) for token_id in tokenized]
        repeats = (context_tokens + len(base_ids) - 1) // len(base_ids)
        prompts_by_length[context_tokens] = (prompt, (base_ids * repeats)[:context_tokens])

    workloads: list[WorkloadSpec] = []
    for context_tokens in context_lengths:
        prompt, prompt_ids = prompts_by_length[context_tokens]
        for batch_size in batch_sizes:
            requests: list[RequestSpec] = []
            for request_index in range(batch_size):
                requests.append(
                    {
                        "request_id": (
                            f"esme-c{context_tokens}-b{batch_size}-r{request_index:03d}"
                        ),
                        "prompt": prompt,
                        "prompt_ids": list(prompt_ids),
                    }
                )
            workloads.append(
                {
                    "batch_size": batch_size,
                    "context_tokens": context_tokens,
                    "requests": requests,
                }
            )
    return workloads


def percentile(values: list[float], percentile_value: float) -> float:
    """Nearest-rank percentile for small benchmark samples."""
    if not values:
        raise ValueError("cannot compute a percentile of an empty sample")
    if not 0 < percentile_value <= 100:
        raise ValueError(f"percentile must be in (0, 100]; got {percentile_value}")
    ordered = sorted(values)
    rank = max(1, (len(ordered) * int(percentile_value) + 99) // 100)
    return ordered[min(rank, len(ordered)) - 1]


def _gpu_uuid() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    return result.stdout.splitlines()[0].strip()


def process_identity() -> ProcessIdentity:
    return {"pid": os.getpid(), "hostname": socket.gethostname(), "gpu_uuid": _gpu_uuid()}


def validate_child_identity(parent: ProcessIdentity, child: ProcessIdentity, system: str) -> None:
    """Reject a result that did not come from a distinct process on this exact GPU host."""
    if child["pid"] == parent["pid"]:
        raise AssertionError(
            f"{system} must run in a child process, but pid={child['pid']} matches"
        )
    for field in ("hostname", "gpu_uuid"):
        if child[field] != parent[field]:
            raise AssertionError(
                f"{system} did not run on the reserved host: "
                f"parent {field}={parent[field]!r}, child={child[field]!r}"
            )


def worker_command(system: str, input_path: Path, output_path: Path) -> list[str]:
    if system not in {"oracle", "llm_infer", "vllm"}:
        raise ValueError(f"unknown worker system {system!r}")
    return [
        VLLM_PYTHON if system == "vllm" else sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        system,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]


def _run_child_process(
    system: str, worker_input: WorkerInput, directory: Path
) -> dict[str, object]:
    """Run one CUDA owner to completion in a child, then release its full reservation."""
    input_path = directory / f"{system}-input.json"
    output_path = directory / f"{system}-output.json"
    input_path.write_text(json.dumps(worker_input), encoding="utf-8")
    result = subprocess.run(
        worker_command(system, input_path, output_path),
        capture_output=True,
        text=True,
        timeout=2 * 60 * 60,
        check=False,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-4000:]
        stdout_tail = result.stdout[-4000:]
        raise RuntimeError(
            f"{system} child failed with exit {result.returncode}\n"
            f"stdout tail:\n{stdout_tail}\nstderr tail:\n{stderr_tail}"
        )
    if not output_path.is_file():
        raise RuntimeError(f"{system} child exited successfully without writing {output_path}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_oracle_process(worker_input: WorkerInput, directory: Path) -> OracleResult:
    return _run_child_process("oracle", worker_input, directory)


def run_engine_process(system: str, worker_input: WorkerInput, directory: Path) -> WorkerResult:
    if system not in {"llm_infer", "vllm"}:
        raise ValueError(f"engine system must be 'llm_infer' or 'vllm'; got {system!r}")
    return _run_child_process(system, worker_input, directory)


def _run_oracle_worker(worker_input: WorkerInput) -> OracleResult:
    import torch

    from llm_infer.benchmarks.esme_three_way import BF16_AGREEMENT_TOLERANCE
    from llm_infer.model.runtime import load_model_runtime

    assert torch.cuda.is_available(), "fp32 oracle worker needs CUDA"
    load_start = time.perf_counter()
    runtime = load_model_runtime(
        "esme",
        bundle_path=Path(worker_input["bundle_path"]),
        dtype=torch.float32,
        device="cuda",
    )
    torch.cuda.synchronize()
    model_startup_seconds = time.perf_counter() - load_start

    unique_requests: dict[int, RequestSpec] = {}
    for workload in worker_input["workloads"]:
        representative = workload["requests"][0]
        prior = unique_requests.setdefault(workload["context_tokens"], representative)
        if prior["prompt_ids"] != representative["prompt_ids"]:
            raise AssertionError(
                f"context {workload['context_tokens']} has more than one oracle prompt"
            )

    generation_start = time.perf_counter()
    cases: dict[str, OracleCase] = {}
    tolerance = BF16_AGREEMENT_TOLERANCE
    for context_tokens, request in unique_requests.items():
        tokens = list(request["prompt_ids"])
        output_tokens: list[int] = []
        steps: list[OracleStep] = []
        for _ in range(worker_input["max_new_tokens"]):
            logits = runtime.model.logits(tokens)[-1].float()
            max_logit_tensor = logits.max()
            near_indices = torch.nonzero(
                max_logit_tensor - logits <= tolerance, as_tuple=False
            ).flatten()
            next_token = int(torch.argmax(logits).item())
            top2 = torch.topk(logits, 2).values
            steps.append(
                {
                    "token_id": next_token,
                    "max_logit": float(max_logit_tensor.item()),
                    "top2_gap": float((top2[0] - top2[1]).item()),
                    "near_token_logits": {
                        str(int(token_id)): float(logits[int(token_id)].item())
                        for token_id in near_indices.tolist()
                    },
                }
            )
            output_tokens.append(next_token)
            tokens.append(next_token)
            if not worker_input["ignore_eos"] and next_token in runtime.eos_token_ids:
                break
        cases[str(context_tokens)] = {
            "context_tokens": context_tokens,
            "output_tokens": output_tokens,
            "steps": steps,
            "tie_tolerance": tolerance,
        }
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    return {
        "system": "oracle",
        "identity": process_identity(),
        "cases": cases,
        "model_startup_seconds": model_startup_seconds,
        "generation_seconds": generation_seconds,
    }


def _run_llm_infer_worker(worker_input: WorkerInput) -> WorkerResult:
    import torch

    from llm_infer.benchmarks.report import library_versions
    from llm_infer.kernels.flashinfer_paged import FlashInferPagedAttention
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "llm_infer worker needs CUDA"
    load_start = time.perf_counter()
    runtime = load_model_runtime(
        "esme",
        bundle_path=Path(worker_input["bundle_path"]),
        dtype=torch.bfloat16,
        device="cuda",
    )
    torch.cuda.synchronize()
    model_startup_seconds = time.perf_counter() - load_start
    if not isinstance(runtime.model.backend, FlashInferPagedAttention):
        raise AssertionError(
            "same-host comparison requires llm_infer's FlashInfer path; "
            f"resolved {type(runtime.model.backend).__name__}"
        )

    graph_capture_seconds = enable_decode_graphs_if_cuda(
        runtime.model, tuple(worker_input["capture_sizes"])
    )
    largest_batch = max(workload["batch_size"] for workload in worker_input["workloads"])
    longest_context = max(workload["context_tokens"] for workload in worker_input["workloads"])
    blocks_per_request = (
        longest_context + worker_input["max_new_tokens"] + worker_input["block_size"] - 1
    ) // worker_input["block_size"]
    num_blocks = largest_batch * blocks_per_request + largest_batch
    engine_start = time.perf_counter()
    engine = InferenceEngine(
        runtime.model,
        block_size=worker_input["block_size"],
        num_blocks=num_blocks,
        device="cuda",
        capabilities=runtime.capabilities,
    )
    torch.cuda.synchronize()
    engine_startup_seconds = time.perf_counter() - engine_start

    rows: list[WorkerRow] = []
    for workload in worker_input["workloads"]:
        per_iter_seconds: list[float] = []
        final_outputs: dict[str, list[int]] = {}
        repetitions = worker_input["warmup"] + worker_input["iters"]
        for repetition in range(repetitions):
            id_map: dict[str, str] = {}
            for request_spec in workload["requests"]:
                worker_id = f"{request_spec['request_id']}-iteration-{repetition}"
                id_map[worker_id] = request_spec["request_id"]
                engine.add_request(
                    Request(
                        worker_id,
                        list(request_spec["prompt_ids"]),
                        worker_input["max_new_tokens"],
                        frozenset() if worker_input["ignore_eos"] else runtime.eos_token_ids,
                    )
                )
            torch.cuda.synchronize()
            start = time.perf_counter()
            raw_outputs = engine.run()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            outputs = {id_map[request_id]: ids for request_id, ids in raw_outputs.items()}
            if repetition >= worker_input["warmup"]:
                per_iter_seconds.append(elapsed)
                final_outputs = outputs
        rows.append(
            {
                "batch_size": workload["batch_size"],
                "context_tokens": workload["context_tokens"],
                "per_iter_seconds": per_iter_seconds,
                "outputs": final_outputs,
            }
        )
    return {
        "system": "llm_infer",
        "identity": process_identity(),
        "rows": rows,
        "versions": library_versions(),
        "model_startup_seconds": model_startup_seconds,
        "engine_startup_seconds": engine_startup_seconds,
        "graph_capture_seconds": graph_capture_seconds,
        "attention_backend": type(runtime.model.backend).__name__,
    }


def _run_vllm_worker(worker_input: WorkerInput) -> WorkerResult:
    from llm_infer.benchmarks.esme_paged import EsmeBenchRequest
    from llm_infer.benchmarks.esme_three_way import build_vllm_llm, vllm_decode_closure
    from llm_infer.benchmarks.report import library_versions

    max_model_len = max(
        workload["context_tokens"] + worker_input["max_new_tokens"]
        for workload in worker_input["workloads"]
    )
    build_start = time.perf_counter()
    llm, vllm_config = build_vllm_llm(
        Path(worker_input["hf_checkpoint"]), max_model_len=max_model_len
    )
    model_startup_seconds = time.perf_counter() - build_start

    # EOS is fixed by the validated Esme bundle manifest and conversion contract. Avoid loading
    # the bundle model in this process; vLLM remains the sole CUDA owner here.
    eos_token_ids = frozenset({2})
    rows: list[WorkerRow] = []
    for workload in worker_input["workloads"]:
        requests = [
            EsmeBenchRequest(
                request_id=request["request_id"],
                prompt=request["prompt"],
                prompt_ids=tuple(request["prompt_ids"]),
            )
            for request in workload["requests"]
        ]
        decode_once = vllm_decode_closure(
            llm,
            requests,
            max_new_tokens=worker_input["max_new_tokens"],
            eos_token_ids=eos_token_ids,
            ignore_eos=worker_input["ignore_eos"],
        )
        for _ in range(worker_input["warmup"]):
            decode_once()
        per_iter_seconds: list[float] = []
        outputs: dict[str, list[int]] = {}
        for _ in range(worker_input["iters"]):
            start = time.perf_counter()
            outputs = decode_once()
            per_iter_seconds.append(time.perf_counter() - start)
        rows.append(
            {
                "batch_size": workload["batch_size"],
                "context_tokens": workload["context_tokens"],
                "per_iter_seconds": per_iter_seconds,
                "outputs": outputs,
            }
        )
    return {
        "system": "vllm",
        "identity": process_identity(),
        "rows": rows,
        "versions": library_versions(),
        "model_startup_seconds": model_startup_seconds,
        "engine_startup_seconds": None,
        "graph_capture_seconds": None,
        "attention_backend": None,
        "vllm_config": vllm_config,
    }


def _worker_main(system: str, input_path: Path, output_path: Path) -> None:
    worker_input: WorkerInput = json.loads(input_path.read_text(encoding="utf-8"))
    if system == "oracle":
        result = _run_oracle_worker(worker_input)
    elif system == "llm_infer":
        result = _run_llm_infer_worker(worker_input)
    elif system == "vllm":
        result = _run_vllm_worker(worker_input)
    else:
        raise ValueError(f"unknown worker system {system!r}")
    output_path.write_text(json.dumps(result), encoding="utf-8")


def gate_outputs(
    requests: list[RequestSpec],
    outputs: dict[str, list[int]],
    oracle_case: OracleCase,
    eos_token_ids: frozenset[int] = frozenset({2}),
) -> EsmeAgreement:
    """Apply the existing first-divergence tie rule from serialized fp32 oracle steps.

    The oracle child records every token whose fp32 logit is within 0.1 of that step's maximum.
    Up to the first divergence the fast and oracle prefixes are identical, so membership in that
    recorded set is exactly the same check as recomputing the step in
    ``compare_under_tie_tolerance``. The parent can therefore gate without owning a CUDA context.
    """
    from llm_infer.benchmarks.esme_three_way import EsmeAgreement
    from llm_infer.benchmarks.report import normalize_at_eos

    expected_ids = {request["request_id"] for request in requests}
    output_ids = set(outputs)
    extra = sorted(output_ids - expected_ids)
    if extra:
        raise ValueError(f"worker returned outputs for requests never sent: {extra[:3]}")
    exact = 0
    ties: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    failures: list[dict[str, object]] = [
        {"request": request_id, "detail": "missing outputs"}
        for request_id in sorted(expected_ids - output_ids)
    ]

    golden = normalize_at_eos(oracle_case["output_tokens"], eos_token_ids)
    for request in requests:
        request_id = request["request_id"]
        if request_id not in outputs:
            continue
        fast = normalize_at_eos(outputs[request_id], eos_token_ids)
        step = next(
            (
                index
                for index, pair in enumerate(zip(fast, golden, strict=False))
                if pair[0] != pair[1]
            ),
            None,
        )
        if step is None:
            if len(fast) == len(golden):
                exact += 1
            else:
                failures.append(
                    {
                        "request": request_id,
                        "detail": (
                            "no token diverged but lengths differ "
                            f"(fast={len(fast)}, golden={len(golden)})"
                        ),
                    }
                )
            continue

        fast_token = fast[step]
        golden_token = golden[step]
        oracle_step = oracle_case["steps"][step]
        near_tokens = oracle_step["near_token_logits"]
        if str(fast_token) in near_tokens and str(golden_token) in near_tokens:
            ties.append(
                {
                    "request": request_id,
                    "step": step,
                    "gap": oracle_step["top2_gap"],
                    "fast_token": fast_token,
                    "golden_token": golden_token,
                }
            )
        else:
            tolerance = oracle_case["tie_tolerance"]
            reviews.append(
                {
                    "request": request_id,
                    "step": step,
                    "fast_token": fast_token,
                    "golden_token": golden_token,
                    "automatic_boundary": tolerance,
                    "reference_gap": oracle_step["top2_gap"],
                    "detail": (
                        f"step {step}: token {fast_token} is not within {tolerance:g} "
                        "of the fp32 max; "
                        f"golden token {golden_token}, top2 gap {oracle_step['top2_gap']:.6g}"
                    ),
                }
            )

    return EsmeAgreement(
        exact=exact,
        tie=len(ties),
        nontie=len(reviews) + len(failures),
        total=len(requests),
        ties_sample=ties[:3],
        divergences_sample=[*reviews, *failures][:3],
        review_required=len(reviews),
        failed=len(failures),
        numerical_evidence=[*ties, *reviews],
    )


def finalize_row(
    worker_row: WorkerRow,
    *,
    system: str,
    agreement: EsmeAgreement,
    total_output_tokens: int,
) -> dict[str, object]:
    """Attach gate and steady-state statistics; suppress speed for a failed gate."""
    import dataclasses

    from llm_infer.benchmarks.reference_policy import build_system_evidence_record

    timings = worker_row["per_iter_seconds"]
    median_seconds = statistics.median(timings)
    return {
        "system": system,
        "batch_size": worker_row["batch_size"],
        "context_tokens": worker_row["context_tokens"],
        "agreement": dataclasses.asdict(agreement),
        "per_iter_seconds": timings,
        "median_seconds": median_seconds,
        "p95_seconds": percentile(timings, 95),
        "total_output_tokens": total_output_tokens,
        **build_system_evidence_record(
            agreement=agreement,
            median_seconds=median_seconds,
            total_tokens=total_output_tokens,
        ),
    }


@app.function(
    image=comparison_image,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=3 * 60 * 60,
)
def run_same_host(
    batch_sizes: list[int],
    context_lengths: list[int],
    max_new_tokens: int,
    warmup: int,
    iters: int,
) -> str:
    """Reserve one A100, run both systems in children, then apply the fp32 output gate."""
    from llm_infer.benchmarks import gpu_snapshot
    from llm_infer.benchmarks.report import total_output_tokens
    from llm_infer.model.pretrain_bundle_loader import read_json_object
    from llm_infer.model.runtime import TokenizersJsonTokenizer
    from scripts.convert_esme_to_hf import convert

    esme_bundles.reload()
    parent_identity = process_identity()

    convert_start = time.perf_counter()
    convert(Path(REMOTE_BUNDLE_PATH), Path(REMOTE_HF_PATH), max_position_embeddings=1024)
    conversion_seconds = time.perf_counter() - convert_start

    manifest = read_json_object(Path(REMOTE_BUNDLE_PATH) / "manifest.json")
    chat_template = manifest.get("chat_template")
    if not isinstance(chat_template, dict):
        raise ValueError("Esme manifest must contain an object chat_template")
    decoding = manifest.get("decoding")
    add_special_tokens = (
        bool(decoding.get("default_add_special_tokens", True))
        if isinstance(decoding, dict)
        else True
    )
    tokenizer = TokenizersJsonTokenizer(
        Path(REMOTE_BUNDLE_PATH) / "tokenizer.json",
        default_add_special_tokens=add_special_tokens,
        chat_template=chat_template,
    )
    workloads = build_context_workloads(tokenizer, tuple(batch_sizes), tuple(context_lengths))

    worker_input: WorkerInput = {
        "bundle_path": REMOTE_BUNDLE_PATH,
        "hf_checkpoint": REMOTE_HF_PATH,
        "workloads": workloads,
        "max_new_tokens": max_new_tokens,
        "warmup": warmup,
        "iters": iters,
        "block_size": BLOCK_SIZE,
        "capture_sizes": list(CAPTURE_SIZES),
        "ignore_eos": True,
    }
    worker_results: list[WorkerResult] = []
    with tempfile.TemporaryDirectory(prefix="esme-vllm-baseline-") as temp_dir:
        directory = Path(temp_dir)
        oracle_result = run_oracle_process(worker_input, directory)
        validate_child_identity(parent_identity, oracle_result["identity"], "oracle")
        for system in ("llm_infer", "vllm"):
            child = run_engine_process(system, worker_input, directory)
            validate_child_identity(parent_identity, child["identity"], system)
            worker_results.append(child)

    workloads_by_shape = {
        (workload["batch_size"], workload["context_tokens"]): workload for workload in workloads
    }
    rows: list[dict[str, object]] = []
    for worker_result in worker_results:
        for worker_row in worker_result["rows"]:
            shape = (worker_row["batch_size"], worker_row["context_tokens"])
            workload = workloads_by_shape[shape]
            agreement = gate_outputs(
                workload["requests"],
                worker_row["outputs"],
                oracle_result["cases"][str(worker_row["context_tokens"])],
                eos_token_ids=frozenset(),
            )
            expected_tokens = worker_row["batch_size"] * max_new_tokens
            output_tokens = total_output_tokens(worker_row["outputs"], frozenset())
            if output_tokens != expected_tokens:
                raise AssertionError(
                    f"{worker_result['system']} row {shape} produced {output_tokens} tokens, "
                    f"expected {expected_tokens}"
                )
            rows.append(
                finalize_row(
                    worker_row,
                    system=worker_result["system"],
                    agreement=agreement,
                    total_output_tokens=output_tokens,
                )
            )

    return json.dumps(
        {
            "rows": rows,
            "startup": {
                "conversion_seconds": conversion_seconds,
                "oracle_load_seconds": oracle_result["model_startup_seconds"],
                "reference_generation_seconds": oracle_result["generation_seconds"],
                "workers": {
                    result["system"]: {
                        "model_startup_seconds": result["model_startup_seconds"],
                        "engine_startup_seconds": result["engine_startup_seconds"],
                        "graph_capture_seconds": result["graph_capture_seconds"],
                    }
                    for result in worker_results
                },
            },
            "processes": {
                "parent": parent_identity,
                "children": {
                    "oracle": oracle_result["identity"],
                    **{result["system"]: result["identity"] for result in worker_results},
                },
            },
            "systems": {
                result["system"]: {
                    "versions": result["versions"],
                    "attention_backend": result["attention_backend"],
                    "vllm_config": result.get("vllm_config"),
                }
                for result in worker_results
            },
            "gpu": gpu_snapshot(),
        }
    )


def _command_config(command: str) -> tuple[tuple[int, ...], tuple[int, ...], int, int, int]:
    if command == "smoke":
        return (1,), (32,), 8, 0, 1
    if command == "focused":
        return (1, 8), CONTEXT_LENGTHS, MAX_NEW_TOKENS, WARMUPS, MEASURED_ITERS
    if command == "full":
        return FULL_BATCH_SIZES, CONTEXT_LENGTHS, MAX_NEW_TOKENS, WARMUPS, MEASURED_ITERS
    raise ValueError(f"command must be 'smoke', 'focused', or 'full'; got {command!r}")


@app.local_entrypoint()
def main(command: str = "focused", bundle_path: str = "") -> None:
    batch_sizes, context_lengths, max_new_tokens, warmup, iters = _command_config(command)
    local_bundle = local_bundle_path(bundle_path)
    stage_bundle(esme_bundles, local_bundle, label="esme-vllm-baseline")
    result = json.loads(
        run_same_host.remote(
            list(batch_sizes), list(context_lengths), max_new_tokens, warmup, iters
        )
    )
    record = {
        **result,
        "config": {
            "command": command,
            "model": "Esme-214M-Chat",
            "batch_sizes": list(batch_sizes),
            "context_lengths": list(context_lengths),
            "max_new_tokens": max_new_tokens,
            "warmup": warmup,
            "iters": iters,
            "reference": (
                "fp32 PretrainBundleModel.logits() greedy decode; "
                "bf16 first-divergence tie tolerance 0.1"
            ),
            "same_reserved_host": True,
            "separate_processes": True,
            "prefix_caching": False,
            "ignore_eos": True,
            "repro_command": (f"modal run scripts/modal_esme_vllm_baseline.py --command {command}"),
        },
    }
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-vllm-baseline-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-vllm-baseline] wrote {out_path}")


def _parse_worker_args() -> argparse.Namespace | None:
    if "--worker" not in sys.argv:
        return None
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("oracle", "llm_infer", "vllm"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    worker_args = _parse_worker_args()
    if worker_args is not None:
        _worker_main(worker_args.worker, worker_args.input, worker_args.output)
