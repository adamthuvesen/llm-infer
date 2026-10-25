"""Fused elementwise kernels on full-depth decode graphs, and the HTTP server's cold start.

Entry 15 compiled ``rms_norm`` and ``apply_rope`` with ``torch.compile``: kernels per step halved,
but raw wall did not follow while FlashInfer's ``plan()`` synced every step. With that sync gone
(entry 19) and all 30 layers in one graph (entry 21), this probe asks whether fusion now pays.

``ab`` runs two arms on persistent engines that both group every layer: ``full_step`` (the server
default) and ``full_step_fused`` (the same, with the compiled functions swapped into the model
modules while that arm's engine captures and runs). 11 measured rounds per batch size alternate
which arm goes first. Every arm gates against the fp32 oracle, which runs with the original
functions, and the record stores direct token parity. ``scripts/judge_paired_ab.py`` applies the
rule registered in docs/findings.md.

``startup`` runs in its own fresh container: one cold ``build_app_from_runtime`` with the server
defaults, then warm engines at the server's cache settings and buckets: no grouped graphs, four
grouped layers, and all 30.

    modal run scripts/modal_esme_fusion_ab.py
    modal run scripts/modal_esme_fusion_ab.py --command startup
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
from pathlib import Path

import modal

from scripts import modal_stall_guard as stall_guard
from scripts.modal_esme_bundle import (
    ESME_BUNDLE_MOUNT,
    REMOTE_BUNDLE_PATH,
    VOLUME_NAME,
    local_bundle_path,
    stage_bundle,
)
from scripts.modal_flash_image import FLASH_IMAGE, REPO_ROOT

BATCH_SIZES = (1, 8, 64)
ARMS = ("full_step", "full_step_fused")
WARMUP_ROUNDS = 2
MEASURED_ROUNDS = 11
NEW_TOKENS = 128
BLOCK_SIZE = 128
STARTUP_REPEATS = 2

app = modal.App("llm-infer-esme-fusion-ab")
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _arm_rows(runtime, batch_sizes: list[int]):
    """Per-batch walls for both arms on persistent engines, plus each arm's final outputs."""
    import statistics

    import torch

    from llm_infer.benchmarks.esme_paged import build_requests
    from llm_infer.model.fused_elementwise import use_fused_elementwise
    from llm_infer.serving import InferenceEngine, Request

    model = runtime.model
    token_steps = NEW_TOKENS - 1
    rows: list[dict] = []
    outputs: dict[str, dict[int, dict[str, list[int]]]] = {arm: {} for arm in ARMS}
    for size in batch_sizes:
        requests = build_requests(runtime.tokenizer, size)
        needed = sum(math.ceil((len(req.prompt_ids) + NEW_TOKENS) / BLOCK_SIZE) for req in requests)
        engines = {}
        build_s = {}
        for arm in ARMS:
            use_fused_elementwise(arm == "full_step_fused")
            start = time.perf_counter()
            engines[arm] = InferenceEngine(
                model,
                block_size=BLOCK_SIZE,
                num_blocks=needed + max(4, size),
                device="cuda",
                capabilities=runtime.capabilities,
                grouped_decode_graphs=True,
                grouped_capture_sizes=(size,),
                grouped_layers=model.num_layers,
            )
            build_s[arm] = time.perf_counter() - start
            if size not in engines[arm].grouped_decode_runners:
                raise AssertionError(f"{arm}: no grouped runner captured for batch {size}")

        def submit(engine, requests=requests) -> None:
            for req in requests:
                engine.add_request(
                    Request(req.request_id, list(req.prompt_ids), NEW_TOKENS, frozenset())
                )

        def decode(arm: str, engines=engines) -> float:
            use_fused_elementwise(arm == "full_step_fused")
            engine = engines[arm]
            submit(engine)
            engine.step()
            torch.cuda.synchronize()
            start = time.perf_counter()
            while engine.scheduler.has_work():
                engine.step()
            torch.cuda.synchronize()
            return time.perf_counter() - start

        for _ in range(WARMUP_ROUNDS):
            for arm in ARMS:
                decode(arm)
        walls: dict[str, list[float]] = {arm: [] for arm in ARMS}
        for round_index in range(MEASURED_ROUNDS):
            for arm in ARMS if round_index % 2 == 0 else ARMS[::-1]:
                walls[arm].append(decode(arm))
        row: dict[str, object] = {"batch_size": size}
        for arm in ARMS:
            use_fused_elementwise(arm == "full_step_fused")
            submit(engines[arm])
            outputs[arm][size] = engines[arm].run()
            runner = engines[arm].grouped_decode_runners[size]
            wall_ms = statistics.median(walls[arm]) * 1000.0 / token_steps
            row[arm] = {
                "walls_s": walls[arm],
                "wall_ms_per_token_step": wall_ms,
                "raw_tokens_per_second": size * 1000.0 / wall_ms,
                # The fused arm's first build includes compilation inside capture warmup.
                "engine_build_s": build_s[arm],
                # A grouped runner that never fired would mean the arm measured piecewise.
                "grouped_steps_handled": runner.steps_handled,
            }
            if runner.steps_handled == 0:
                raise AssertionError(f"{arm}: the grouped runner never ran at batch {size}")
        use_fused_elementwise(False)
        rows.append(row)
        print(
            f"[fusion] b{size}: full_step {row['full_step']['wall_ms_per_token_step']:.3f} -> "
            f"fused {row['full_step_fused']['wall_ms_per_token_step']:.3f} ms/step"
        )
        stall_guard.progress()
        # The closures and the last runner hold this row's engines too; drop every reference so
        # the cache can return their graphs, KV pages, and workspaces before the next batch.
        del engines, runner, decode
        torch.cuda.empty_cache()
    return rows, outputs


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def fusion_ab(batch_sizes: list[int]) -> str:
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_paged import build_requests
    from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
    from llm_infer.benchmarks.reference_policy import normalized_outputs_match
    from llm_infer.model.decode import greedy_decode
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime

    stall_guard.arm()
    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    # Piecewise buckets serve ramp-up steps below the exact batch. They are captured with the
    # original functions, so both arms share them.
    enable_decode_graphs_if_cuda(runtime.model, tuple(sorted(set(batch_sizes))))
    rows, outputs = _arm_rows(runtime, batch_sizes)
    stall_guard.progress()

    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    for row in rows:
        size = int(row["batch_size"])
        requests = build_requests(runtime.tokenizer, size)
        reference = {
            req.request_id: greedy_decode(
                oracle_runtime.model,
                list(req.prompt_ids),
                max_new_tokens=NEW_TOKENS,
                eos_token_ids=set(),
            )
            for req in requests
        }
        for arm in ARMS:
            agreement = tie_tolerant_agreement(
                oracle_runtime.model, requests, outputs[arm][size], reference, frozenset()
            )
            row[arm]["agreement"] = dataclasses.asdict(agreement)
        baseline, candidate = outputs["full_step"][size], outputs["full_step_fused"][size]
        row["parity_status"] = (
            "failed"
            if set(baseline) != set(candidate)
            else "exact"
            if normalized_outputs_match(baseline, candidate, frozenset())
            else "review_required"
        )
        row["identical_requests"] = sum(baseline[key] == candidate[key] for key in baseline)
        print(
            f"[fusion] b{size}: parity {row['parity_status']} "
            f"({row['identical_requests']}/{size} identical), nontie "
            f"{row['full_step']['agreement']['nontie']}/"
            f"{row['full_step_fused']['agreement']['nontie']}"
        )
        stall_guard.progress()

    return json.dumps(
        {
            "probe": "decode-fusion-ab",
            "container_id": os.environ.get("MODAL_TASK_ID", "unknown"),
            "baseline_arm": "full_step",
            "candidate_arm": "full_step_fused",
            "configs": {"grouped": rows},
            "rounds": MEASURED_ROUNDS,
            "compiled_functions": ["rope_utils.rms_norm", "rope_utils.apply_rope"],
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=30 * 60,
)
def server_startup() -> str:
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serve import (
        DEFAULT_BLOCK_SIZE,
        DEFAULT_DECODE_GRAPH_BUCKETS,
        DEFAULT_NUM_BLOCKS,
        build_app_from_runtime,
    )
    from llm_infer.serving import InferenceEngine

    stall_guard.arm()
    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    load_start = time.perf_counter()
    runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    load_s = time.perf_counter() - load_start
    torch.cuda.synchronize()
    reserved_before = torch.cuda.memory_reserved()
    build_start = time.perf_counter()
    build_app_from_runtime(runtime, device="cuda")
    torch.cuda.synchronize()
    cold = {
        "model_load_s": load_s,
        "build_app_s": time.perf_counter() - build_start,
        "build_app_reserved_bytes": torch.cuda.memory_reserved() - reserved_before,
        "buckets": list(DEFAULT_DECODE_GRAPH_BUCKETS),
    }
    print(f"[startup] cold build_app_from_runtime: {cold['build_app_s']:.1f} s")
    stall_guard.progress()

    # Warm comparison at the server's own cache settings. The app above owns one engine; each
    # engine here is built and dropped. ``none`` has no grouped graphs, so the reserved-memory
    # difference against it is what grouped capture adds: graphs, their buffers, and the shared
    # FlashInfer workspace.
    warm: dict[str, list[dict]] = {"none": [], "grouped4": [], "full_step": []}
    layers = {"none": None, "grouped4": 4, "full_step": runtime.model.num_layers}
    for repeat in range(STARTUP_REPEATS):
        order = tuple(warm) if repeat % 2 == 0 else tuple(reversed(warm))
        for arm in order:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            reserved_before = torch.cuda.memory_reserved()
            start = time.perf_counter()
            engine = InferenceEngine(
                runtime.model,
                block_size=DEFAULT_BLOCK_SIZE,
                num_blocks=DEFAULT_NUM_BLOCKS,
                device="cuda",
                capabilities=runtime.capabilities,
                grouped_decode_graphs=arm != "none",
                grouped_capture_sizes=DEFAULT_DECODE_GRAPH_BUCKETS,
                grouped_layers=layers[arm],
            )
            torch.cuda.synchronize()
            runners = engine.grouped_decode_runners
            warm[arm].append(
                {
                    "engine_build_s": time.perf_counter() - start,
                    "capture_s": sum(runner.capture_seconds for runner in runners.values()),
                    "reserved_bytes": torch.cuda.memory_reserved() - reserved_before,
                }
            )
            del engine, runners
            stall_guard.progress()
    for arm, runs in warm.items():
        print(
            f"[startup] warm {arm}: build "
            + ", ".join(f"{run['engine_build_s']:.1f} s" for run in runs)
            + "; capture "
            + ", ".join(f"{run['capture_s']:.1f} s" for run in runs)
            + "; reserved "
            + ", ".join(f"{run['reserved_bytes'] / 2**20:.0f} MiB" for run in runs)
        )
    return json.dumps(
        {
            "probe": "server-startup",
            "container_id": os.environ.get("MODAL_TASK_ID", "unknown"),
            "cold": cold,
            "warm_grouped_capture": warm,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(command: str = "ab", bundle_path: str = "", batch_sizes: str = "1,8,64") -> None:
    if command not in {"ab", "startup"}:
        raise ValueError(f"command must be 'ab' or 'startup'; got {command!r}")
    sizes = [int(part) for part in batch_sizes.split(",") if part.strip()]
    if not sizes or min(sizes) < 1:
        raise ValueError(f"batch sizes must be positive integers; got {batch_sizes!r}")
    stage_bundle(esme_bundles, local_bundle_path(bundle_path), label="fusion")
    if command == "startup":
        record = json.loads(server_startup.remote())
        record["config"] = {
            "model": "Esme-214M-Chat",
            "repro_command": "modal run scripts/modal_esme_fusion_ab.py --command startup",
        }
        name = "esme-server-startup"
    else:
        record = json.loads(fusion_ab.remote(sizes))
        record["config"] = {
            "model": "Esme-214M-Chat",
            "batch_sizes": sizes,
            "new_tokens": NEW_TOKENS,
            "warmup_rounds": WARMUP_ROUNDS,
            "measured_rounds": MEASURED_ROUNDS,
            "eos": "ignored",
            "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant, plus "
            "direct full_step/full_step_fused parity",
            "repro_command": (
                f"modal run scripts/modal_esme_fusion_ab.py --batch-sizes {batch_sizes}"
            ),
        }
        name = "esme-fusion-ab"
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{name}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[fusion] wrote {out_path}")
