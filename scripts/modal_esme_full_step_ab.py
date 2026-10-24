"""Full-step decode graphs: does grouping every layer beat the server's four-layer group?

The HTTP server's CUDA default captures the first four Esme layers, with their KV writes and
FlashInfer attention, in one graph per exact batch; the other 26 layers replay piecewise graphs
with eager KV writes and attention between them. ``full_step`` groups all 30 layers, so a
token step is one graph launch plus its ``plan()``. Both arms run on persistent engines in one
container, 11 measured rounds per batch size that alternate which arm goes first. Every arm
gates against the fp32 oracle, and the record stores direct token parity between the arms.
``scripts/judge_paired_ab.py`` applies the rule registered in docs/findings.md.

    modal run scripts/modal_esme_full_step_ab.py
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
ARMS = ("grouped4", "full_step")
WARMUP_ROUNDS = 2
MEASURED_ROUNDS = 11
NEW_TOKENS = 128
BLOCK_SIZE = 128

app = modal.App("llm-infer-esme-full-step-ab")
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _arm_rows(runtime, batch_sizes: list[int]):
    """Per-batch walls for both arms on persistent engines, plus each arm's final outputs."""
    import statistics

    import torch

    from llm_infer.benchmarks.esme_paged import build_requests
    from llm_infer.serving import InferenceEngine, Request

    model = runtime.model
    grouped_layers = {"grouped4": 4, "full_step": model.num_layers}
    token_steps = NEW_TOKENS - 1
    rows: list[dict] = []
    outputs: dict[str, dict[int, dict[str, list[int]]]] = {arm: {} for arm in ARMS}
    for size in batch_sizes:
        requests = build_requests(runtime.tokenizer, size)
        needed = sum(math.ceil((len(req.prompt_ids) + NEW_TOKENS) / BLOCK_SIZE) for req in requests)
        engines = {}
        capture_s = {}
        for arm in ARMS:
            start = time.perf_counter()
            engines[arm] = InferenceEngine(
                model,
                block_size=BLOCK_SIZE,
                num_blocks=needed + max(4, size),
                device="cuda",
                capabilities=runtime.capabilities,
                grouped_decode_graphs=True,
                grouped_capture_sizes=(size,),
                grouped_layers=grouped_layers[arm],
            )
            capture_s[arm] = time.perf_counter() - start
            if size not in engines[arm].grouped_decode_runners:
                raise AssertionError(f"{arm}: no grouped runner captured for batch {size}")

        def submit(engine, requests=requests) -> None:
            for req in requests:
                engine.add_request(
                    Request(req.request_id, list(req.prompt_ids), NEW_TOKENS, frozenset())
                )

        def decode(arm: str, engines=engines) -> float:
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
            submit(engines[arm])
            outputs[arm][size] = engines[arm].run()
            runner = engines[arm].grouped_decode_runners[size]
            wall_ms = statistics.median(walls[arm]) * 1000.0 / token_steps
            row[arm] = {
                "grouped_layers": grouped_layers[arm],
                "walls_s": walls[arm],
                "wall_ms_per_token_step": wall_ms,
                "raw_tokens_per_second": size * 1000.0 / wall_ms,
                "engine_build_s": capture_s[arm],
                # A grouped runner that never fired would mean the arm measured piecewise.
                "grouped_steps_handled": runner.steps_handled,
            }
            if runner.steps_handled == 0:
                raise AssertionError(f"{arm}: the grouped runner never ran at batch {size}")
        rows.append(row)
        print(
            f"[full-step] b{size}: grouped4 {row['grouped4']['wall_ms_per_token_step']:.3f} -> "
            f"full_step {row['full_step']['wall_ms_per_token_step']:.3f} ms/step"
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
def full_step_ab(batch_sizes: list[int]) -> str:
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
    # The server captures piecewise buckets first; ramp-up steps below the exact batch use them.
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
        baseline, candidate = outputs["grouped4"][size], outputs["full_step"][size]
        row["parity_status"] = (
            "failed"
            if set(baseline) != set(candidate)
            else "exact"
            if normalized_outputs_match(baseline, candidate, frozenset())
            else "review_required"
        )
        print(
            f"[full-step] b{size}: parity {row['parity_status']}, nontie "
            f"{row['grouped4']['agreement']['nontie']}/{row['full_step']['agreement']['nontie']}"
        )
        stall_guard.progress()

    return json.dumps(
        {
            "probe": "decode-full-step-ab",
            "container_id": os.environ.get("MODAL_TASK_ID", "unknown"),
            "baseline_arm": "grouped4",
            "candidate_arm": "full_step",
            "configs": {"grouped": rows},
            "rounds": MEASURED_ROUNDS,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(bundle_path: str = "", batch_sizes: str = "1,8,64") -> None:
    sizes = [int(part) for part in batch_sizes.split(",") if part.strip()]
    if not sizes or min(sizes) < 1:
        raise ValueError(f"batch sizes must be positive integers; got {batch_sizes!r}")
    stage_bundle(esme_bundles, local_bundle_path(bundle_path), label="full-step")
    record = json.loads(full_step_ab.remote(sizes))
    record["config"] = {
        "model": "Esme-214M-Chat",
        "batch_sizes": sizes,
        "new_tokens": NEW_TOKENS,
        "warmup_rounds": WARMUP_ROUNDS,
        "measured_rounds": MEASURED_ROUNDS,
        "eos": "ignored",
        "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant, plus "
        "direct grouped4/full_step parity",
        "repro_command": (
            f"modal run scripts/modal_esme_full_step_ab.py --batch-sizes {batch_sizes}"
        ),
    }
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"esme-full-step-ab-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[full-step] wrote {out_path}")
