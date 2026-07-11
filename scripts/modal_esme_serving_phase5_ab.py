"""Same-container A100 A/B of Phase 5 sampled-serving changes against main.

Two full repo trees ride in one image: the working tree (candidate) at ``/root/llm-infer``
and a git worktree of ``origin/main`` (baseline) staged locally at
``../llm-infer-phase5-baseline``. One GPU container runs each arm as a subprocess with
``PYTHONPATH``/cwd pointed at its tree, so both arms share the machine, driver, clocks, and
bundle — the code revision is the only variable. The candidate arm runs FIRST so any
shared-JIT-cache warmth biases toward the baseline, making a candidate improvement claim
conservative.

Each arm measures the phase0 greedy+sampled HTTP workloads through the serving eval
(grouped decode graphs on, the HTTP-server CUDA default) plus a raw engine-only greedy
bench for the regression bar. Baseline sampled b8/b64 rows report ``ref=failed`` by
construction — main still gates sampled rows on cross-shape exactness — so compare their
raw latency metrics, which the harness always keeps.

    git worktree add ../llm-infer-phase5-baseline origin/main
    modal run scripts/modal_esme_serving_phase5_ab.py --batch-sizes 1,8,64
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

from scripts.modal_esme_bundle import (
    ESME_BUNDLE_MOUNT,
    REMOTE_BUNDLE_PATH,
    VOLUME_NAME,
    local_bundle_path,
    stage_bundle,
)
from scripts.modal_flash_image import FLASH_IMAGE, IGNORE, REPO_ROOT

# The remote path deliberately mirrors the local sibling-directory name: this module is
# imported inside the container too, where REPO_ROOT.parent is /root, so the same
# BASELINE_LOCAL expression resolves to the baked baseline tree and the guard below holds
# in both contexts.
BASELINE_LOCAL = REPO_ROOT.parent / "llm-infer-phase5-baseline"
REMOTE_BASELINE_ROOT = "/root/llm-infer-phase5-baseline"
REMOTE_CANDIDATE_ROOT = "/root/llm-infer"

app = modal.App("llm-infer-esme-serving-phase5-ab")
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

if not BASELINE_LOCAL.is_dir():
    raise RuntimeError(
        f"baseline tree missing at {BASELINE_LOCAL}; create it with "
        "`git worktree add ../llm-infer-phase5-baseline origin/main`"
    )

AB_IMAGE = FLASH_IMAGE.add_local_dir(
    BASELINE_LOCAL, remote_path=REMOTE_BASELINE_ROOT, copy=True, ignore=IGNORE
)

# Runs identically against both revisions: it only touches APIs that exist on main —
# phase0 workloads, the network serving eval, and the raw paged engine.
ARM_RUNNER_SOURCE = '''
import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import torch

from llm_infer.benchmarks.esme_paged import build_requests
from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
from llm_infer.model.runtime import load_model_runtime
from llm_infer.serving import InferenceEngine, Request
from scripts.esme_serving_eval import phase0_http_workloads, run_network_http_workload

CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
ENGINE_BENCH_BLOCK_SIZE = 128
ENGINE_BENCH_NEW_TOKENS = 64


def engine_greedy_bench(runtime, num_requests):
    """Raw engine-only greedy wall (fresh engine per iter, median of 3). Diagnostic only."""

    def once():
        requests = build_requests(runtime.tokenizer, num_requests)
        needed = sum(
            math.ceil((len(r.prompt_ids) + ENGINE_BENCH_NEW_TOKENS) / ENGINE_BENCH_BLOCK_SIZE)
            for r in requests
        )
        engine = InferenceEngine(
            runtime.model,
            block_size=ENGINE_BENCH_BLOCK_SIZE,
            num_blocks=needed + max(4, num_requests),
            device="cuda",
            capabilities=runtime.capabilities,
        )
        for r in requests:
            engine.add_request(
                Request(
                    r.request_id,
                    list(r.prompt_ids),
                    ENGINE_BENCH_NEW_TOKENS,
                    runtime.eos_token_ids,
                )
            )
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = engine.run()
        torch.cuda.synchronize()
        wall = time.perf_counter() - start
        return wall, sum(len(ids) for ids in outputs.values())

    once()  # warmup at this shape
    walls = []
    tokens = 0
    for _ in range(3):
        wall, tokens = once()
        walls.append(wall)
    median_wall = statistics.median(walls)
    return {
        "batch_size": num_requests,
        "median_wall_s": median_wall,
        "output_tokens": tokens,
        "raw_tokens_per_s": tokens / median_wall if median_wall > 0 else None,
        "per_iter_wall_s": walls,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-sizes", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--measured-runs", type=int, required=True)
    args = parser.parse_args()
    batch_sizes = [int(part) for part in args.batch_sizes.split(",") if part.strip()]

    assert torch.cuda.is_available(), "no CUDA in the arm subprocess"
    oracle_runtime = load_model_runtime(
        "esme",
        bundle_path=args.bundle,
        dtype=torch.float32,
        device="cuda",
        attention_backend_name="torch_naive",
    )
    runtime = load_model_runtime(
        "esme",
        bundle_path=args.bundle,
        dtype=torch.bfloat16,
        device="cuda",
        attention_backend_name="auto",
    )
    capture_s = enable_decode_graphs_if_cuda(runtime.model, CAPTURE_SIZES)

    rows = []
    for batch_size in batch_sizes:
        for workload in phase0_http_workloads(
            batch_size,
            max_new_tokens=args.max_new_tokens,
            grouped_decode_graphs=True,
            grouped_capture_sizes=(batch_size,),
        ):
            reference_runtime = (
                oracle_runtime if workload.requests[0].sampling.is_greedy else runtime
            )
            result = asyncio.run(
                run_network_http_workload(
                    runtime,
                    workload,
                    device="cuda",
                    reference_runtime=reference_runtime,
                    warmup_runs=args.warmup_runs,
                    measured_runs=args.measured_runs,
                )
            )
            result["batch_size"] = batch_size
            result["grouped"] = True
            rows.append(result)
            metrics = result["metrics"]
            print(
                f"[arm] {workload.name}: ref={result['reference']['status']} "
                f"tok/s={metrics['throughput_tokens_per_s']} "
                f"raw tok/s={metrics['raw_throughput_tokens_per_s']} "
                f"TTFT p50/p95={metrics['ttft_p50_s']}/{metrics['ttft_p95_s']} "
                f"ITL p50/p95={metrics['itl_p50_s']}/{metrics['itl_p95_s']}",
                flush=True,
            )

    engine_bench = [engine_greedy_bench(runtime, size) for size in (1, 8, 64, 256)]
    for row in engine_bench:
        print(
            f"[arm] engine-greedy b{row['batch_size']}: raw {row['raw_tokens_per_s']:.1f} tok/s",
            flush=True,
        )

    args.output.write_text(
        json.dumps({"rows": rows, "engine_greedy_bench": engine_bench, "capture_s": capture_s})
    )


if __name__ == "__main__":
    main()
'''


@app.function(
    image=AB_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=3 * 60 * 60,
)
def measure_phase5_ab(
    batch_sizes: list[int],
    max_new_tokens: int,
    warmup_runs: int,
    measured_runs: int,
) -> str:
    """Run both arms sequentially on one GPU; candidate first (bias favors baseline)."""
    import os
    import subprocess
    import sys

    from llm_infer.benchmarks import gpu_snapshot, library_versions

    esme_bundles.reload()
    runner_path = Path("/tmp/phase5_arm_runner.py")
    runner_path.write_text(ARM_RUNNER_SOURCE)

    arms = {}
    for arm_name, tree in (
        ("candidate", REMOTE_CANDIDATE_ROOT),
        ("baseline", REMOTE_BASELINE_ROOT),
    ):
        output = Path(f"/tmp/phase5-{arm_name}.json")
        env = dict(os.environ)
        env["PYTHONPATH"] = tree
        print(f"[phase5-ab] running {arm_name} arm from {tree}", flush=True)
        started = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(runner_path),
                "--bundle",
                REMOTE_BUNDLE_PATH,
                "--output",
                str(output),
                "--batch-sizes",
                ",".join(str(size) for size in batch_sizes),
                "--max-new-tokens",
                str(max_new_tokens),
                "--warmup-runs",
                str(warmup_runs),
                "--measured-runs",
                str(measured_runs),
            ],
            check=True,
            cwd=tree,
            env=env,
        )
        arm_record = json.loads(output.read_text())
        arm_record["arm_wall_s"] = time.perf_counter() - started
        arm_record["tree"] = tree
        arms[arm_name] = arm_record

    return json.dumps(
        {
            "arms": arms,
            "arm_order": ["candidate", "baseline"],
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


def _parse_batch_sizes(raw: str) -> list[int]:
    sizes = [int(part) for part in raw.split(",") if part.strip()]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError(f"batch sizes must be positive integers; got {raw!r}")
    return sizes


def _metric_delta(candidate: dict, baseline: dict, key: str) -> str:
    cand_value, base_value = candidate.get(key), baseline.get(key)
    if not isinstance(cand_value, int | float) or not isinstance(base_value, int | float):
        return "n/a"
    if base_value == 0:
        return "n/a"
    return f"{(cand_value / base_value - 1) * 100:+.1f}%"


def _print_comparison(record: dict) -> None:
    baseline_rows = {row["name"]: row for row in record["arms"]["baseline"]["rows"]}
    for row in record["arms"]["candidate"]["rows"]:
        base = baseline_rows.get(row["name"])
        if base is None:
            continue
        cand_metrics, base_metrics = row["metrics"], base["metrics"]
        print(
            f"[phase5-ab] {row['name']}: "
            f"raw tok/s {base_metrics['raw_throughput_tokens_per_s']:.1f} -> "
            f"{cand_metrics['raw_throughput_tokens_per_s']:.1f} "
            f"({_metric_delta(cand_metrics, base_metrics, 'raw_throughput_tokens_per_s')}), "
            f"TTFT p50 {_metric_delta(cand_metrics, base_metrics, 'ttft_p50_s')} "
            f"p95 {_metric_delta(cand_metrics, base_metrics, 'ttft_p95_s')}, "
            f"ITL p50 {_metric_delta(cand_metrics, base_metrics, 'itl_p50_s')} "
            f"p95 {_metric_delta(cand_metrics, base_metrics, 'itl_p95_s')}, "
            f"candidate ref={row['reference']['status']} base ref={base['reference']['status']}"
        )
    base_bench = {
        row["batch_size"]: row for row in record["arms"]["baseline"]["engine_greedy_bench"]
    }
    for row in record["arms"]["candidate"]["engine_greedy_bench"]:
        base = base_bench.get(row["batch_size"])
        if base is None or not base["raw_tokens_per_s"]:
            continue
        change = (row["raw_tokens_per_s"] / base["raw_tokens_per_s"] - 1) * 100
        print(
            f"[phase5-ab] engine-greedy b{row['batch_size']}: "
            f"{base['raw_tokens_per_s']:.1f} -> {row['raw_tokens_per_s']:.1f} tok/s "
            f"({change:+.1f}%)"
        )


@app.local_entrypoint()
def main(
    batch_sizes: str = "1,8,64",
    max_new_tokens: int = 128,
    warmup_runs: int = 2,
    measured_runs: int = 5,
    bundle_path: str = "",
) -> None:
    sizes = _parse_batch_sizes(batch_sizes)
    stage_bundle(esme_bundles, local_bundle_path(bundle_path), label="esme-phase5-ab")
    record = json.loads(measure_phase5_ab.remote(sizes, max_new_tokens, warmup_runs, measured_runs))
    record["config"] = {
        "model": "Esme-214M-Chat",
        "batch_sizes": sizes,
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "workloads": ["streaming-greedy", "streaming-sampled"],
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 32, "seed": 17},
        "arms": "candidate = working tree; baseline = origin/main worktree; one container",
        "reference": (
            "greedy: fp32 full-recompute with traced bf16 ties; candidate sampled: same-shape "
            "seeded replay; baseline sampled rows keep raw metrics only (main's cross-shape "
            "gate fails them by construction)"
        ),
        "timing": (
            "localhost Uvicorn HTTP request wall only; model load, graph capture, engine/KV "
            "construction, server startup, metric scrapes, and reference checks excluded; "
            "engine_greedy_bench is raw engine wall, fresh engine per iteration"
        ),
        "repro_command": (
            "git worktree add ../llm-infer-phase5-baseline origin/main && "
            "modal run scripts/modal_esme_serving_phase5_ab.py "
            f"--batch-sizes {batch_sizes} --max-new-tokens {max_new_tokens} "
            f"--warmup-runs {warmup_runs} --measured-runs {measured_runs}"
        ),
    }
    output_dir = REPO_ROOT / "bench-results"
    output_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = output_dir / f"esme-serving-phase5-ab-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[phase5-ab] wrote {out_path}")
    _print_comparison(record)
