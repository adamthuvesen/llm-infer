"""Modal A100 harness for reference-gated persistent Esme HTTP measurements.

The measured interval contains only concurrent localhost Uvicorn/TCP requests against a started
server. Model
loading, decode-graph capture, engine/KV-pool construction, server startup, metric scrapes, and
the fp32 reference check are recorded or run outside that interval.

    modal run scripts/modal_esme_serving_eval.py
    modal run scripts/modal_esme_serving_eval.py --batch-sizes 1,8 --max-new-tokens 32
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
from scripts.modal_flash_image import FLASH_IMAGE, REPO_ROOT

CAPTURE_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)

app = modal.App("llm-infer-esme-serving-eval")
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _parse_batch_sizes(raw: str) -> list[int]:
    sizes = [int(part) for part in raw.split(",") if part.strip()]
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError(f"batch sizes must be positive integers; got {raw!r}")
    return sizes


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=3 * 60 * 60,
)
def measure_serving(
    batch_sizes: list[int],
    max_new_tokens: int,
    warmup_runs: int,
    measured_runs: int,
    grouped: bool = False,
    grouped_ab: bool = False,
) -> str:
    """Run greedy and sampled HTTP rows in one A100 process.

    ``grouped_ab`` measures every row twice in this same container — piecewise-only
    engines, then engines with the exact-batch grouped runner — so the old-default versus
    new-default comparison is like-for-like on one GPU. Each row carries a ``grouped``
    field naming its arm.
    """
    import asyncio

    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime
    from scripts.esme_serving_eval import phase0_http_workloads, run_network_http_workload

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    oracle_runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.float32,
        device="cuda",
        attention_backend_name="torch_naive",
    )
    runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
        attention_backend_name="auto",
    )
    capture_s = enable_decode_graphs_if_cuda(runtime.model, CAPTURE_SIZES)
    arms = (False, True) if grouped_ab else (grouped,)
    rows: list[dict[str, object]] = []
    for arm_grouped in arms:
        for batch_size in batch_sizes:
            # Grouped runners are per-engine and exact-batch; capture only this workload's
            # steady batch so per-workload engine builds stay cheap. Ramp-up windows below
            # the steady batch fall back to the piecewise buckets, the serving reality.
            for workload in phase0_http_workloads(
                batch_size,
                max_new_tokens=max_new_tokens,
                grouped_decode_graphs=arm_grouped,
                grouped_capture_sizes=(batch_size,) if arm_grouped else None,
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
                        warmup_runs=warmup_runs,
                        measured_runs=measured_runs,
                        # phase0 measures the steady batch; gating admission keeps every
                        # run's decode shapes identical so sampled replay can gate.
                        admission_barrier=True,
                    )
                )
                result["batch_size"] = batch_size
                result["grouped"] = arm_grouped
                rows.append(result)
                metrics = result["metrics"]
                reference = result["reference"]
                print(
                    f"[serving] {workload.name} grouped={arm_grouped}: "
                    f"ref={reference['status']} "
                    f"tok/s={metrics['throughput_tokens_per_s']} "
                    f"TTFT p50/p95={metrics['ttft_p50_s']}/{metrics['ttft_p95_s']} "
                    f"ITL p50/p95={metrics['itl_p50_s']}/{metrics['itl_p95_s']}"
                )
    return json.dumps(
        {
            "rows": rows,
            "decode_graphs": {
                "capture_sizes": list(CAPTURE_SIZES),
                "capture_s": capture_s,
            },
            # The experiment mode, not one arm's flag: grouped_ab records both arms in
            # `rows` (each row's `grouped` field names its arm).
            "grouped_mode": "ab" if grouped_ab else ("grouped" if grouped else "baseline"),
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
            "attention_backend": type(runtime.model.backend).__name__,
        }
    )


@app.local_entrypoint()
def main(
    batch_sizes: str = "1,8,64,256",
    max_new_tokens: int = 128,
    warmup_runs: int = 2,
    measured_runs: int = 10,
    bundle_path: str = "",
    grouped: bool = False,
    grouped_ab: bool = False,
) -> None:
    sizes = _parse_batch_sizes(batch_sizes)
    if max_new_tokens < 2:
        raise ValueError(f"max_new_tokens must be >= 2 to measure ITL; got {max_new_tokens}")
    if warmup_runs < 1:
        raise ValueError(f"warmup_runs must be >= 1; got {warmup_runs}")
    if measured_runs < 1:
        raise ValueError(f"measured_runs must be >= 1; got {measured_runs}")
    stage_bundle(
        esme_bundles,
        local_bundle_path(bundle_path),
        label="esme-serving",
    )
    record = json.loads(
        measure_serving.remote(
            sizes, max_new_tokens, warmup_runs, measured_runs, grouped, grouped_ab
        )
    )
    record["config"] = {
        "model": "Esme-214M-Chat",
        "batch_sizes": sizes,
        "max_new_tokens": max_new_tokens,
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
        "workloads": ["streaming-greedy", "streaming-sampled"],
        "sampling": {"temperature": 0.8, "top_p": 0.95, "top_k": 32, "seed": 17},
        "reference": (
            "greedy: fp32 full-recompute with traced bf16 ties; sampled: same-shape seeded "
            "replay (every record of one signature identical), with the seeded single-request "
            "bf16 stream kept as divergence evidence, not the gate"
        ),
        "timing": (
            "localhost Uvicorn HTTP request wall only; model load, graph capture, "
            "engine/KV construction, "
            "server startup, metric scrapes, and reference checks excluded"
        ),
        "repro_command": (
            "modal run scripts/modal_esme_serving_eval.py "
            f"--batch-sizes {batch_sizes} --max-new-tokens {max_new_tokens} "
            f"--warmup-runs {warmup_runs} --measured-runs {measured_runs}"
            + (" --grouped-ab" if grouped_ab else "")
            + (" --grouped" if grouped and not grouped_ab else "")
        ),
    }
    output_dir = REPO_ROOT / "bench-results"
    output_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    if grouped_ab:
        label = "esme-serving-grouped-ab"
    elif grouped:
        label = "esme-serving-grouped"
    else:
        label = "esme-serving-baseline"
    output_path = output_dir / f"{label}-{stamp}.json"
    output_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-serving] wrote {output_path}")
