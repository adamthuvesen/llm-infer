"""What fused elementwise kernels cost when the HTTP server starts and serves its first requests.

``--fused-elementwise`` compiles RMSNorm and RoPE with ``torch.compile`` before the server's
warmup and graph capture. Compilation is a one-time cost per process, and the compile caches on
disk hide it after the first run. So each arm runs in its own child process with empty
``torch.compile`` and Triton cache directories, inside one reserved A100. A child builds the app
with ``build_app_from_runtime`` and the server defaults, then sends each prompt size twice
through the real HTTP routes. The first request at a size shows any compile left for serving
time; the second shows the steady latency.

    modal run scripts/modal_esme_fused_startup.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
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
from scripts.modal_flash_image import FLASH_IMAGE, REMOTE_ROOT, REPO_ROOT

ARMS = ("plain", "fused")
REPEATS = 2
# Rough prompt sizes in words; the child records the token count each one produced.
PROMPT_WORDS = (24, 200, 600)
MAX_TOKENS = 16

app = modal.App("llm-infer-esme-fused-startup")
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _child(arm: str) -> dict:
    """Cold server start and first requests, in this fresh process."""
    import torch
    from fastapi.testclient import TestClient

    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serve import build_app_from_runtime

    load_start = time.perf_counter()
    runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    load_s = time.perf_counter() - load_start
    build_start = time.perf_counter()
    app_ = build_app_from_runtime(runtime, device="cuda", fused_elementwise=arm == "fused")
    torch.cuda.synchronize()
    build_s = time.perf_counter() - build_start
    requests: list[dict] = []
    with TestClient(app_) as client:
        model_id = client.get("/v1/models").json()["data"][0]["id"]
        for words in PROMPT_WORDS:
            prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * (words // 9 + 1))
            for attempt in ("first", "repeat"):
                start = time.perf_counter()
                response = client.post(
                    "/v1/completions",
                    json={
                        "model": model_id,
                        "prompt": prompt,
                        "max_tokens": MAX_TOKENS,
                        "temperature": 0,
                    },
                )
                elapsed = time.perf_counter() - start
                response.raise_for_status()
                usage = response.json()["usage"]
                requests.append(
                    {
                        "prompt_tokens": usage["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"],
                        "attempt": attempt,
                        "latency_s": elapsed,
                    }
                )
    return {"arm": arm, "model_load_s": load_s, "build_app_s": build_s, "requests": requests}


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=45 * 60,
)
def fused_startup() -> str:
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions

    stall_guard.arm(limit_s=20 * 60)
    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    runs: list[dict] = []
    for repeat in range(REPEATS):
        for arm in ARMS if repeat % 2 == 0 else ARMS[::-1]:
            with tempfile.TemporaryDirectory(prefix=f"fused-startup-{arm}-") as cache:
                env = dict(
                    os.environ,
                    TORCHINDUCTOR_CACHE_DIR=f"{cache}/inductor",
                    TRITON_CACHE_DIR=f"{cache}/triton",
                )
                output = Path(cache) / "result.json"
                start = time.perf_counter()
                subprocess.run(
                    [sys.executable, "-m", "scripts.modal_esme_fused_startup", arm, str(output)],
                    cwd=REMOTE_ROOT,
                    env=env,
                    check=True,
                )
                result = json.loads(output.read_text(encoding="utf-8"))
                result["process_s"] = time.perf_counter() - start
                runs.append(result)
            print(
                f"[fused-startup] {arm}: build_app {result['build_app_s']:.1f} s, first requests "
                + ", ".join(
                    f"{r['prompt_tokens']} tok {r['latency_s']:.2f} s"
                    for r in result["requests"]
                    if r["attempt"] == "first"
                )
            )
            stall_guard.progress()
    return json.dumps(
        {
            "probe": "fused-startup",
            "container_id": os.environ.get("MODAL_TASK_ID", "unknown"),
            "runs": runs,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(bundle_path: str = "") -> None:
    stage_bundle(esme_bundles, local_bundle_path(bundle_path), label="fused-startup")
    record = json.loads(fused_startup.remote())
    record["config"] = {
        "model": "Esme-214M-Chat",
        "arms": list(ARMS),
        "repeats": REPEATS,
        "prompt_words": list(PROMPT_WORDS),
        "max_tokens": MAX_TOKENS,
        "server": "build_app_from_runtime defaults on CUDA bf16",
        "repro_command": "modal run scripts/modal_esme_fused_startup.py",
    }
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"esme-fused-startup-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[fused-startup] wrote {out_path}")


if __name__ == "__main__" and len(sys.argv) == 3:
    Path(sys.argv[2]).write_text(json.dumps(_child(sys.argv[1])), encoding="utf-8")
