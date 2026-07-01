"""Modal A100 harness: validate Esme bundle serving against direct bundle logits.

Esme currently loads from ``llm_pretrain_dense_v1`` export bundles and serves through the
engine by full recompute. This check keeps that contract explicit: the engine path must match
direct ``PretrainBundleModel.logits()`` greedy decode before any Esme benchmark number is useful.

The local entrypoint stages the serving bundle files into the repo-scoped Modal volume
``llm-infer-esme-bundles`` under ``/esme-214m-chat``. By default it reads the local headline
bundle path, or set ``ESME_BUNDLE_PATH`` / ``--bundle-path``.

    modal run scripts/modal_esme_reference_check.py --command smoke
    modal run scripts/modal_esme_reference_check.py --command check
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/llm-infer"
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"

DEFAULT_LOCAL_BUNDLE = Path("/Users/adamthuvesen/dev/menti/esme-posttrain/exports/esme-214m-chat")
VOLUME_NAME = "llm-infer-esme-bundles"
ESME_BUNDLE_DIR = "esme-214m-chat"
ESME_BUNDLE_MOUNT = "/esme-bundles"
REMOTE_BUNDLE_PATH = f"{ESME_BUNDLE_MOUNT}/{ESME_BUNDLE_DIR}"
REQUIRED_BUNDLE_FILES = ("manifest.json", "config.json", "tokenizer.json", "weights.pt")

app = modal.App("llm-infer-esme-reference-check")

_IGNORE = [
    "**/.git",
    "**/.venv",
    "**/__pycache__",
    "**/*.pyc",
    "**/.pytest_cache",
    "**/.ruff_cache",
    "**/.DS_Store",
    "bench-results/**",
]

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.11")
    .pip_install("torch>=2.2", "transformers>=4.43", "numpy>=1.26")
    .add_local_dir(REPO_ROOT, remote_path=REMOTE_ROOT, copy=True, ignore=_IGNORE)
    .workdir(REMOTE_ROOT)
    .run_commands("pip install --no-deps -e .")
)

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _local_bundle_path(bundle_path: str) -> Path:
    if bundle_path:
        return Path(bundle_path).expanduser()
    env_path = os.environ.get("ESME_BUNDLE_PATH")
    return Path(env_path).expanduser() if env_path else DEFAULT_LOCAL_BUNDLE


def _validate_local_bundle(bundle_path: Path) -> None:
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{bundle_path} is missing required bundle files: {missing}")
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{bundle_path}/manifest.json must contain a model object")
    if model.get("name") != "Esme-214M-Chat":
        raise ValueError(f"expected Esme-214M-Chat bundle, found model.name={model.get('name')!r}")
    if model.get("id") != "esme-214m-chat":
        raise ValueError(f"expected esme-214m-chat bundle id, found model.id={model.get('id')!r}")
    if manifest.get("eos_token_ids") != [2]:
        raise ValueError(f"expected Esme EOS [2], found {manifest.get('eos_token_ids')!r}")


def _stage_bundle(bundle_path: Path) -> None:
    _validate_local_bundle(bundle_path)
    print(f"[esme-reference] staging {bundle_path} -> {VOLUME_NAME}:/{ESME_BUNDLE_DIR}")
    with esme_bundles.batch_upload(force=True) as batch:
        for name in REQUIRED_BUNDLE_FILES:
            batch.put_file(bundle_path / name, f"/{ESME_BUNDLE_DIR}/{name}")
    print(f"[esme-reference] staged {len(REQUIRED_BUNDLE_FILES)} bundle files")


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=30 * 60,
)
def check_esme(num_requests: int, max_new_tokens: int) -> str:
    """Fail if engine generation diverges from direct bundle-logits greedy decode."""
    import torch

    from llm_infer.benchmarks import normalize_at_eos
    from llm_infer.model.decode import greedy_decode
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving import InferenceEngine, Request

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
    )
    if runtime.backend_id != "esme":
        raise AssertionError(f"expected esme backend, got {runtime.backend_id!r}")
    if runtime.model_id != "esme-214m-chat":
        raise AssertionError(f"expected esme-214m-chat, got {runtime.model_id!r}")
    if runtime.eos_token_ids != frozenset({2}):
        raise AssertionError(f"expected EOS [2], got {sorted(runtime.eos_token_ids)}")

    prompts = (
        "Write a tiny Python function that doubles an integer.",
        "Explain KV caching in one short sentence.",
        "Give one SQL query that counts rows in a table named events.",
        "Name two practical checks before trusting a benchmark.",
    )
    requests: list[tuple[str, tuple[int, ...]]] = []
    for index in range(num_requests):
        tokenized = runtime.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompts[index % len(prompts)]}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if not isinstance(tokenized, list) or not tokenized:
            raise ValueError(f"Esme tokenizer returned invalid prompt ids for request {index}")
        requests.append((f"esme-{index:03d}", tuple(int(token_id) for token_id in tokenized)))

    reference = {
        request_id: greedy_decode(
            runtime.model,
            list(prompt_ids),
            max_new_tokens=max_new_tokens,
            eos_token_ids=set(runtime.eos_token_ids),
        )
        for request_id, prompt_ids in requests
    }
    engine = InferenceEngine(
        runtime.model,
        block_size=128,
        num_blocks=len(requests) + 8,
        device="cuda",
        capabilities=runtime.capabilities,
    )
    for request_id, prompt_ids in requests:
        engine.add_request(
            Request(request_id, list(prompt_ids), max_new_tokens, runtime.eos_token_ids)
        )
    outputs = engine.run()
    mismatches = [
        request_id
        for request_id, output in outputs.items()
        if normalize_at_eos(output, runtime.eos_token_ids)
        != normalize_at_eos(reference[request_id], runtime.eos_token_ids)
    ]
    if mismatches:
        raise AssertionError(f"Esme engine diverged from direct bundle logits: {mismatches[:3]}")
    return (
        f"Esme reference check PASSED: {len(requests)}/{len(requests)} exact against "
        "direct PretrainBundleModel.logits() greedy decode"
    )


@app.local_entrypoint()
def main(command: str = "smoke", bundle_path: str = "") -> None:
    """Run the cheap smoke or fuller Esme reference check on the A100.

    smoke:   modal run scripts/modal_esme_reference_check.py --command smoke
    check:   modal run scripts/modal_esme_reference_check.py --command check
    """
    if command == "smoke":
        num_requests, max_new_tokens = 2, 8
    elif command == "check":
        num_requests, max_new_tokens = 8, 64
    else:
        raise ValueError(f"command must be 'smoke' or 'check', got {command!r}")
    _stage_bundle(_local_bundle_path(bundle_path))
    print(check_esme.remote(num_requests, max_new_tokens))
