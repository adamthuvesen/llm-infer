"""Modal A100 harness: validate Esme bundle serving against direct bundle logits.

Esme currently loads from ``llm_pretrain_dense_v1`` export bundles and serves through the
engine by full recompute. This check keeps that contract explicit: the engine path must match
direct ``PretrainBundleModel.logits()`` greedy decode before any Esme benchmark number is useful.

The local entrypoint stages the serving bundle files into the repo-scoped Modal volume
``llm-infer-esme-bundles`` under ``/esme-214m-chat``. Set ``--bundle-path`` or
``ESME_BUNDLE_PATH``; otherwise it tries the standard sibling checkout export at
``../esme-posttrain/exports/esme-214m-chat``.

    modal run scripts/modal_esme_reference_check.py --command smoke
    modal run scripts/modal_esme_reference_check.py --command check
"""

from __future__ import annotations

from pathlib import Path

import modal

from scripts.modal_esme_bundle import (
    ESME_BUNDLE_MOUNT,
    REMOTE_BUNDLE_PATH,
    VOLUME_NAME,
    local_bundle_path,
    stage_bundle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_ROOT = "/root/llm-infer"
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"

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
    missing = sorted(set(reference) - set(outputs))
    extra = sorted(set(outputs) - set(reference))
    mismatches = [{"request": request_id, "detail": "missing output"} for request_id in missing]
    mismatches.extend(
        {"request": request_id, "detail": "unexpected output"} for request_id in extra
    )
    for request_id in reference:
        if request_id not in outputs:
            continue
        if normalize_at_eos(outputs[request_id], runtime.eos_token_ids) != normalize_at_eos(
            reference[request_id], runtime.eos_token_ids
        ):
            mismatches.append({"request": request_id, "detail": "tokens diverged"})
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
    stage_bundle(esme_bundles, local_bundle_path(bundle_path), label="esme-reference")
    print(check_esme.remote(num_requests, max_new_tokens))
