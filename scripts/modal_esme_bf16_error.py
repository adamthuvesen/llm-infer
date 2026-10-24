"""Measured bf16 logit error on many unique prompts: what supports the 0.1-logit tie rule?

The engine decodes a pool of unique chat prompts in bf16 (FlashInfer, piecewise CUDA graphs,
decode windows: the raw-engine default) with EOS ignored. A recorder keeps every decode step's
logits row. The fp32 oracle and the bf16 model then recompute logits over exactly the token
contexts the engine saw, so each comparison is pure numerical error on one context, never a
different continuation. Three comparisons separate the error sources:

* ``engine_vs_fp32``: what the tie rule must cover.
* ``bf16_recompute_vs_fp32``: whole-model bf16 arithmetic alone, without the cached decode path.
* ``engine_vs_bf16_recompute``: what the cached path (paged KV, FlashInfer, graphs) adds.

Prompts are instructions from ``databricks/databricks-dolly-15k`` at a pinned revision
(CC BY-SA 3.0). The record keeps their line indices and a hash, not their text.

    modal run scripts/modal_esme_bf16_error.py --command smoke   # local CPU wiring check
    modal run scripts/modal_esme_bf16_error.py                   # A100 run
"""

from __future__ import annotations

import hashlib
import json
import math
import random
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

DOLLY_REPO = "databricks/databricks-dolly-15k"
DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_FILE = "databricks-dolly-15k.jsonl"
PROMPT_SEED = 0
PROMPT_CHARS = (16, 400)
PROMPT_COUNT = 1024
BATCH_SIZE = 64
NEW_TOKENS = 128
BLOCK_SIZE = 128
TOLERANCE = 0.1
WORST_FLIPS_KEPT = 10

app = modal.App("llm-infer-esme-bf16-error")
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def load_prompts(count: int) -> tuple[list[int], list[str]]:
    """Seeded sample of short, context-free Dolly instructions: (line indices, texts)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DOLLY_REPO, DOLLY_FILE, repo_type="dataset", revision=DOLLY_REVISION)
    eligible: list[tuple[int, str]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        instruction = record["instruction"].strip()
        if not record["context"].strip() and PROMPT_CHARS[0] <= len(instruction) <= PROMPT_CHARS[1]:
            eligible.append((index, instruction))
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible Dolly prompts; asked for {count}")
    chosen = random.Random(PROMPT_SEED).sample(eligible, count)
    return [index for index, _ in chosen], [text for _, text in chosen]


def _record_decode_logits(engine) -> list[tuple[tuple[str, ...], object]]:
    """Keep a copy of every decode step's logits with the request ids of its rows.

    The decode entry points receive the batch's requests in row order; the model calls under
    them return ``(batch, vocab)`` logits, sometimes in a graph's static output buffer, so each
    row block is cloned before the next step overwrites it.
    """
    records: list[tuple[tuple[str, ...], object]] = []
    current: dict[str, tuple[str, ...]] = {}

    def track_rows(name: str) -> None:
        original = getattr(engine, name)

        def tracked(requests, result):
            current["ids"] = tuple(request.request_id for request in requests)
            return original(requests, result)

        setattr(engine, name, tracked)

    def keep_logits(name: str) -> None:
        original = getattr(engine.model, name)

        def kept(*args, **kwargs):
            logits = original(*args, **kwargs)
            records.append((current["ids"], logits.detach().clone()))
            return logits

        setattr(engine.model, name, kept)

    for name in ("_decode_window_step", "_decode_normal"):
        track_rows(name)
    for name in ("decode_window_step", "decode_many"):
        keep_logits(name)
    return records


def measure(bf16_runtime, fp32_runtime, prompts: list[str], *, batch_size: int, new_tokens: int):
    """Decode ``prompts`` in batches on the bf16 engine and compare every decode step's logits."""
    import torch

    from llm_infer.benchmarks.esme_paged import build_requests
    from llm_infer.serving import InferenceEngine, Request
    from llm_infer.validation.bf16_error import concat, position_errors, summarize

    if len(prompts) % batch_size:
        raise ValueError(f"{len(prompts)} prompts do not split into batches of {batch_size}")
    requests = build_requests(bf16_runtime.tokenizer, len(prompts), tuple(prompts))
    comparisons = ("engine_vs_fp32", "bf16_recompute_vs_fp32", "engine_vs_bf16_recompute")
    parts: dict[str, list] = {name: [] for name in comparisons}
    worst_flips: list[dict[str, object]] = []
    decode_s = 0.0
    for start in range(0, len(requests), batch_size):
        batch = requests[start : start + batch_size]
        needed = sum(math.ceil((len(req.prompt_ids) + new_tokens) / BLOCK_SIZE) for req in batch)
        engine = InferenceEngine(
            bf16_runtime.model,
            block_size=BLOCK_SIZE,
            num_blocks=needed + batch_size,
            device=bf16_runtime.model.device,
            capabilities=bf16_runtime.capabilities,
        )
        records = _record_decode_logits(engine)
        for req in batch:
            engine.add_request(
                Request(req.request_id, list(req.prompt_ids), new_tokens, frozenset())
            )
        decode_start = time.perf_counter()
        outputs = engine.run()
        decode_s += time.perf_counter() - decode_start
        for name in ("decode_window_step", "decode_many"):
            delattr(engine.model, name)
        for req in batch:
            output = outputs[req.request_id]
            rows = [
                logits[ids.index(req.request_id)]
                for ids, logits in records
                if req.request_id in ids
            ]
            if len(output) != new_tokens or len(rows) != new_tokens - 1:
                raise AssertionError(
                    f"{req.request_id}: {len(output)} tokens and {len(rows)} decode rows for "
                    f"{new_tokens} new tokens"
                )
            engine_logits = torch.stack(rows)
            mismatched = int((engine_logits.argmax(-1).cpu() != torch.tensor(output[1:])).sum())
            if mismatched:
                raise AssertionError(
                    f"{req.request_id}: {mismatched} recorded rows disagree with the emitted "
                    "tokens; the row-to-request mapping is wrong"
                )
            context = list(req.prompt_ids) + output[:-1]
            positions = slice(len(req.prompt_ids), len(context))
            with torch.no_grad():
                fp32_logits = fp32_runtime.model.logits(context)[positions]
                recompute_logits = bf16_runtime.model.logits(context)[positions]
            engine_logits = engine_logits.to(fp32_logits.device)
            pairs = {
                "engine_vs_fp32": (fp32_logits, engine_logits),
                "bf16_recompute_vs_fp32": (fp32_logits, recompute_logits),
                "engine_vs_bf16_recompute": (recompute_logits, engine_logits),
            }
            for name, (reference, candidate) in pairs.items():
                errors = position_errors(reference, candidate)
                parts[name].append(errors)
                if name == "engine_vs_fp32" and bool(errors.flipped.any()):
                    step = int(errors.flip_gap.argmax())
                    worst_flips.append(
                        {
                            "request_id": req.request_id,
                            "prompt_index": start + batch.index(req),
                            "decode_step": step + 1,
                            "flip_gap": float(errors.flip_gap[step]),
                            "fp32_margin": float(errors.fp32_margin[step]),
                        }
                    )
        del engine, records
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[bf16-error] measured {start + len(batch)}/{len(requests)} prompts")
        stall_guard.progress()
    worst_flips.sort(key=lambda flip: -float(flip["flip_gap"]))
    return {
        "comparisons": {
            name: summarize(concat(parts[name]), tolerance=TOLERANCE) for name in comparisons
        },
        "worst_flips": worst_flips[:WORST_FLIPS_KEPT],
        "prompts_with_a_flip": len(worst_flips),
        "decode_wall_s": decode_s,
    }


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def bf16_error(prompts: list[str]) -> str:
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime

    stall_guard.arm()
    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    bf16_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    capture_s = enable_decode_graphs_if_cuda(bf16_runtime.model, (BATCH_SIZE,))
    fp32_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    result = measure(
        bf16_runtime, fp32_runtime, prompts, batch_size=BATCH_SIZE, new_tokens=NEW_TOKENS
    )
    return json.dumps(
        {
            **result,
            "attention_backend": type(bf16_runtime.model.backend).__name__,
            "graph_capture_s": capture_s,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(command: str = "run", bundle_path: str = "") -> None:
    """``run`` measures on the A100; ``smoke`` runs a tiny CPU version locally."""
    if command not in ("run", "smoke"):
        raise ValueError(f"command must be 'run' or 'smoke', got {command!r}")
    local_bundle = local_bundle_path(bundle_path)
    count = PROMPT_COUNT if command == "run" else 4
    prompt_indices, prompts = load_prompts(count)
    if command == "smoke":
        import torch

        from llm_infer.model.runtime import load_model_runtime

        record = measure(
            load_model_runtime("esme", bundle_path=local_bundle, dtype=torch.bfloat16),
            load_model_runtime("esme", bundle_path=local_bundle, dtype=torch.float32),
            prompts,
            batch_size=2,
            new_tokens=8,
        )
        batch_size, new_tokens = 2, 8
    else:
        stage_bundle(esme_bundles, local_bundle, label="bf16-error")
        record = json.loads(bf16_error.remote(prompts))
        batch_size, new_tokens = BATCH_SIZE, NEW_TOKENS
    record["config"] = {
        "command": command,
        "model": "Esme-214M-Chat",
        "path": "bf16 engine default: FlashInfer paged decode, piecewise graphs, decode windows",
        "prompt_source": {
            "dataset": DOLLY_REPO,
            "revision": DOLLY_REVISION,
            "filter": f"empty context, {PROMPT_CHARS[0]}-{PROMPT_CHARS[1]} chars",
            "seed": PROMPT_SEED,
            "line_indices": prompt_indices,
            "sha256": hashlib.sha256("\n".join(prompts).encode()).hexdigest(),
        },
        "prompts": count,
        "batch_size": batch_size,
        "new_tokens": new_tokens,
        "eos": "ignored",
        "tolerance": TOLERANCE,
        "reference": "fp32 PretrainBundleModel.logits() on the engine's own token contexts",
        "repro_command": f"modal run scripts/modal_esme_bf16_error.py --command {command}",
    }
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-bf16-error-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    summary = record["comparisons"]["engine_vs_fp32"]
    shrink = summary["competitor_shrink"]
    print(
        f"[bf16-error] engine vs fp32: {summary['positions']} positions, shrink p99.9 "
        f"{shrink['p99.9']:.4f}, max {shrink['max']:.4f}, "
        f"flips {summary['flips']} ({summary['flips_beyond_tolerance']} beyond {TOLERANCE})"
    )
    print(f"[bf16-error] wrote {out_path}")
