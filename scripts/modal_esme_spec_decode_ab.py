"""Prompt-lookup speculative decoding on the CUDA engine: does it pay at batch 1?

Speculation is off on CUDA (docs/findings.md entry 10): its one positive row was a single
repetition-heavy request on the old flash-attn backend. A speculative step verifies a draft in one
multi-token forward but gives up the decode window and its graph path. This study measures that
trade on two Dolly workloads at batch 1:

* ``open``: context-free instructions, where answers rarely repeat the prompt.
* ``grounded``: summarization, extraction, and closed QA over a passage, where answers copy it.

Each prompt runs once on each arm (``baseline``: the default engine; ``speculative``: prompt
lookup with up to 4 draft tokens), in alternating order, so every prompt is one paired
measurement. Every arm gates against the fp32 oracle, and the record stores direct token parity.
``scripts/judge_paired_ab.py`` applies the rule registered in docs/findings.md.

    modal run scripts/modal_esme_spec_decode_ab.py
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
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
GROUNDED_CATEGORIES = frozenset({"summarization", "information_extraction", "closed_qa"})
INSTRUCTION_CHARS = (16, 400)
CONTEXT_CHARS = (200, 1200)
PROMPTS_PER_WORKLOAD = 64
WARMUP_PROMPTS = 4
PROMPT_SEED = 1
NEW_TOKENS = 128
BLOCK_SIZE = 128
MAX_DRAFT_TOKENS = 4
MAX_NGRAM_SIZE = 4
ARMS = ("baseline", "speculative")

app = modal.App("llm-infer-esme-spec-decode-ab")
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def load_workloads() -> dict[str, tuple[list[int], list[str]]]:
    """Seeded Dolly samples per workload: (line indices, prompt texts), distinct texts only."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DOLLY_REPO, DOLLY_FILE, repo_type="dataset", revision=DOLLY_REVISION)
    pools: dict[str, dict[str, int]] = {"open": {}, "grounded": {}}
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        instruction, context = record["instruction"].strip(), record["context"].strip()
        if not INSTRUCTION_CHARS[0] <= len(instruction) <= INSTRUCTION_CHARS[1]:
            continue
        if not context:
            pools["open"].setdefault(instruction, index)
        elif (
            record["category"] in GROUNDED_CATEGORIES
            and CONTEXT_CHARS[0] <= len(context) <= CONTEXT_CHARS[1]
        ):
            pools["grounded"].setdefault(f"{context}\n\n{instruction}", index)
    count = WARMUP_PROMPTS + PROMPTS_PER_WORKLOAD
    workloads = {}
    for name, pool in pools.items():
        chosen = random.Random(PROMPT_SEED).sample(sorted(pool.items()), count)
        workloads[name] = ([index for _, index in chosen], [text for text, _ in chosen])
    return workloads


def _workload_row(runtime, prompts: list[str]):
    """Per-prompt walls for both arms on persistent batch-1 engines, plus acceptance counts."""
    import statistics

    import torch

    from llm_infer.benchmarks.esme_paged import build_requests
    from llm_infer.serving import InferenceEngine, Request
    from llm_infer.serving.speculative import SpeculativeDecodingConfig

    requests = build_requests(runtime.tokenizer, len(prompts), tuple(prompts))
    longest = max(len(req.prompt_ids) for req in requests)
    engines = {
        arm: InferenceEngine(
            runtime.model,
            block_size=BLOCK_SIZE,
            num_blocks=2 * math.ceil((longest + NEW_TOKENS + MAX_DRAFT_TOKENS) / BLOCK_SIZE) + 4,
            device="cuda",
            capabilities=runtime.capabilities,
            speculative=(
                SpeculativeDecodingConfig(
                    max_draft_tokens=MAX_DRAFT_TOKENS, max_ngram_size=MAX_NGRAM_SIZE
                )
                if arm == "speculative"
                else None
            ),
        )
        for arm in ARMS
    }
    verify = {"steps": 0, "tokens": 0}
    speculative_engine = engines["speculative"]
    original_verify = speculative_engine._decode_speculative

    def counted_verify(request, draft, result):
        before = len(request.generated)
        original_verify(request, draft, result)
        verify["steps"] += 1
        verify["tokens"] += len(request.generated) - before

    speculative_engine._decode_speculative = counted_verify

    def run(arm: str, req) -> tuple[float, list[int]]:
        engine = engines[arm]
        torch.cuda.synchronize()
        start = time.perf_counter()
        engine.add_request(Request(req.request_id, list(req.prompt_ids), NEW_TOKENS, frozenset()))
        outputs = engine.run()
        torch.cuda.synchronize()
        return time.perf_counter() - start, outputs[req.request_id]

    for req in requests[:WARMUP_PROMPTS]:
        for arm in ARMS:
            run(arm, req)
    verify.update(steps=0, tokens=0)
    walls: dict[str, list[float]] = {arm: [] for arm in ARMS}
    outputs: dict[str, dict[str, list[int]]] = {arm: {} for arm in ARMS}
    for position, req in enumerate(requests[WARMUP_PROMPTS:]):
        for arm in ARMS if position % 2 == 0 else ARMS[::-1]:
            wall, output = run(arm, req)
            walls[arm].append(wall)
            outputs[arm][req.request_id] = output
        if position % 16 == 15:
            stall_guard.progress()
    measured = requests[WARMUP_PROMPTS:]
    row: dict[str, object] = {"batch_size": 1}
    for arm in ARMS:
        row[arm] = {
            "walls_s": walls[arm],
            "median_request_s": statistics.median(walls[arm]),
            "raw_tokens_per_second": NEW_TOKENS * len(measured) / sum(walls[arm]),
        }
    row["speculative"]["verify_steps"] = verify["steps"]
    row["speculative"]["tokens_per_verify"] = (
        verify["tokens"] / verify["steps"] if verify["steps"] else 0.0
    )
    return row, measured, outputs


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def spec_decode_ab(workloads: dict[str, list[str]]) -> str:
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement
    from llm_infer.benchmarks.paired_change import paired_change
    from llm_infer.model.decode import greedy_decode
    from llm_infer.model.decode_graph import enable_decode_graphs_if_cuda
    from llm_infer.model.runtime import load_model_runtime

    stall_guard.arm()
    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    esme_bundles.reload()
    runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.bfloat16, device="cuda"
    )
    enable_decode_graphs_if_cuda(runtime.model, (1,))
    oracle = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    configs: dict[str, list[dict]] = {}
    for name, prompts in workloads.items():
        row, measured, outputs = _workload_row(runtime, prompts)
        reference = {
            req.request_id: greedy_decode(
                oracle.model, list(req.prompt_ids), max_new_tokens=NEW_TOKENS, eos_token_ids=set()
            )
            for req in measured
        }
        for arm in ARMS:
            agreement = tie_tolerant_agreement(
                oracle.model, measured, outputs[arm], reference, frozenset()
            )
            row[arm]["agreement"] = dataclasses.asdict(agreement)
        identical = [
            position
            for position, req in enumerate(measured)
            if outputs["baseline"][req.request_id] == outputs["speculative"][req.request_id]
        ]
        row["parity_status"] = "exact" if len(identical) == len(measured) else "review_required"
        row["identical_prompts"] = len(identical)
        # Exploratory, registered in advance: the paired change over prompts whose two arms
        # emitted the same tokens. It is reported, never claimed.
        if len(identical) >= 2:
            row["identical_subset_change"] = dataclasses.asdict(
                paired_change(
                    [row["baseline"]["walls_s"][i] for i in identical],
                    [row["speculative"]["walls_s"][i] for i in identical],
                )
            )
        configs[name] = [row]
        print(
            f"[spec] {name}: baseline {row['baseline']['median_request_s']:.3f} s, speculative "
            f"{row['speculative']['median_request_s']:.3f} s per request; "
            f"{row['speculative']['tokens_per_verify']:.2f} tokens/verify; "
            f"{len(identical)}/{len(measured)} identical; nontie "
            f"{row['baseline']['agreement']['nontie']}/{row['speculative']['agreement']['nontie']}"
        )
        stall_guard.progress()
    return json.dumps(
        {
            "probe": "decode-speculative-ab",
            "container_id": os.environ.get("MODAL_TASK_ID", "unknown"),
            "baseline_arm": "baseline",
            "candidate_arm": "speculative",
            "configs": configs,
            "rounds": PROMPTS_PER_WORKLOAD,
            "gpu": gpu_snapshot(),
            "versions": library_versions(),
        }
    )


@app.local_entrypoint()
def main(bundle_path: str = "") -> None:
    workloads = load_workloads()
    stage_bundle(esme_bundles, local_bundle_path(bundle_path), label="spec-decode")
    record = json.loads(
        spec_decode_ab.remote({name: texts for name, (_, texts) in workloads.items()})
    )
    record["config"] = {
        "model": "Esme-214M-Chat",
        "prompt_source": {
            "dataset": DOLLY_REPO,
            "revision": DOLLY_REVISION,
            "seed": PROMPT_SEED,
            "line_indices": {name: indices for name, (indices, _) in workloads.items()},
        },
        "prompts_per_workload": PROMPTS_PER_WORKLOAD,
        "warmup_prompts": WARMUP_PROMPTS,
        "new_tokens": NEW_TOKENS,
        "eos": "ignored",
        "speculative": {"max_draft_tokens": MAX_DRAFT_TOKENS, "max_ngram_size": MAX_NGRAM_SIZE},
        "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant, plus "
        "direct baseline/speculative parity per prompt",
        "repro_command": "modal run scripts/modal_esme_spec_decode_ab.py",
    }
    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"esme-spec-decode-ab-{time.strftime('%Y%m%dT%H%M%S')}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[spec] wrote {out_path}")
