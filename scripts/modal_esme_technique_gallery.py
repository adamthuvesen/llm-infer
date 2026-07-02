"""Modal A100 technique gallery for Esme-214M-Chat: one gated experiment per technique.

Runs every experiment in ``llm_infer/benchmarks/esme_gallery.py`` in ONE container, so each
on/off comparison is same-GPU decision-grade (cross-container A100 variance is ±20% for
this CPU-bound engine; see docs/benchmark.md). The engine rows run the headline
configuration — bf16 flash-attn — and every row is gated on the fp32
``PretrainBundleModel.logits()`` oracle with the audited tie-tolerant rule.

Workloads (chat template except where noted):

* prefix caching — 16 sibling requests sharing one long manual+question prompt.
* chunked prefill — 8 short chats decoding while 4 long prompts arrive in a burst.
* preemption — 12 chats into a KV pool sized for roughly a third of their worst case.
* speculative decoding — one request continuing repetition-heavy plain text (encode, not
  chat template: the honest niche is repetitive continuations, which chat answers rarely are).

    modal run scripts/modal_esme_technique_gallery.py --command smoke   # tiny shapes
    modal run scripts/modal_esme_technique_gallery.py --command gallery # the published run
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

app = modal.App("llm-infer-esme-technique-gallery")

esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_MANUAL_SENTENCES = (
    "The Acme audio player ships with a rechargeable battery rated for twenty hours.",
    "Hold the power button for three seconds to turn the device on or off.",
    "Pair over Bluetooth by holding the mode button until the light blinks blue.",
    "A double press of the mode button skips to the next track in the queue.",
    "Firmware updates install automatically whenever the player is charging on wifi.",
    "The equalizer offers five presets and one custom profile per user account.",
    "Storage holds roughly eight thousand songs at the default encoding quality.",
    "If playback stutters, disable other Bluetooth devices within a few meters.",
)

_REPEATED_PARAGRAPH = (
    "The quarterly report shows revenue grew nine percent, churn fell to two percent, "
    "and the support backlog cleared within one week. "
)


def _long_chat_prompt(tokenizer, *, target_tokens: int, question: str, tag: str) -> list[int]:
    """A chat prompt around ``target_tokens`` long: repeated manual text plus one question."""
    sentences: list[str] = []
    index = 0
    while True:
        sentences.append(_MANUAL_SENTENCES[index % len(_MANUAL_SENTENCES)])
        index += 1
        content = f"{tag}Product manual: " + " ".join(sentences) + f"\n\nQuestion: {question}"
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True
        )
        if len(ids) >= target_tokens:
            return [int(t) for t in ids]
        if index > 200:
            raise ValueError(f"could not reach {target_tokens} tokens")


@app.function(
    image=FLASH_IMAGE,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=60 * 60,
)
def run_gallery(command: str) -> str:
    """All four technique experiments, back to back on one GPU, bf16 flash engine rows."""
    import torch

    from llm_infer.benchmarks import gpu_snapshot, library_versions
    from llm_infer.benchmarks.esme_gallery import (
        run_chunked_prefill_latency,
        run_preemption_starved_pool,
        run_prefix_cache_on_off,
        run_speculative_batch1,
    )
    from llm_infer.benchmarks.esme_paged import DEFAULT_PROMPTS, HEADLINE_PROMPTS
    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.model.runtime import load_model_runtime
    from llm_infer.serving.speculative import SpeculativeDecodingConfig

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    smoke = command == "smoke"
    oracle_runtime = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    )
    flash_runtime = load_model_runtime(
        "esme",
        bundle_path=Path(REMOTE_BUNDLE_PATH),
        dtype=torch.bfloat16,
        device="cuda",
        attention_backend=FlashAttnPagedAttention(),
    )
    tokenizer = flash_runtime.tokenizer

    def chat_ids(content: str) -> list[int]:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True
        )
        return [int(t) for t in ids]

    results: list[dict[str, object]] = []

    # 1. Prefix caching: siblings sharing one long prompt; on = one prefill, off = N.
    shared_prompt = _long_chat_prompt(
        tokenizer,
        target_tokens=64 if smoke else 512,
        question="How do I skip to the next track?",
        tag="",
    )
    results.append(
        run_prefix_cache_on_off(
            oracle_runtime,
            flash_runtime,
            prompt_ids=shared_prompt,
            num_siblings=4 if smoke else 16,
            max_new_tokens=8 if smoke else 32,
            block_size=128,
            num_blocks=64 if smoke else 192,
            device="cuda",
            warmup=1,
            iters=1 if smoke else 3,
        )
    )
    print(f"[gallery] prefix-caching done: {results[-1]['wall_speedup_on_vs_off']:.2f}x wall")

    # 2. Chunked prefill: short chats decoding, then a burst of long prompts arrives.
    active_prompts = [chat_ids(p) for p in (DEFAULT_PROMPTS if smoke else HEADLINE_PROMPTS)]
    long_prompts = [
        _long_chat_prompt(
            tokenizer,
            target_tokens=96 if smoke else 700,
            question="Summarize the manual in one sentence.",
            tag=f"Copy {index}. ",
        )
        for index in range(1 if smoke else 4)
    ]
    results.append(
        run_chunked_prefill_latency(
            oracle_runtime,
            flash_runtime,
            active_prompts=active_prompts,
            active_max_new_tokens=16 if smoke else 128,
            long_prompts=long_prompts,
            long_max_new_tokens=8 if smoke else 32,
            arrival_after_steps=4 if smoke else 16,
            prefill_chunk_size=64 if smoke else 128,
            block_size=128,
            num_blocks=128 if smoke else 256,
            device="cuda",
        )
    )
    rows = results[-1]["rows"]
    stall_summary = " vs ".join(
        f"{row['stall_vs_pre_arrival_step']:.1f} steps ({row['label']})" for row in rows
    )
    print(f"[gallery] chunked-prefill done: stall {stall_summary}")

    # 3. Preemption: a starved pool must evict, and completions must stay exact.
    preempt_prompts = [
        chat_ids(HEADLINE_PROMPTS[index % len(HEADLINE_PROMPTS)])
        for index in range(3 if smoke else 12)
    ]
    results.append(
        run_preemption_starved_pool(
            oracle_runtime,
            flash_runtime,
            prompts=preempt_prompts,
            max_new_tokens=12 if smoke else 96,
            block_size=8 if smoke else 16,
            num_blocks=6 if smoke else 40,
            device="cuda",
        )
    )
    on_row = results[-1]["rows"][0]
    print(f"[gallery] preemption done: {on_row['preemptions']} preemptions, exact completions")

    # 4. Speculative decoding: batch-1 continuation of repetition-heavy plain text.
    spec_prompt = [int(t) for t in tokenizer.encode(_REPEATED_PARAGRAPH * (2 if smoke else 3))]
    results.append(
        run_speculative_batch1(
            oracle_runtime,
            flash_runtime,
            prompt_ids=spec_prompt,
            max_new_tokens=16 if smoke else 128,
            speculative=SpeculativeDecodingConfig(max_draft_tokens=4, max_ngram_size=4),
            block_size=128,
            num_blocks=16 if smoke else 32,
            device="cuda",
            warmup=1,
            iters=1 if smoke else 3,
        )
    )
    print(
        f"[gallery] speculative done: {results[-1]['latency_speedup_on_vs_off']:.2f}x latency, "
        f"{results[-1]['mean_tokens_per_verify_step']} tok/verify-step"
    )

    return json.dumps(
        {"experiments": results, "gpu": gpu_snapshot(), "versions": library_versions()}
    )


@app.local_entrypoint()
def main(command: str = "gallery", bundle_path: str = "") -> None:
    """Stage the bundle, run the four experiments in one container, write the JSON record."""
    if command not in ("smoke", "gallery"):
        raise ValueError(f"command must be 'smoke' or 'gallery', got {command!r}")
    local_bundle = local_bundle_path(bundle_path)
    stage_bundle(esme_bundles, local_bundle, label="esme-gallery")

    print(f"[esme-gallery] {command}: 4 experiments, one A100 container, bf16 flash rows")
    record = json.loads(run_gallery.remote(command))
    record["config"] = {
        "command": command,
        "model": "Esme-214M-Chat",
        "backend": "FlashAttnPagedAttention (bf16)",
        "reference": "fp32 PretrainBundleModel.logits() greedy decode, tie-tolerant",
        "same_container": True,
        "repro_command": f"modal run scripts/modal_esme_technique_gallery.py --command {command}",
    }

    out_dir = REPO_ROOT / "bench-results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    out_path = out_dir / f"esme-gallery-{command}-{stamp}.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[esme-gallery] wrote {out_path}")
