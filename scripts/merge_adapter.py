"""Merge the historical llm-rlvr GRPO LoRA adapter into the Qwen base model.

The archived Qwen rollout benchmark serves a *merged* checkpoint so the engine stays LoRA-free:
the llm-rlvr GRPO run
(`grpo-s0`, the shipped main result) saved a rank-32 PEFT LoRA adapter over the pinned base
``Qwen/Qwen2.5-Coder-3B-Instruct``. This script loads that base, applies the adapter with
PEFT ``merge_and_unload``, and writes the merged bf16 weights to a Modal volume that the
rollout benchmark then serves to *both* llm-infer and vLLM (identical weights, fair timing).
bf16 is the model's native dtype (Qwen2.5 trains/serves bf16; the GRPO run was bf16) and the
dtype the engine's greedy reference check already validates, so serving keeps one uniform precision.

The adapter is read-only input on llm-rlvr's Modal volume ``text2sql-runs`` at
``grpo/grpo-s0/``. The merge is a one-time step; the rollout benchmark never re-merges.

    modal run scripts/merge_adapter.py            # verify, merge, write /merged/grpo-s0
    modal run scripts/merge_adapter.py --check    # verify adapter path + base pin only (no merge)

The adapter's ``base_model_name_or_path`` is asserted against the project's pin before any
merge — a mismatched base would silently corrupt every rollout number, so it fails loudly.
"""

from __future__ import annotations

import json

import modal

REMOTE_ROOT = "/root/llm-infer"
HF_CACHE = "/hf-cache"
RUNS_MOUNT = "/runs"  # llm-rlvr's text2sql-runs volume (read-only input)
MERGED_MOUNT = "/merged"  # this project's merged-weights artifact volume (output)
ADAPTER_SUBPATH = "grpo/grpo-s0"  # the shipped GRPO main result (ANCHOR: grpo-s0 only)
MERGED_NAME = "grpo-s0"

# Pinned base — the Instruct variant and its chat template the rollout numbers depend on.
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
BASE_MODEL_REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"

app = modal.App("llm-infer-merge")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2",
        "transformers>=4.43",
        "peft>=0.11",
        "accelerate>=0.30",
        "safetensors>=0.4",
        "huggingface_hub>=0.23",
    )
    .env({"HF_HOME": HF_CACHE})
)

hf_cache = modal.Volume.from_name("llm-infer-hf-cache", create_if_missing=True)
# llm-rlvr's training-runs volume (its GRPO adapters). Must already exist — we only read it.
runs = modal.Volume.from_name("text2sql-runs")
merged = modal.Volume.from_name("llm-infer-merged", create_if_missing=True)


def _verify_adapter(adapter_dir: str) -> dict:
    """Assert the adapter exists and was trained on the pinned base; return its config."""
    from pathlib import Path

    config_path = Path(adapter_dir) / "adapter_config.json"
    weights_present = (Path(adapter_dir) / "adapter_model.safetensors").exists() or (
        Path(adapter_dir) / "adapter_model.bin"
    ).exists()
    if not config_path.exists():
        raise FileNotFoundError(
            f"no adapter_config.json at {adapter_dir} — expected the GRPO adapter "
            f"'{ADAPTER_SUBPATH}' on the text2sql-runs volume. STOP: do not merge."
        )
    if not weights_present:
        raise FileNotFoundError(f"adapter weights missing at {adapter_dir}; STOP: do not merge.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config.get("base_model_name_or_path")
    if base != BASE_MODEL_ID:
        raise ValueError(
            f"adapter base {base!r} != pinned base {BASE_MODEL_ID!r}; merging onto the wrong "
            "base would invalidate every rollout number. STOP and report."
        )
    return config


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={HF_CACHE: hf_cache, RUNS_MOUNT: runs, MERGED_MOUNT: merged},
    timeout=30 * 60,
)
def merge(check_only: bool = False) -> str:
    """Verify the adapter, then (unless ``check_only``) merge it into the base and save bf16."""
    from pathlib import Path

    import peft
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = f"{RUNS_MOUNT}/{ADAPTER_SUBPATH}"
    adapter_config = _verify_adapter(adapter_dir)
    summary = {
        "adapter_dir": adapter_dir,
        "base": adapter_config["base_model_name_or_path"],
        "r": adapter_config.get("r"),
        "lora_alpha": adapter_config.get("lora_alpha"),
        "target_modules": sorted(adapter_config.get("target_modules", [])),
        "peft_version_trained": adapter_config.get("peft_version"),
    }
    if check_only:
        return "adapter OK (check-only, no merge): " + json.dumps(summary)

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"
    # Merge on the GPU: half-precision matmul (B·A folded into W) is unsupported on CPU.
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, revision=BASE_MODEL_REVISION, dtype=torch.bfloat16
    ).to("cuda")
    hf_cache.commit()  # persist freshly downloaded base weights for the rollout functions
    peft_model = PeftModel.from_pretrained(base, adapter_dir)
    merged_model = peft_model.merge_and_unload()  # folds B·A into W; engine stays LoRA-free

    out_dir = Path(MERGED_MOUNT) / MERGED_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(out_dir), safe_serialization=True)
    # Save the pinned base tokenizer alongside (the served path feeds pre-tokenized prompt ids,
    # so this is only for vLLM's tokenizer init / EOS ids — keep it byte-pinned to the base).
    AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION).save_pretrained(
        str(out_dir)
    )

    provenance = {
        **summary,
        "base_revision": BASE_MODEL_REVISION,
        "merged_dtype": "bfloat16",
        "merge_method": "peft.merge_and_unload",
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
        },
    }
    (out_dir / "merge_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    merged.commit()

    saved = sorted(p.name for p in out_dir.iterdir())
    return f"merged → {out_dir} (bf16). files: {saved}\nprovenance: {json.dumps(provenance)}"


@app.local_entrypoint()
def main(check: bool = False) -> None:
    """Merge grpo-s0 into the base (or just verify the adapter with ``--check``)."""
    print(merge.remote(check_only=check))
