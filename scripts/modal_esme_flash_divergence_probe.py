"""Modal A100 diagnostic: attribute the esme-001 step-22 flash divergence (token 712 vs 4817).

The Esme flash gate flagged a real divergence: bf16 flash chose 712 where the fp32 oracle chose
4817 (fp32 top-2 gap 0.0119, ~12x the tie tolerance). This probe decides whether that is a
flash-attn bug or inherent whole-model bf16 numerics, by decoding esme-001 four ways through the
SAME engine paged path and printing the step-22 logits for tokens 712 and 4817:

* fp32 model, torch_naive  — the oracle / fp32 paged reference (picks 4817).
* bf16 model, torch_naive  — same backend as the reference, only the model dtype changed to bf16.
* bf16 model, flash        — the gate's path under test.
* fp32 model, flash        — flash with an fp32-precision model (isolates the kernel from dtype).

The CPU pre-check (run before writing this) already showed, on the real bundle: fp32 model picks
4817 (+0.0119), **bf16 model + torch_naive picks 712** (-0.0156), bf16 model + bf16 attention picks
712 (0.0). So a bf16 *torch_naive* run flips identically — no flash involved. If this GPU probe
confirms the same (bf16 torch_naive flips to 712 like bf16 flash, while fp32 flash stays 4817), the
divergence is inherent whole-model bf16 rounding at a genuinely narrow margin, NOT a flash bug:
classification (b). The Esme flash gate compares bf16 flash to the *fp32* oracle, so it surfaces
that dtype gap (unlike the Qwen flash gate, which compares bf16 flash to a bf16 golden).

    modal run scripts/modal_esme_flash_divergence_probe.py
"""

from __future__ import annotations

from pathlib import Path

import modal

from scripts.modal_flash_image import FLASH_IMAGE

DEFAULT_LOCAL_BUNDLE = Path("/Users/adamthuvesen/dev/menti/esme-posttrain/exports/esme-214m-chat")
VOLUME_NAME = "llm-infer-esme-bundles"
ESME_BUNDLE_DIR = "esme-214m-chat"
ESME_BUNDLE_MOUNT = "/esme-bundles"
REMOTE_BUNDLE_PATH = f"{ESME_BUNDLE_MOUNT}/{ESME_BUNDLE_DIR}"
REQUIRED_BUNDLE_FILES = ("manifest.json", "config.json", "tokenizer.json", "weights.pt")

# esme-001 is index 1 of the flash gate's prompt list; the divergence is at decode step 22.
PROMPTS = (
    "Write a tiny Python function that doubles an integer.",
    "Explain KV caching in one short sentence.",
    "Give one SQL query that counts rows in a table named events.",
    "Name two practical checks before trusting a benchmark.",
)
DIVERGENCE_PROMPT_INDEX = 1
DIVERGENCE_STEP = 22
TOKEN_FLASH = 712
TOKEN_ORACLE = 4817

app = modal.App("llm-infer-esme-flash-divergence-probe")
flash_image = FLASH_IMAGE
esme_bundles = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def _stage_bundle(bundle_path: Path) -> None:
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{bundle_path} is missing required bundle files: {missing}")
    with esme_bundles.batch_upload(force=True) as batch:
        for name in REQUIRED_BUNDLE_FILES:
            batch.put_file(bundle_path / name, f"/{ESME_BUNDLE_DIR}/{name}")
    print(f"[probe] staged {len(REQUIRED_BUNDLE_FILES)} bundle files")


@app.function(
    image=flash_image,
    gpu="A100-80GB",
    volumes={ESME_BUNDLE_MOUNT: esme_bundles},
    timeout=30 * 60,
)
def probe() -> str:
    """Decode esme-001 four ways through the engine; print step-22 logits for 712 and 4817."""
    import torch

    from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
    from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
    from llm_infer.model.runtime import load_model_runtime

    assert torch.cuda.is_available(), "no CUDA on the Modal worker"

    def runtime(dtype: torch.dtype, flash: bool):
        return load_model_runtime(
            "esme",
            bundle_path=Path(REMOTE_BUNDLE_PATH),
            dtype=dtype,
            device="cuda",
            attention_backend=FlashAttnPagedAttention() if flash else None,
        )

    oracle = runtime(torch.float32, flash=False).model
    tokenizer = load_model_runtime(
        "esme", bundle_path=Path(REMOTE_BUNDLE_PATH), dtype=torch.float32, device="cuda"
    ).tokenizer
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPTS[DIVERGENCE_PROMPT_INDEX]}],
        add_generation_prompt=True,
        tokenize=True,
    )

    # fp32-oracle greedy prefix up to the divergence step; the step-22 query token is toks[-1].
    toks = list(prompt_ids)
    for _ in range(DIVERGENCE_STEP):
        toks.append(int(torch.argmax(oracle.logits(toks)[-1])))
    ctx, query_tok = toks[:-1], toks[-1]

    def step22_logits(model) -> torch.Tensor:
        """Prefill ctx, decode the query token through the engine paged path, return its logits."""
        cache = PagedKVCache(
            num_layers=model.num_layers,
            num_blocks=64,
            block_size=16,
            num_kv_heads=model.num_kv_heads,
            head_dim=model.head_dim,
            dtype=model.dtype,
        )
        table = cache.new_request()
        model.prefill(ctx, cache, table)
        return model.decode_one(cache, table, int(query_tok))

    rows = []
    for label, dtype, flash in (
        ("fp32 model, torch_naive (oracle)", torch.float32, False),
        ("bf16 model, torch_naive (ref backend)", torch.bfloat16, False),
        ("bf16 model, flash (gate path)", torch.bfloat16, True),
        ("fp32 model, flash (kernel-only)", torch.float32, True),
    ):
        logits = step22_logits(runtime(dtype, flash).model)
        argmax = int(torch.argmax(logits))
        l712 = float(logits[TOKEN_FLASH])
        l4817 = float(logits[TOKEN_ORACLE])
        rows.append(
            f"{label:40}: argmax={argmax:5d}  l712={l712:.5f}  l4817={l4817:.5f}  "
            f"gap(4817-712)={l4817 - l712:+.5f}"
        )

    verdict = (
        "(b) inherent bf16: a bf16 torch_naive run flips like bf16 flash; fp32 flash stays 4817 "
        "→ whole-model bf16 rounding at a narrow fp32 margin, not a flash bug."
    )
    return (
        "esme-001 step-22 divergence probe\n" + "\n".join(rows) + "\n\nexpected verdict: " + verdict
    )


@app.local_entrypoint()
def main(bundle_path: str = "") -> None:
    local = Path(bundle_path).expanduser() if bundle_path else DEFAULT_LOCAL_BUNDLE
    _stage_bundle(local)
    print(probe.remote())
