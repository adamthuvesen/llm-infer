"""Generate the committed golden fixture: HF greedy token ids for a few text-to-SQL prompts.

This is the only step that runs HuggingFace. It records the full pinned environment
(model revision, transformers/torch versions, dtype, decoding config, prompt-builder
provenance), then for each case applies the Instruct chat template, runs HF greedy
decode, and stores the prompt ids + continuation ids. The oracle replays these goldens
without ever re-running HF.

HF truth is computed by **full-recompute greedy** — one ``forward`` over the whole
sequence per step, ``argmax``, append — *not* ``model.generate``. This matches the
llm-infer engine's algorithm class exactly (the Phase A engine has no KV-cache and
recomputes the full sequence each step). ``generate`` runs a fused/cached attention
kernel whose fp32 reduction order differs from token-by-token recompute; on a genuine
near-tie step those two HF paths themselves pick different tokens, so pinning the
oracle to ``generate`` would test "do you replicate HF's kernel fusion," not "do you
decode the model correctly." See ``docs/fixture-format.md`` for the full trace of the
one step where this mattered. Both HF paths and the engine load identically; the only
moving part is the attention reduction order, which the oracle deliberately holds
fixed by using the same recompute algorithm on both sides.

CPU-runnable by design — keep cases short. Run:

    uv run python scripts/generate_goldens.py

Writes ``tests/correctness/goldens/qwen2_5_coder_3b_instruct_cot.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the prompt builder importable when run as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_infer.model.config import MODEL_ID, MODEL_REVISION  # noqa: E402
from tests.correctness.cases import GOLDEN_CASES  # noqa: E402
from tests.correctness.prompt import build_cot_messages  # noqa: E402

# Decoding config — pinned into the fixture so the oracle decodes identically.
DTYPE = "float32"  # fp32: greedy tie-breaks near-vanish (honesty bar, docs/fixture-format.md)
MAX_NEW_TOKENS = 40
DO_SAMPLE = False  # greedy / temperature 0

GOLDEN_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "correctness"
    / "goldens"
    / "qwen2_5_coder_3b_instruct_cot.json"
)


def _torch_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def main() -> None:
    dtype = _torch_dtype(DTYPE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=dtype
    ).eval()
    eos_token_ids = sorted(_eos_ids(model, tokenizer))

    cases_out = []
    for case in GOLDEN_CASES:
        messages = build_cot_messages(case.schema, case.question)
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        )
        prompt_ids = inputs["input_ids"][0].tolist()
        continuation_ids = _hf_full_recompute_greedy(
            model, prompt_ids, max_new_tokens=MAX_NEW_TOKENS, eos_token_ids=set(eos_token_ids)
        )
        cases_out.append(
            {
                "case_id": case.case_id,
                "db_id": case.schema.db_id,
                "question": case.question,
                "prompt_ids": prompt_ids,
                "continuation_ids": continuation_ids,
                "continuation_text": tokenizer.decode(continuation_ids, skip_special_tokens=True),
            }
        )

    fixture = {
        "format_version": 1,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "environment": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "dtype": DTYPE,
        },
        "decoding": {
            "do_sample": DO_SAMPLE,
            "temperature": 0,
            "max_new_tokens": MAX_NEW_TOKENS,
            "eos_token_ids": eos_token_ids,
            "hf_method": "full_recompute_greedy",
        },
        "prompt_builder": {
            "source": "rlvr-sql eval/prompt.py build_messages(style='cot'), exemplars=()",
            "replica": "tests/correctness/prompt.py",
            "style": "cot",
            "few_shot_k": 0,
        },
        "cases": cases_out,
    }

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
    GOLDEN_PATH.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {len(cases_out)} cases to {GOLDEN_PATH}")
    print(f"sha256: {digest}")


def _hf_full_recompute_greedy(
    model: object, prompt_ids: list[int], *, max_new_tokens: int, eos_token_ids: set[int]
) -> list[int]:
    """HF greedy by full recompute: one forward over the whole sequence per step, argmax, append.

    The same algorithm the llm-infer engine runs, so the only thing the oracle compares
    is the model math — not a difference in cached-vs-recompute attention reduction order.
    """
    tokens = list(prompt_ids)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(torch.tensor([tokens])).logits[0, -1]
        next_id = int(torch.argmax(logits).item())
        tokens.append(next_id)
        generated.append(next_id)
        if next_id in eos_token_ids:
            break
    return generated


def _eos_ids(model: object, tokenizer: object) -> set[int]:
    """The generation-stopping ids HF greedy uses (config eos plus the tokenizer eos)."""
    ids: set[int] = set()
    cfg_eos = model.generation_config.eos_token_id
    if isinstance(cfg_eos, int):
        ids.add(cfg_eos)
    elif cfg_eos is not None:
        ids.update(cfg_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


if __name__ == "__main__":
    main()
