# Correctness-fixture format

The correctness oracle (`tests/correctness/`) is the day-one gate every attention
backend is validated against. It never re-runs HuggingFace at test time; instead it
replays a **committed golden fixture** of HF greedy token ids and asserts the
llm-infer engine reproduces them token-for-token. This document specifies how a
golden case is built, stored, and validated — change the format only with a matching
change to `scripts/generate_goldens.py` and the oracle.

## What a golden case is

A golden case is a single text-to-SQL request plus its frozen HF greedy answer:

1. A **schema** (`tests/correctness/cases.py`), in the llm-rlvr `DatabaseSchema`
   shape — tables, columns, primary keys, foreign keys.
2. A **question**.
3. The **prompt token ids** — the llm-rlvr `cot` chat messages
   (`tests/correctness/prompt.py`, a byte-exact copy of llm-rlvr's builder) rendered
   through the pinned Instruct tokenizer's chat template with
   `add_generation_prompt=True`.
4. The **continuation token ids** — HF greedy decode of that prompt.

Cases are kept tiny (short schemas, `max_new_tokens = 40`) so generating goldens and
running the oracle need no GPU.

## Storage

One JSON file per (model, prompt-style): `tests/correctness/goldens/<name>.json`.

```jsonc
{
  "format_version": 1,
  "model":       { "id": "Qwen/Qwen2.5-Coder-3B-Instruct", "revision": "488639f1ff808d1d3d0ba301aef8c11461451ec5" },
  "environment": { "transformers": "<ver>", "torch": "<ver>", "dtype": "float32" },
  "decoding":    { "do_sample": false, "temperature": 0, "max_new_tokens": 40,
                   "eos_token_ids": [151643, 151645], "hf_method": "full_recompute_greedy" },
  "prompt_builder": { "source": "llm-rlvr eval/prompt.py build_messages(style='cot'), exemplars=()",
                      "replica": "tests/correctness/prompt.py", "style": "cot", "few_shot_k": 0 },
  "cases": [
    { "case_id": "...", "db_id": "...", "question": "...",
      "prompt_ids": [...], "continuation_ids": [...], "continuation_text": "..." }
  ]
}
```

Everything needed to reproduce the truth is pinned in the file: model revision,
library versions, dtype, the greedy/temperature-0 decoding config, the EOS ids, and
the exact prompt-builder provenance. The oracle reads its dtype, EOS ids, and
`max_new_tokens` *from the fixture* so the engine decodes under identical settings.

## Regenerating

```bash
uv run python scripts/generate_goldens.py
```

Loads the pinned model in fp32 on CPU, applies the chat template, runs HF greedy for
each case, and rewrites the JSON. Add or edit cases in `tests/correctness/cases.py`,
then regenerate and commit the updated fixture.

## The honesty bar

The bar is **exact token ids** on the single-request unit path. A bf16 greedy step
can diverge from HF only on a genuine numerical tie (top logits equal within
tolerance); such a divergence is acceptable **only** when traced to that tie and
documented here — never waved off as "close enough." The oracle runs in **fp32** by
default, where these ties near-vanish.

## HF truth = full-recompute greedy, not `generate()`

The golden truth is computed by **full-recompute greedy**: one `forward` over the
whole sequence per step, `argmax`, append — the *same algorithm class* as the oracle
engine (which has no KV-cache and recomputes the full sequence each step). It is
deliberately **not** `model.generate`.

This was a real decision, forced by a traced divergence (see below). `generate` runs a
fused/cached attention kernel whose fp32 reduction order differs from token-by-token
recompute. Pinning the oracle to `generate` would test "does the engine replicate
HF's kernel fusion," not "does the engine decode the model correctly." Holding the
attention algorithm fixed on both sides isolates the model math — which is what the
oracle exists to check.

### Traced divergence (the one that set this policy)

Case `single_table_group_by`, step 32, fp32, all paths loading the identical pinned
model:

| Path | token at step 32 |
| --- | --- |
| llm-infer engine (full recompute, `torch_naive`) | `3270` ("write") |
| HF manual full recompute (`forward` per step)     | `3270` ("write") |
| HF `generate` (cached)                            | `42430` ("construct") |
| HF `generate(use_cache=False)`                    | `42430` ("construct") |

The two HF cached/uncached paths **agree with each other** and the two full-recompute
paths **agree with each other** — the split is recompute-vs-`generate`, not
cache-vs-no-cache. Feeding every path the *same* agreed 32-token prefix, the engine
and HF `forward` both produce, for the contested step:

```
logit[3270 "write"]     = 22.99959   (engine)   22.99964 (HF forward)
logit[42430 "construct"] = 22.60251   (engine)   22.60253 (HF forward)
gap = 0.397   |   max |engine − HF forward| over the whole vocab = 5.3e-5
```

The gap between the top two tokens is **0.397 logits — about 7,000× the engine-vs-HF
logit noise (5.3e-5)**. This is *not* a numerical tie: under identical
full-recompute math the winner is unambiguous and the engine matches HF exactly. The
only reason `generate` picks the other token is its different fused-kernel reduction
order nudging a near-tie at an *earlier* step that cascades. Using full-recompute as
the truth, the engine matches HF **token-for-token on every case with zero
divergences** — so no bf16-style tie waiver is needed at all in oracle.

## What the oracle proves

For each committed case, the llm-infer single-request greedy decode through the
`torch_naive` reference backend produces token ids **identical** to HF full-recompute
greedy on the pinned Instruct model, under the pinned dtype and decoding config, using
the exact llm-rlvr `cot` prompt. That is the trusted reference every future backend
(paged, flash, …) must reproduce.

## The flash-attn backend and the tie-tolerance bar (flash-attn)

`flash_attn_paged` is the first backend that runs a **fused** kernel in **bf16**, so
its fp32 reduction order differs from `torch_naive`'s materialized softmax. That is the
same class of effect as the `generate`-vs-recompute split above: at a *genuine*
numerical tie — two top tokens whose logits are equal to within tiny noise — the fused
reduction order can flip the argmax. Unlike oracle (full-recompute fp32, where ties
near-vanish and the bar is bit-exact), the flash backend therefore needs a principled
tie waiver. The bar stays falsifiable:

- The flash backend's greedy continuation must match the committed golden
  **token-for-token**, with **one** exception class: a genuine numerical tie.
- When the flash token first differs from the golden at step `t`, the oracle recomputes
  step `t`'s logits with the **`torch_naive` fp32 reference path** over the canonical
  prefix (`prompt + golden[:t]` — the two sequences agree up to `t`, so this is exactly
  the context the flash backend decoded from). The divergence is accepted **only if**
  the reference's **top-2 logit gap ≤ tolerance** (`DEFAULT_TIE_TOLERANCE = 1e-3`); the
  step, both candidate tokens, the gap, and both reference logits are traced below.
- A divergence whose reference gap is **above** tolerance is a **FAIL** — under
  unambiguous reference math one token wins and the fused kernel picked the loser, which
  is a real kernel/layout bug, not a tie. There is no blanket "close enough" and no
  unconditional tolerance.

This mirrors the oracle discipline exactly: the *non*-tie gap traced above was 0.397
logits (~7000× the cross-path logit noise of 5.3e-5) and was correctly classified as
**not** a tie. The 1e-3 tolerance sits well above that observed numerical noise yet
hundreds of times below a real decision margin like 0.397, so it can launder genuine
ties but never a real bug. The policy lives in `tests/correctness/tie_tolerance.py`; the
GPU oracle is `tests/correctness/test_flash_attn_paged.py` (auto-skipped off CUDA, run
on the target A100 via `scripts/modal_oracle.py`). The exact `torch_naive` CPU oracle is
unchanged and remains the local gate.

### Traced flash-attn divergences

Validated on Modal A100-80GB (`scripts/modal_oracle.py --command oracle`), bf16 flash
backend vs the committed fp32 goldens, `max_new_tokens = 40` across all three cases
(`single_table_count`, `two_table_join`, `single_table_group_by`):

**Zero divergences — no tie waiver was needed.** Over all 120 decode steps the
`flash_attn_paged` bf16 fused kernel produced token ids **identical** to the golden
full-recompute greedy continuation on every case (`3 passed in 17.83s`). The fused
fp32-reduction-order difference never flipped an argmax on these cases, so the
tie-tolerance bar above — recompute the contested step with the `torch_naive` fp32
reference, accept only if the top-2 logit gap ≤ `1e-3` — was never exercised. It stays
in place as the principled safety net for any future case where a genuine near-tie does
flip: such a step would be accepted only on proof it is a numerical tie, and a non-tie
divergence would still FAIL the oracle. This mirrors oracle, where full-recompute fp32
also matched HF token-for-token with zero divergences and no waiver was needed.
