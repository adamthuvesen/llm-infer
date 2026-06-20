# AGENTS.md — llm-infer

A minimal, honest paged LLM inference engine for **Qwen2.5-Coder-3B-Instruct**,
built to be benchmarked as an rlvr-sql GRPO rollout backend. Full scope and build
order live in [`docs/scoping.md`](docs/scoping.md). Read it before proposing work.

## Core doctrine: the correctness oracle comes first

The project is sequenced so the **trusted reference comes before anything fast**.
Every attention backend — now and later — is validated token-for-token against
HuggingFace greedy decoding by the oracle in `tests/correctness/`. A backend that
fails the oracle reports no throughput. There is no "correct-ish": exact token ids
on the single-request unit path, or a divergence traced to a numerical tie and
documented (see the honesty bar below).

## Model — pinned, do not substitute

- `Qwen/Qwen2.5-Coder-3B-Instruct`
- HF revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`
- Use the **Instruct** variant and its chat template. Plain `-3B` is a different
  model and would be wrong.

Pin lives in `llm_infer/model/config.py` (`MODEL_ID`, `MODEL_REVISION`).

## Layout

```
llm_infer/
  model/        # Qwen loading, weights, tokenizer, config + the greedy decode loop  [Phase A]
  kernels/      # AttentionBackend protocol + torch_naive reference                  [Phase A]
  kv_cache/     # block allocator, block tables, page metadata                       [Phase B]
  scheduler/    # prefill/decode admission, continuous batching                      [Phase B]
  serving/      # request queue, sampler, streaming loop                             [Phase E]
  benchmarks/   # naive HF vs llm-infer vs vLLM                                       [Phase D]
tests/correctness/   # the HF-exact greedy oracle + committed golden fixtures
docs/                # scoping.md (source of truth for scope) + fixture-format spec
scripts/             # golden generation script
```

Only `model/`, `kernels/`, and the oracle are implemented in Phase A. The other
subpackages are empty stubs whose docstrings name the phase that fills them.

## The honesty bar (Phase A)

Exact token ids on the single-request unit path is the bar. bf16 greedy can diverge
from HF on a genuine tie-break step; that is acceptable **only** when each divergence
is traced to a numerical tie (logits equal within tolerance) and documented — never
waved off as "close enough." The oracle runs in **fp32** by default, where ties
near-vanish. See `docs/fixture-format.md`.

## Scope firewall — NOT in Phase A

No paged KV-cache, no scheduler/continuous batching, no flash-attn or any fast
kernel, no batch-correctness suite, no benchmarks, no vLLM comparison, no rlvr-sql
rollout hook, no LoRA, no quantization, no server/streaming, no multi-GPU. If a
change touches these, it belongs to a later phase.

## Development

Managed with [`uv`](https://docs.astral.sh/uv/). Python 3.11+.

```bash
uv sync --extra dev          # create .venv and install deps
uv run ruff check            # lint (must be clean)
uv run pytest tests/correctness -q   # the oracle (runs against committed goldens)
```

Phase A is **CPU-runnable by design**: the oracle runs against small committed
golden token-id fixtures and needs no GPU. Regenerating goldens loads the 3B model
in fp32 on CPU — fine for a few short greedy generations. **Do not use a GPU or
Modal in Phase A.**

## Conventions

- Atomic [conventional commits](https://www.conventionalcommits.org/) (`type(scope): message`),
  one per logical unit.
- Type hints throughout; no `Any` casts — fix the type.
- `pathlib.Path` for filesystem work.
- Comments explain *why*, not *what*. Names use concrete domain terms.
- Validate untrusted input loudly; keep `try` bodies small.
