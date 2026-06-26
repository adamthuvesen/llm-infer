# AGENTS.md — llm-infer

A minimal, honest paged LLM inference engine with pluggable model backends. Qwen2.5-Coder
is the pinned HF-oracle backend; exported llm-pretrain DenseBackbone bundles are a sibling
backend. Full scope and build order live in [`docs/scoping.md`](docs/scoping.md). Read it
before proposing work.

## Core doctrine: the correctness oracle comes first

The project is sequenced so the **trusted reference comes before anything fast**.
Every optimized path is validated against a trusted greedy reference before it reports a
single tok/s. For Qwen, that reference is HuggingFace greedy decoding at the pinned revision.
For exported DenseBackbone bundles, the reference is the source bundle logits/generation
contract. There is no "correct-ish": exact token ids on the single-request unit path, or a
divergence traced to a numerical tie and documented.

## Backends

- `qwen`: `Qwen/Qwen2.5-Coder-3B-Instruct` at HF revision
  `488639f1ff808d1d3d0ba301aef8c11461451ec5`. Use the **Instruct** variant and its
  chat template. Plain `-3B` is a different model and would invalidate the Qwen oracle.
- `dense`: `llm_pretrain_dense_v1` export bundles from `llm-pretrain`, loaded from
  `manifest.json`, `config.json`, `tokenizer.json`, and `weights.pt`.

Qwen's default pin lives in `llm_infer/model/config.py` (`MODEL_ID`, `MODEL_REVISION`).
Backend selection lives in `llm_infer/model/runtime.py`.

## Layout

```
llm_infer/
  model/        # backend interface/registry, Qwen backend, DenseBackbone bundle backend [A,B]
  kernels/      # AttentionBackend protocol, torch_naive reference, flash_attn_paged     [A,C]
  kv_cache/     # block allocator, block tables, paged page store                        [B]
  scheduler/    # prefill/decode admission, continuous batching                          [B]
  serving/      # request queue, greedy sampler, continuous-batching decode loop         [B]
  benchmarks/   # naive HF vs llm-infer vs vLLM                                          [Phase D]
tests/correctness/   # the HF-exact greedy oracle + committed golden fixtures + the flash tie bar
docs/                # scoping.md (source of truth for scope) + architecture.md + fixture-format spec
scripts/             # golden generation + the Modal A100 flash oracle harness
```

Implemented: `model/`, `kernels/`, `kv_cache/`, `scheduler/`, `serving` (OpenAI-compatible
routes, streaming, metrics, loadgen), `benchmarks`, and the correctness oracles.

## The honesty bar (Phase A)

Exact token ids on the single-request unit path is the bar. bf16 greedy can diverge
from HF on a genuine tie-break step; that is acceptable **only** when each divergence
is traced to a numerical tie (logits equal within tolerance) and documented — never
waved off as "close enough." The oracle runs in **fp32** by default, where ties
near-vanish. See `docs/fixture-format.md`.

## Scope firewall — NOT in Phase A

No paged KV-cache, no scheduler/continuous batching, no flash-attn or any fast
kernel, no batch-correctness suite, no benchmarks, no vLLM comparison, no llm-rlvr-sql
rollout hook, no LoRA, no quantization, no server/streaming, no multi-GPU. If a
change touches these, it belongs to a later phase.

## Development

Managed with [`uv`](https://docs.astral.sh/uv/). Python 3.11+.

```bash
uv sync --extra dev          # create .venv and install deps
uv run ruff check            # lint (must be clean)
uv run pytest tests/correctness -q   # the oracle (runs against committed goldens)
```

The CPU oracle is the local gate and is **CPU-runnable by design**: the
`torch_naive` reference path runs against small committed golden token-id fixtures
and needs no GPU. Regenerating goldens loads the 3B model in fp32 on CPU — fine for a
few short greedy generations. The flash-attn backend is the **one** GPU-only path:
flash-attn needs a CUDA build, so its oracle (`tests/correctness/test_flash_attn_paged.py`,
auto-skipped off CUDA) runs on the target GPU via `scripts/modal_oracle.py`. Keep
Modal runs short — the flash oracle is a few seconds on an A100.

## Conventions

- Atomic [conventional commits](https://www.conventionalcommits.org/) (`type(scope): message`),
  one per logical unit.
- Type hints throughout; no `Any` casts — fix the type.
- `pathlib.Path` for filesystem work.
- Comments explain *why*, not *what*. Names use concrete domain terms.
- Validate untrusted input loudly; keep `try` bodies small.
