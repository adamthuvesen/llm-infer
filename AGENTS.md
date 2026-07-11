# AGENTS.md: llm-infer

`llm-infer` is a small inference engine focused on building real serving techniques
and measuring them clearly. `Esme-214M-Chat` is the headline and default documented
model. Qwen2.5-Coder is kept as the independent HuggingFace reference for
correctness regression. Full scope and what the repo proves live in
[`docs/scoping.md`](docs/scoping.md). Read it before proposing work.

## Core rule: check the reference before publishing speed

Raw benchmark timing is always kept. A public headline speed claim first has to qualify against a
known-good reference output on the same prompt and model. For Esme bundles, the reference is the source bundle
logits/generation contract. The Qwen reference uses the pinned HuggingFace
full-recompute greedy output. Experimental paths are allowed, but they must be
labeled as experimental until they pass those checks. The single-request unit path still requires
exact token ids. For Esme A/B experiments, keep fp32 reference status separate from direct
candidate/baseline parity: exact direct parity can support a relative result even when both paths
share a numerical review.

## Backends

- `esme`: Esme export bundles, with `Esme-214M-Chat` as the headline path. Internally these use
  the `llm_pretrain_dense_v1` DenseBackbone export format. Real paged KV runs through the same
  engine prefill/decode and scheduler path, with prefix caching, speculative decode, and
  preemption all checked for parity. `PretrainBundleModel.logits()` stays the full-recompute
  reference for the paged path: exact greedy-token parity on the real bundle, with cached logits
  within documented fp32 BLAS reduction-order noise and well below any decision margin. On CUDA
  fp16/bf16 bundles `auto` selects `FlashInferPagedAttention` for batched decode (see
  `_resolve_attention_backend` in `llm_infer/model/runtime.py`); `torch_naive` stays the fp32/CPU
  reference oracle. See `BackendCapabilities` in `llm_infer/model/interface.py`.
- `dense`: compatibility alias for the same internal bundle loader. Public docs/examples
  use `esme`.
- `qwen`: independent HF reference backend for `Qwen/Qwen2.5-Coder-3B-Instruct` at HF revision
  `488639f1ff808d1d3d0ba301aef8c11461451ec5`. Use the **Instruct** variant and its
  chat template for reference reproduction. Plain `-3B` is a different model and would
  invalidate the Qwen reference.

Qwen's revision pin lives in `llm_infer/model/config.py` (`MODEL_ID`, `MODEL_REVISION`).
Backend selection lives in `llm_infer/model/runtime.py`.

## Layout

```
llm_infer/
  model/             # backend registry, Esme/DenseBackbone loader, Qwen reference, rope utils
  kernels/           # AttentionBackend protocol, torch_naive reference, flash_attn_paged
  kv_cache/          # block allocator, block tables, paged page store
  scheduler/         # prefill/decode admission, continuous batching, optional preemption
  serving/           # InferenceEngine step loop, sampler, OpenAI HTTP server, metrics
  benchmarks/        # Esme workloads, reference policy, timing, and report helpers
tests/correctness/   # reference tests + batch/prefix/preemption/speculative suites
docs/                # scoping.md, architecture.md, benchmark.md, serving-eval.md, fixture-format.md
scripts/             # generate_goldens, Qwen reference check, Esme Modal harnesses,
                     # serving/loadgen tools, benchmark evidence checks, real KV trace generator
visualizer/          # schema-v3 KV trace replay UI
```

Implemented features include greedy reference tests, paged KV cache, continuous batching,
chunked prefill, prefix caching, flash-attn backend, speculative decode v1, request
preemption, KV trace visualizer, and OpenAI-compatible serving with metrics/loadgen.

## Benchmark rules

The single-request unit path must produce exact token ids. Esme policy-v2 statuses are `exact`,
`accepted_numerical`, `review_required`, and `failed`; direct parity also allows `not_applicable`.
The 0.1-logit bf16 boundary is automatic acceptance, not a measured error distribution. Larger
numerical differences need a durable review. Missing/extra output and confirmed bugs fail. Keep raw
timing in all measured rows, gate public tok/s with `headline_eligible`, and do not turn historical
benchmark migration into a reason to rerun otherwise settled experiments. See
`docs/internal/reference-policy-retro-audit.md` (maintainer-local notes, untracked).

## Development

Managed with [`uv`](https://docs.astral.sh/uv/). Python 3.11+.

```bash
uv sync --extra dev --extra serving   # dev tests + HTTP server deps
uv run ruff check                     # lint (must be clean)
uv run mypy llm_infer                 # package type check
uv run pytest tests/correctness -q    # fast CPU reference check (slow 3B reference checks deselected)
make check                            # lint, format, types, and full fast suite before merge
uv run pytest tests/correctness -q -m slow  # opt-in 3B CPU reference check
```

The local check is **CPU-runnable**: no GPU required. The slow 3B reference checks are opt-in
because they load the pinned Qwen model and can take minutes on CPU.
The flash-attn backend is GPU-only; Esme and Qwen reference checks run on the target
GPU via the Modal harnesses in `scripts/`. Benchmark harnesses use Modal A100-80GB, with
Esme in `scripts/modal_esme_*` and the Qwen flash check in `scripts/modal_reference_check.py`.

## Conventions

- Atomic [conventional commits](https://www.conventionalcommits.org/) (`type(scope): message`),
  one per logical unit.
- Type hints throughout; no `Any` casts. Fix the type.
- `pathlib.Path` for filesystem work.
- Comments explain _why_, not _what_. Names use concrete domain terms.
- Validate untrusted input loudly; keep `try` bodies small.
