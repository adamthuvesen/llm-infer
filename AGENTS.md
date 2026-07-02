# AGENTS.md — llm-infer

`llm-infer` is a small inference engine focused on building real serving techniques
and measuring them clearly. `Esme-214M-Chat` is the headline and default documented
model. Qwen2.5-Coder is kept as the independent HuggingFace reference for
correctness regression. Full scope and what the repo proves live in
[`docs/scoping.md`](docs/scoping.md). Read it before proposing work.

## Core rule: match the reference before measuring speed

For any speed claim, the engine first has to match a known-good reference output on
the same prompt and model. For Esme bundles, the reference is the source bundle
logits/generation contract. The Qwen reference uses the pinned HuggingFace
full-recompute greedy output. Experimental paths are allowed, but they must be
labeled as experimental until they pass those checks. There is no "correct-ish":
exact token ids on the single-request unit path, or a divergence traced to a
numerical tie and documented.

## Backends

- `esme`: Esme export bundles, with `Esme-214M-Chat` as the headline path. Internally these use
  the `llm_pretrain_dense_v1` DenseBackbone export format. Real paged KV through the same engine
  prefill/decode + scheduler path — prefix caching, speculative decode, and preemption,
  all parity-gated. `PretrainBundleModel.logits()` stays the full-recompute reference oracle the
  paged path is validated against (exact greedy-token parity on the real bundle; cached logits
  within documented fp32 BLAS reduction-order noise, well below any decision margin). Flash-attn
  is off by default (the bundle uses the `torch_naive` backend). See `BackendCapabilities` in
  `llm_infer/model/interface.py`.
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
  benchmarks/        # shared workload + runners (Modal harnesses in scripts/)
tests/correctness/   # reference tests + batch/prefix/preemption/speculative suites
docs/                # scoping.md, architecture.md, benchmark.md; internal notes in docs/internal/
scripts/             # generate_goldens, Qwen reference harnesses, Esme Modal harnesses,
                     # modal_esme_reference_check, modal_esme_benchmark, merge_adapter,
                     # build_rollout_fixture, loadgen, kv trace fixture
visualizer/          # schema-v3 KV trace replay UI
```

Implemented features include greedy reference tests, paged KV cache, continuous batching,
chunked prefill, prefix caching, flash-attn backend, speculative decode v1, request
preemption, KV trace visualizer, and OpenAI-compatible serving with metrics/loadgen.

## Benchmark rules

The single-request unit path must produce exact token ids. bf16 greedy can diverge
from a reference on a genuine tie-break step; that is acceptable **only** when each divergence
is traced to a numerical tie (logits equal within tolerance) and documented — never
waved off as "close enough." The reference check runs in **fp32** by default, where ties
near-vanish. See `docs/internal/fixture-format.md`. A backend that fails the reference check reports no tok/s.

## Development

Managed with [`uv`](https://docs.astral.sh/uv/). Python 3.11+.

```bash
uv sync --extra dev --extra serving   # dev tests + HTTP server deps
uv run ruff check                     # lint (must be clean)
uv run pytest tests/correctness -q    # fast CPU reference check (slow 3B reference checks deselected)
uv run pytest -q                      # full fast suite before merge
uv run pytest tests/correctness -q -m slow  # opt-in 3B CPU reference check
```

The local check is **CPU-runnable by design** — no GPU required, and the slow 3B
reference checks are opt-in because they load the pinned Qwen model and can take minutes on CPU.
The flash-attn backend is GPU-only; Esme and Qwen reference checks run on the target
GPU via the Modal harnesses in `scripts/`. Benchmark harnesses use Modal A100-80GB, with
Esme in `scripts/modal_esme_*` and the Qwen reference in `scripts/modal_benchmark.py` /
`scripts/modal_rollout.py`.

## Conventions

- Atomic [conventional commits](https://www.conventionalcommits.org/) (`type(scope): message`),
  one per logical unit.
- Type hints throughout; no `Any` casts — fix the type.
- `pathlib.Path` for filesystem work.
- Comments explain _why_, not _what_. Names use concrete domain terms.
- Validate untrusted input loudly; keep `try` bodies small.
