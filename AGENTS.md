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
  `_resolve_attention_backend` in `llm_infer/model/runtime.py`), and the HTTP server compiles
  RMSNorm and RoPE into fused kernels (`llm_infer/model/fused_elementwise.py`). `torch_naive`
  stays the fp32/CPU reference oracle. See `BackendCapabilities` in `llm_infer/model/interface.py`.
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
  kernels/           # AttentionBackend protocol, torch_naive reference, torch_sdpa, FlashInfer, flash-attn
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
chunked prefill, prefix caching, FlashInfer and flash-attn backends, piecewise and grouped CUDA
decode graphs, prompt-lookup speculative decode, request preemption, KV trace visualizer, and
OpenAI-compatible serving with metrics/loadgen.

## Benchmark rules

The single-request unit path must produce exact token ids. Esme policy-v2 statuses are `exact`,
`accepted_numerical`, `review_required`, and `failed`; direct parity also allows `not_applicable`.
The 0.1-logit bf16 boundary is automatic acceptance. It stays below the measured bf16 error
(p99.9 shrink 0.25, `docs/findings.md` entry 20) so the gate can still see a bug. Larger
numerical differences need a durable review. Entry 20 lists when that review may cite it.
Missing/extra output and confirmed bugs fail. Keep raw
timing in all measured rows, gate public tok/s with `headline_eligible`, and do not rerun a settled
experiment only to move its record to a newer policy format.

## Development

Use Python 3.13 to match the GPU image, [`uv`](https://docs.astral.sh/uv/), and Node 24 for
visualizer tests. CI installs Node 24 and runs the same check command.

```bash
uv sync --extra dev --extra serving
make check
```

`make help` lists commands from their target descriptions.
`make check` first checks policy imports, then runs lint, formatting, types, and the fast Python suite.
It also runs benchmark evidence checks and visualizer tests.
`make reproduce` runs the benchmark evidence check alone.
Use `uv run pytest tests/correctness -q -m slow` for the opt-in 3B reference check.

An architecture failure lists the forbidden import path and its source line.
Keep agreement data in `benchmarks.reference_policy.EsmeAgreement` and execution code outside policy readers.
After correcting the import, run `make architecture-check`, then `make check`.

The local check is **CPU-runnable**: no GPU required. The slow 3B reference checks are opt-in
because they load the pinned Qwen model and can take minutes on CPU.
The FlashInfer and flash-attn backends are GPU-only; Esme and Qwen reference checks run on the target
GPU via the Modal harnesses in `scripts/`. Benchmark harnesses use Modal A100-80GB, with
Esme in `scripts/modal_esme_*` and the Qwen flash check in `scripts/modal_reference_check.py`.

Modal runs bill until they finish or time out, and a hung run prints nothing. A GPU function
calls `stall_guard.arm()` from `scripts/modal_stall_guard.py` and `stall_guard.progress()` after
each row or batch, so a stall exits with thread stacks instead of running to its timeout. Run
GPU jobs one at a time unless the experiment needs them together. Whoever launches a background
run also watches its log and raises an alarm after 10 minutes without output.

## Enforced Rules

| Rule | Enforcement | Proof | Added |
| --- | --- | --- | --- |
| Policy readers must not import inference execution | Import Linter checks direct and indirect imports in `pyproject.toml` | `make architecture-check` | 2026-11-04 |
| Package and evidence CLI code must typecheck | `pyproject.toml` lists checked paths without package exclusions | `make typecheck` | 2026-11-04 |
| Public tok/s requires qualified fp32 reference evidence | `benchmarks/reference_policy.py` constructs and reads policy-v2 evidence | `tests/benchmarks/test_evidence_contract.py`, `make evidence-check` | 2026-11-04 |
| A/B parity and fp32 reference status stay separate | `benchmarks/paired_ab.py` judges records through the reference policy | `tests/benchmarks/test_evidence_contract.py`, `make evidence-check` | 2026-11-04 |
| Deferred tokens survive abort and batch changes | `serving/decode_window.py` owns staged buffers, `serving/engine.py` consumes them | `tests/serving/test_decode_window.py` | 2026-11-04 |
| Scheduler membership changes use scheduler methods | `Scheduler.waiting` and `Scheduler.running` expose read-only collection types | `make typecheck`, `tests/scheduler/test_scheduler.py` | 2026-11-04 |
| Trace replay rejects malformed events | `tracing.py` defines events, `visualizer/trace_loader.js` validates input | `tests/serving/test_tracing.py`, `make visualizer-test` | 2026-11-04 |

Module paths in these tables start inside `llm_infer/`.
Test paths, visualizer paths, and `pyproject.toml` start at the repository root.
The scheduler views restrict typed callers. They expose live collections and do not enforce
runtime immutability. Change membership through `add`, `admit`, `release`, `requeue`, or `cancel`.

## Change Navigation

| Change | Start here | Check |
| --- | --- | --- |
| Backend support or grouped graph construction | `model/interface.py`, `model/runtime.py` | `tests/model/test_runtime.py`, `tests/serving/test_engine_grouped_decode.py` |
| Request lifecycle, decode windows, or KV release | `serving/engine.py`, `serving/decode_window.py`, `scheduler/scheduler.py` | `tests/serving/test_decode_window.py`, `tests/correctness/` |
| Benchmark qualification or A/B verdicts | `benchmarks/reference_policy.py`, `benchmarks/paired_ab.py` | `tests/benchmarks/`, `make evidence-check` |
| Trace events or replay | `tracing.py`, `visualizer/trace_loader.js` | `tests/serving/test_tracing.py`, `make visualizer-test` |

## Conventions

- Atomic [conventional commits](https://www.conventionalcommits.org/) (`type(scope): message`),
  one per logical unit.
- Type hints throughout; no `Any` casts. Fix the type.
- `pathlib.Path` for filesystem work.
- Comments explain _why_, not _what_. Names use concrete domain terms.
- Validate untrusted input loudly; keep `try` bodies small.
