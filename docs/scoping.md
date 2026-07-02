# llm-infer Scope

`llm-infer` is a small inference engine focused on real serving techniques and clear
measurement. The primary path is Esme, with `Esme-214M-Chat` loaded from
`llm_pretrain_dense_v1` export bundles. Qwen2.5-Coder remains only as historical
public-baseline evidence and regression coverage.

## Capability Statement

The repo implements a from-scratch paged inference engine with a generic model runtime:

- Esme bundle loading checked against source bundle outputs.
- Exact greedy decode gates for the unit path.
- Paged KV cache, continuous batching, chunked prefill, prefix caching, speculative decoding,
  and request preemption.
- OpenAI-compatible local serving.
- Replayable KV-cache traces.
- Esme benchmarks against naive HF through a converted HF checkpoint.

The claim is not production serving, and not competing with mature serving engines. The
claim is systems evidence:
serving techniques made legible, speed claims checked against known-good outputs, and measured
limits named plainly.

## Reference Before Speed

For any speed claim, the engine first has to match a known-good reference output on the same
prompt and model.

For Esme bundles, the reference is the source bundle logits/generation contract. The
full-recompute `PretrainBundleModel.logits()` path is the oracle; paged prefill/decode,
prefix caching, speculative decode, and preemption are validated against it.

Historical Qwen checks use the pinned HuggingFace full-recompute greedy output. The
single-request unit path must produce exact token ids. Batched and paged paths are then checked
against that behavior.

bf16 greedy can diverge from an fp32 oracle on a genuine numerical tie. That is acceptable only
when the tie is traced and documented; a non-tie divergence reports no tok/s.

## Backends

- `esme`: headline path for Esme export bundles. Public examples use this backend.
- `dense`: compatibility alias for the same internal bundle loader.
- `qwen`: historical public-baseline backend for `Qwen/Qwen2.5-Coder-3B-Instruct` at revision
  `488639f1ff808d1d3d0ba301aef8c11461451ec5`.

## What This Repo Proves

The repo is in its intended state when these remain true:

1. Esme bundle loading reproduces source logits/generation on fixed prompts.
2. The paged KV engine path matches the reference gate before reporting speed.
3. Continuous batching, chunked prefill, prefix caching, speculative decode, and preemption run
   through the shared engine path.
4. Esme benchmark docs compare naive HF and `llm_infer` on pinned hardware with every row
   oracle-gated.
5. Historical Qwen records remain reproducible without becoming the repo story.

## Non-Goals

- No production-readiness claim.
- No claim to approach mature production serving engines.
- No custom Triton/CUDA kernel in the current engine.
- No quantization or multi-GPU serving in the current engine.
- No claim that historical Qwen rollout economics generalize to Esme.

## Layout

```text
llm_infer/
  model/          # backend registry, Esme bundle loader, Qwen historical backend
  kv_cache/       # block allocator, block tables, paged page store
  scheduler/      # prefill/decode admission, continuous batching
  kernels/        # AttentionBackend protocol + implementations
  serving/        # request queue, sampler, streaming loop, OpenAI server
  benchmarks/     # Esme baselines plus historical Qwen harnesses
tests/correctness/ # reference and equivalence checks
docs/             # public docs
docs/internal/    # archived implementation notes
scripts/          # goldens, benchmarks, loadgen, trace fixtures
visualizer/       # schema-v3 KV trace replay UI
```

Every borrowed component lives behind a narrow interface with tests asserting it matches the
reference backend. The engine owns scheduling, page tables, block allocation, batching,
request state, sampling, benchmarking, and tracing.

## Historical Qwen Public Baselines

The Qwen backend remains for reproducibility and older evidence:

- the Qwen three-way benchmark in [benchmark.md](benchmark.md#historical-qwen-public-baseline);
- the `llm-rlvr` rollout timing in
  [internal/keeping-the-gpu-busy.md](internal/keeping-the-gpu-busy.md);
- the pinned HF golden fixtures used for regression coverage.

New benchmark work should use Esme unless the task is explicitly about reproducing the
historical Qwen record.
