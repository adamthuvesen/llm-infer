# llm-infer Scope

`llm-infer` is a small inference engine focused on real serving techniques and clear
measurement. The primary path is Esme, with `Esme-214M-Chat` loaded from
`llm_pretrain_dense_v1` export bundles. Qwen2.5-Coder is kept as the independent
HuggingFace reference for correctness regression.

## Capability Statement

The repo implements a from-scratch paged inference engine with a generic model runtime:

- Esme bundle loading checked against source bundle outputs.
- Exact greedy decode gates for the unit path.
- Paged KV cache, continuous batching, chunked prefill, prefix caching, speculative decoding,
  and request preemption.
- OpenAI-compatible local serving.
- Replayable KV-cache traces.
- Esme benchmarks against naive HF through a converted HF checkpoint.

The repo claims systems evidence: serving techniques made legible, speed claims checked
against known-good outputs, and measured limits named plainly. Production readiness and
parity with mature serving engines are not project goals.

## Reference Before Public Speed

Benchmarks always keep their raw timing. Public headline speed requires a known-good reference
result on the same prompt and model. A/B experiments also record direct candidate-versus-baseline
parity so an optimization can be judged separately from numerical behavior shared by both paths.

For Esme bundles, the reference is the source bundle logits/generation contract. The
full-recompute `PretrainBundleModel.logits()` path is the reference; paged prefill/decode,
prefix caching, speculative decode, and preemption are validated against it.

The Qwen reference uses the pinned HuggingFace full-recompute greedy output: the
single-request unit path must produce exact token ids, and batched and paged paths are then
checked against that behavior.

bf16 greedy can diverge from fp32 near a token-decision boundary. Policy v2 calls rows `exact`,
`accepted_numerical`, `review_required`, or `failed`. The existing 0.1-logit bound remains the
automatic acceptance boundary. A larger numerical difference needs a recorded review; structural
output failures and confirmed bugs fail. `review_required` is not shorthand for “the engine is
broken,” and it does not erase the measured timing.

For same-run A/B work, `exact` or explicitly reviewed `accepted_numerical` candidate/baseline
parity, plus equal token counts, can support the relative change even when both paths share a
pending fp32 review. Headline tok/s still requires `exact` or `accepted_numerical` reference
status. This keeps the gate useful without making benchmark archaeology the project's main job.

## Backends

- `esme`: headline path for Esme export bundles. Public examples use this backend.
- `dense`: compatibility alias for the same internal bundle loader.
- `qwen`: independent HF reference backend for `Qwen/Qwen2.5-Coder-3B-Instruct` at revision
  `488639f1ff808d1d3d0ba301aef8c11461451ec5`.

## What This Repo Proves

The repo is in its intended state when these remain true:

1. Esme bundle loading reproduces source logits/generation on fixed prompts.
2. The paged KV engine path passes the reference policy before reporting headline speed.
3. Continuous batching, chunked prefill, prefix caching, speculative decode, and preemption run
   through the shared engine path.
4. Esme benchmark docs compare naive HF and `llm_infer` on pinned hardware with every row
   reference-checked.
5. The Qwen reference stays reproducible as a correctness anchor.

## Non-Goals

- No production-readiness claim.
- No claim to approach mature production serving engines.
- No custom Triton/CUDA kernel in the current engine.
- No quantization or multi-GPU serving in the current engine.

## Layout

```text
llm_infer/
  model/          # backend registry, Esme bundle loader, Qwen reference backend
  kv_cache/       # block allocator, block tables, paged page store
  scheduler/      # prefill/decode admission, continuous batching
  kernels/        # AttentionBackend protocol + implementations
  serving/        # request queue, sampler, streaming loop, OpenAI server
  benchmarks/     # Esme benchmark helpers and shared report utilities
tests/correctness/ # reference and equivalence checks
docs/             # public docs
docs/internal/    # benchmark evidence and implementation notes
scripts/          # goldens, benchmarks, loadgen, trace fixtures
visualizer/       # schema-v3 KV trace replay UI
```

Every borrowed component lives behind a narrow interface with tests asserting it matches the
reference backend. The engine owns scheduling, page tables, block allocation, batching,
request state, sampling, benchmarking, and tracing.
