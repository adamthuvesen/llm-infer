# llm-infer Scope

`llm-infer` is a from-scratch paged LLM inference engine with pluggable model
backends. Qwen2.5-Coder is the pinned HuggingFace oracle backend; exported
`llm-pretrain` DenseBackbone bundles are a peer backend. The engine itself is the
work: correctness-first decoding, legible systems code, and honest benchmark
evidence. The `llm-rlvr-sql` rollout hook is one applied benchmark, not the
project's whole identity.

## Capability Statement

A from-scratch paged-inference engine with a generic model runtime: exact greedy
decoding for the pinned Qwen oracle, correctness-first DenseBackbone bundle
loading, paged KV cache, continuous batching, chunked prefill, prefix caching,
speculative decoding, request preemption, an OpenAI-compatible serving surface,
and replayable KV-cache traces. Benchmarks compare against naive HF and vLLM on
one pinned GPU.

The claim is not "beats vLLM" and not "production serving." It is honest systems
evidence: correctness before tok/s, canonical inference techniques made legible,
and every measured ceiling named.

## Evidence Boundary

The engine is release-quality for this repo when all of these hold:

1. Qwen2.5-Coder-3B greedy decoding is token-exact against HF on the unit
   correctness suite, and DenseBackbone bundle loading reproduces source
   logits/generation on fixed prompts.
2. Paged KV cache and continuous batching run end to end.
3. Benchmarks compare naive HF, `llm-infer`, and vLLM on a documented GPU with a
   pinned config.
4. Throughput is reported only for backends that pass the correctness gate.
5. The frozen `llm-rlvr-sql` rollout timing benchmark is measured once and written
   up with its limitations.
6. The docs explain the architecture, measurements, and known non-goals plainly.

## Build Order

| Area | Work | Why |
| --- | --- | --- |
| Trusted oracle | HF greedy baseline, `torch_naive` reference backend, single-request exact-match suite | get a truth oracle before optimizing |
| Systems loop | paged KV allocator, scheduler, two-request vertical slice | prove the loop before the benchmark table |
| Fast backend | `flash_attn_paged` behind the adapter | plug in the fast kernel only after the reference is trusted |
| Evidence | batch-correctness suite, three-way benchmark table | compare naive HF, `llm-infer`, and vLLM on pinned config |
| Applied rollout | one frozen `llm-rlvr-sql` rollout timing comparison and writeup | anchor the engine in real RL rollout economics |

The trusted oracle is the first gate, not a clean-up step after optimization. Every
backend passes the same oracle before it earns a speed number.

## Correctness

"Exact" includes chat template, RoPE position IDs, GQA head mapping, padding/mask,
and dtype. The checks are split into two surfaces:

1. **Unit path:** single request, no batching, token-for-token against HF greedy.
2. **Batch path:** batched prompts must match the same prompts run serially.

The bar is exact token IDs on the unit path. bf16 greedy can diverge on genuine
tie-break steps; that is acceptable only when each divergence is traced to a
numerical tie and documented. Otherwise the backend fails the oracle and reports
no tok/s.

## Applied Rollout Hook

The rollout benchmark replays `llm-rlvr-sql`'s actual inference call pattern rather
than a synthetic microbenchmark:

- checkpoint: a pinned SFT/GRPO weight set, merged before serving;
- slice: a fixed prompt set;
- metric: rollout tokens/s, rollout-batch wall clock, or dollars per 1k rollouts.

Without that connection, the timing number does not say anything useful about GRPO
economics.

## Non-Goals

- No production-readiness claim.
- No claim to beat vLLM.
- No custom Triton/CUDA kernel in the current engine.
- No quantization or multi-GPU serving in the current engine.
- DenseBackbone serving is correctness-first: it uses the engine API but does not
  yet implement true paged KV for dense bundles.

## Architecture

```text
llm_infer/
  model/          # backend registry, Qwen, DenseBackbone loader, rope utils
  kv_cache/       # block allocator, block tables, paged page store
  scheduler/      # prefill/decode admission, continuous batching
  kernels/        # AttentionBackend protocol + implementations
  serving/        # request queue, sampler, streaming loop, OpenAI server
  benchmarks/     # naive HF vs llm-infer vs vLLM
```

Every borrowed component lives behind a narrow interface with tests asserting it
matches the reference backend. The engine owns scheduling, page tables, block
allocation, batching, request state, sampling, benchmarking, and tracing.

## Trace Visualizer

The trace emitter lives in `llm_infer/tracing.py` and `InferenceEngine(trace=...)`.
The visualizer replays schema-versioned JSONL from the engine path, not a
hand-drawn simulation. A bundled synthetic fixture lets the viewer run without a
model or GPU while preserving the same event shapes.

## Forward Track

The main remaining engine work is technique completeness and clarity:

- finish sampling coverage (`repetition_penalty`, engine-level stop handling);
- extend DenseBackbone from correctness-first recompute toward true paged KV;
- expand the writeup around architecture and measured ceilings;
- consider quantization later as the real remaining speed lever.

Speed is useful evidence, but the project value is the correctness boundary and
the legibility of the engine.
