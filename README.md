# llm-infer

A minimal, honest paged LLM inference engine for **Qwen2.5-Coder-3B-Instruct**,
sequenced so the correctness oracle comes before anything fast or batched, and
benchmarked as an **rlvr-sql GRPO rollout backend**.

The differentiator is not "fast" — it is that every backend is proven exact against
HuggingFace greedy decoding before it reports a single tok/s, and the throughput is
measured on rlvr-sql's *actual* rollout workload, not a synthetic microbench. vLLM is
the ceiling, never the thing we beat; the gap to it is named, not hidden.

## Status — v1/v1.5 complete, v2 speed pass complete (measured ceiling)

| Phase | What | State |
|-------|------|-------|
| **A** trusted oracle | `torch_naive` reference + single-request token-for-token vs HF | ✅ |
| **B** systems milestone | paged KV allocator + continuous scheduler, two-request vertical slice | ✅ |
| **C** speed | `flash_attn_paged` behind the `AttentionBackend` adapter, gated by the oracle | ✅ |
| **D** evidence | batch-correctness suite + three-way benchmark ([`docs/benchmark.md`](docs/benchmark.md)) | ✅ |
| **E** differentiator | one frozen rlvr-sql rollout-timing comparison ([`docs/keeping-the-gpu-busy.md`](docs/keeping-the-gpu-busy.md)) | ✅ |
| **v2** speed pass | profiling, no-sync cleanup, packed KV reads/writes, read-plan reuse | ✅ 365.8 tok/s, ceiling named |
| **prefix caching** | refcounted KV-block sharing for G=4 rollout siblings | ✅ 411.5 tok/s, 3026-token path preserved |
| **chunked prefill** | bounded prompt chunks interleaved with active decode work | ✅ |
| **speculative decoding v1** | prompt-lookup n-gram draft + greedy verifier, no second model | ✅ technique/correctness evidence, no speed claim |
| **KV trace emitter MVP** | opt-in typed runtime trace recorder for real engine events | ✅ visualizer later |

The engine itself is the goal — a small, legible paged inference engine. Speed and the
rlvr-sql hook are the fun side-quest. The forward plan is **engine-first**: KV-cache-theater
visualizer from real traces → serving depth → expanded writeup, with quantization as a later
speed lever.
The decode-graph / static-bucket idea was built and rejected (a measured dead-end — see
[`docs/scoping.md`](docs/scoping.md)). See [`docs/scoping.md`](docs/scoping.md) for the
full plan and non-goals.

## Layout

Start with [`docs/architecture.md`](docs/architecture.md) for a guided map of modules, flows,
and diagrams.

- `kernels/` — the `AttentionBackend` protocol, a slow readable `torch_naive` reference
  (the truth every other backend is validated against), and `flash_attn_paged` (the fast
  GPU-only backend).
- `model/` — minimal Qwen2.5-Coder loading + greedy decode + cached prefill/decode.
- `kv_cache/` — block allocator, block tables, paged page store.
- `scheduler/` + `serving/` — continuous-batching admission and the decode loop, plus a
  seeded sampler (temperature 0 == the proven greedy oracle).
- `benchmarks/` — pure workload / runner / report pieces; the Modal harnesses live in
  `scripts/`.
- `tests/correctness/` — the oracle: single-request, token-for-token vs HuggingFace
  greedy, run against small committed golden fixtures.

## Results

**Phase D — synthetic three-way benchmark** (32 requests × 128 new tokens, greedy, A100-80GB,
run 2026-06-21). `llm_infer` beats the naive HF floor by **2.37×** (98.3 vs 41.5 tok/s) and is
the only engine besides vLLM whose every divergence from fp32 truth is a traced numerical tie.
vLLM (4323.6 tok/s) is the ceiling, ~44× ahead.

**Phase E — the real rlvr-sql rollout** (32-completion GRPO batch from rlvr-sql's own prompt
builders, merged `grpo-s0` LoRA → bf16, A100-80GB):

| system | tok/s | $/1k rollouts | vs floor |
|---|---|---|---|
| `hf_sequential` (floor) | 39.4 | $1.85 | 1.00× |
| **`llm_infer`** (ours) | **65.9** | **$1.00** | **1.67×** |
| `vllm` (ceiling) | 2170.6 | $0.03 | ~33× ahead |

The fused batched decode (`decode_many` — all running requests advance in one forward per
step) is what earns the win over naive sequential HF. The ~33× gap to vLLM is the cost of
v1's legibility (vLLM has CUDA graphs, a custom in-place paged kernel, a mature scheduler) —
named in [`docs/keeping-the-gpu-busy.md`](docs/keeping-the-gpu-busy.md), not hidden.

**v2 speed pass — complete, with a measured ceiling** (same frozen rlvr-sql rollout,
A100-80GB PCIe, run 2026-06-21, pinned `bench-results/rollout-rollout-20260621T164905.json`):
`llm_infer` reaches **365.8 tok/s** and **$0.18 / 1k rollouts** after profiling/no-sync
cleanup, packed KV reads, vectorized KV writes, and read-plan reuse — about **5.55×** over the
65.9 tok/s v1.5 baseline. This is a finished chapter, not a paused one: the cheap
KV-materialization wins are harvested and the ceiling is named — the post-cleanup profile shows
`kv_read_gather` is no longer the wall, and the v2 toolkit (no-sync + KV vectorization) cannot
move the remaining decode-orchestration / projection-MLP / attention time further.

Two v2 directions were tried and **rejected with evidence**, not left as TODOs:

- **Decode-graph / static 32-slot bucket — a measured dead-end.** Built and rejected: it was
  *slower* (a static bucket pays for variable-occupancy device compute the eager path skips —
  256 vs 299 tok/s) **and** it shifted the sampled token count (3036 vs 3026, bf16
  batch-composition drift). Any future decode-graph attempt must clear two gates: (a)
  token-identical on GPU bf16, not just CPU fp32; (b) attack variable-occupancy *device*
  compute, not host-launch overhead. The CUDA-graph axis is closed for this workload.
- **Kernel/fusion shapes** — direct FlashAttention GQA, projection/MLP fusion, step-local
  prompt-prefix KV copying, and no-gather paged KV attention all changed the sampled token path
  or regressed speed. Do not retry without a new profile-backed reason.

**Prefix caching evidence — accepted token path preserved** (same frozen rollout, run
2026-06-22, pinned `bench-results/rollout-rollout-20260622T173732.json`): refcounted prompt
block sharing for known G=4 siblings keeps `llm_infer` at the accepted **3026** sampled tokens
while reducing prompt prefill token-ops from **15,416** to **3,854**. The engine reaches
**411.5 tok/s** and **$0.16 / 1k rollouts**, improving over the prior 365.8 tok/s / $0.18
baseline.

## Roadmap

The engine is the goal, so the forward plan is sequenced by **technique-completeness and
legibility**. Speed is a side-quest with its own track, last.

**Done**

1. **v1 / v1.5 — correct minimal engine + differentiator.** HF oracle, paged KV, continuous
   batching, three-way benchmark, and the frozen rlvr-sql rollout proof + writeup.
2. **v2 — speed pass (complete, ceiling named).** 365.8 tok/s / 5.55× over baseline; cheap
   KV-materialization wins harvested, ceiling measured. The decode-graph / static-bucket idea
   was built and **rejected** (a measured dead-end — slower, and it shifted the sampled token
   count); the CUDA-graph axis is closed for this workload.
3. **Prefix caching.** Refcounted KV-block sharing across known sibling completions of the
   same prompt. Full prompt blocks are shared by pointer; only the last partial prompt block
   copy-on-writes on first generated-token append. Frozen rollout sampled token count remains
   3026.
4. **Chunked prefill / mixed prefill-decode.** Active decode work advances between bounded
   prompt chunks instead of waiting for every prompt to fully prefill.
5. **Speculative decoding v1.** A prompt-lookup / n-gram draft source proposes short drafts
   copied from tokens already present in the prompt/history. The main model verifies
   `last_token + draft` in one cached forward, accepts only the matching greedy prefix, and
   falls back to the verifier's next token on mismatch or no draft. It is off by default and
   supports greedy decoding only; sampled rollouts keep the existing path. This is evidence
   that the draft/verify technique is wired correctly, not a claimed speed win.
6. **KV-cache-theater trace emitter MVP.** `InferenceEngine(trace=...)` emits typed,
   schema-versioned events from the real request path: request admission, prefill start and
   progress, decode steps, request finish, batch-size changes, and throughput samples.

**Primary forward track — engine technique-completeness + legibility**

7. **KV-cache-theater visualizer.** Build the separate visualizer only from real engine traces;
   add block lifecycle views after the cache exposes clean allocation/free hooks.
8. **Serving depth.** Streaming, an OpenAI-compatible endpoint, metrics, and a load generator.
9. **Expand *Keeping the GPU Busy*.** Narrate the architecture and the honest dead-ends.

**Secondary track — speed, later, for fun**

9. **Quantization** (gpt-fast style: projection/MLP matmuls + KV bandwidth) is the real
   remaining lever. **650 tok/s is a checkpoint quantization may clear, not a goal to grind
   toward.** Backend/kernel depth (FlexAttention, Triton, no-gather paged) stays parked here.

## Quickstart

```bash
uv sync --extra dev
uv run ruff check
uv run pytest tests/correctness -q
```

The CPU oracle is the local gate and needs no GPU — it checks committed golden token ids.
The flash-attn backend is the one GPU-only path; its oracle runs on the target GPU via
`scripts/modal_oracle.py`. The benchmark and rollout harnesses (`scripts/modal_benchmark.py`,
`scripts/modal_rollout.py`) run on Modal A100-80GB. Regenerating goldens
(`scripts/generate_goldens.py`) loads the 3B model in fp32 on CPU.

See [`AGENTS.md`](AGENTS.md) for the working agreement and the honesty bar.

## Model pin

`Qwen/Qwen2.5-Coder-3B-Instruct` at HF revision
`488639f1ff808d1d3d0ba301aef8c11461451ec5` (the Instruct variant — see
`llm_infer/model/config.py`). Plain `-3B` is a different model and would be wrong.
