# llm-infer — scoping doc

A from-scratch, honest paged LLM inference engine with **pluggable model backends**. Qwen2.5-Coder is the pinned HF-oracle backend; exported `llm-pretrain` DenseBackbone bundles are a peer backend. The engine itself is the goal: learn how a real paged inference engine is built and show it clearly, with code and presentation held to a staff-engineer bar. Honest explanation comes before speed; the llm-rlvr-sql rollout hook is one applied benchmark, not the identity.

## The claim (capability statement, not a number)

> A from-scratch paged-inference engine with a generic model runtime — exact greedy decoding for the pinned Qwen oracle, correctness-first DenseBackbone bundle loading, paged KV-cache + continuous batching, chunked prefill, prefix caching, and speculative decoding; benchmarked against naive HF and vLLM on one GPU; replayable in a KV-cache trace visualizer.

Not "beats vLLM." Not "production serving." Honest systems evidence. The differentiator is the **methodology and completeness** — correctness proven before any tok/s, the canonical techniques built and made legible, every ceiling and dead-end named — not a single number. The llm-rlvr-sql rollout is one applied benchmark that grounds it in real RL economics, not what makes it *yours*.

## Hard stop (evidence-based, ship v1 when ALL hold)

1. Qwen2.5-Coder-3B **greedy decoding is exact** (same token ids as HF) on the unit correctness suite, and DenseBackbone bundle loading reproduces source logits/generation on fixed prompts.
2. Paged KV-cache + continuous batching run **end-to-end**.
3. Benchmarks compare **naive HF vs llm-infer vs vLLM** on one documented GPU, config pinned.
4. The optimized path **beats the naive baseline by a clear, honestly-reported margin** (directional — no target %, no vLLM-relative goal).
5. **llm-rlvr-sql rollout timing measured once** (ugly/unoptimized is fine).
6. Short writeup.

Result is whatever it is. 35% of vLLM, well-measured and explained, is still valuable. Everything beyond this list is expansion, not v1.

Stops 1–4 are the **internal v1 engine milestone**; **public release waits for #5–6 (v1.5)** — the llm-rlvr-sql hook is the unique angle, so it ships *with* the engine, not after.

## v1 build order (sequenced for fast falsification)

| Phase | Tasks | Why |
|-------|-------|-----|
| **A — trusted oracle** | HF greedy baseline · `torch_naive` reference backend · **single-request exact-match suite** | get a truth oracle before anything fast or batched |
| **B — systems milestone** | paged KV allocator · continuous scheduler → **two-request vertical slice** | prove the loop before the table |
| **C — speed** | `flash_attn_paged` behind the adapter, gated by the oracle | plug the fast kernel only once the reference is trusted |
| **D — evidence** | batch-correctness suite · three-way benchmark table | naive HF vs llm-infer vs vLLM, config pinned |
| **E — differentiator (v1.5)** | one frozen llm-rlvr-sql rollout timing comparison · writeup *Keeping the GPU Busy* | anchors the claim; survives an early stop |

**Correctness gate is Phase A**, not after the fast kernel — every backend passes the same oracle from day one, so you never debug "scheduler or kernel?" without a trusted reference.

**The Phase B vertical slice is proof-of-life — ship it before the benchmark table:**
```
admit 2 requests → prefill both → decode N steps → one finishes → admit a third → continue
```
If that loop runs and stays correct, v1 is real.

**Day-one gates:** only the repo skeleton (AttentionBackend protocol + `torch_naive` + unit oracle) and the correctness-fixture format block code-start. The llm-rlvr-sql workload spec below is a **week-1 parallel task** that gates Phase E, not branching.

## Correctness (two suites, precise bar)

"Exact" has surface area — chat template, RoPE position-ids (advance **per request** under paged decode, not per batch row), GQA head mapping, padding/mask, dtype. Split it:

1. **Unit path** — single request, no batching, **token-for-token vs HF** greedy. This is the oracle.
2. **Batch path** — same prompts batched vs run serially; batched output must equal serial greedy.

**The bar, honestly:** exact token-ids on the unit path. Pursue exactness, but bf16 greedy can diverge from HF on genuine tie-break steps — acceptable **only** if each divergence is traced to a numerical tie (equal-to-tolerance logits) and documented, never waved off as "close enough." If ties get noisy, run the unit oracle in **fp32** where they near-vanish. This keeps hard-stop #1 falsifiable instead of an infinite debug. Pin as fixtures: `transformers`, `torch`, dtype, `do_sample=False`/`temperature=0`, and **the exact llm-rlvr-sql prompt builder**.

## llm-rlvr-sql timing hook — workload spec (draft week 1)

One frozen workload replaying llm-rlvr-sql's *actual* inference call pattern (its Modal/vLLM eval path / `forced_revision_generate`), not a synthetic microbench:
- **Checkpoint:** which llm-rlvr-sql SFT/GRPO weights.
- **Slice:** a fixed prompt set (e.g. 32 Spider prompts, or one GRPO rollout batch shape).
- **Primary metric:** one of — rollout tokens/s · rollout-batch wall-clock · $/1k rollouts.

Without this, the timing number won't connect to GRPO economics.

## v1 non-goals (historical scope firewall)

These were explicit **v1** out-of-scope items. Several are now implemented; this section
is kept so old planning docs do not look like current truth:

- ~~no prefix caching · no chunked prefill · no OpenAI-compatible server~~ — **done**
- still out: custom Triton/CUDA kernel · quantization · multi-GPU · production-readiness claim

**Original v1 continuous batching (narrow):** at decode-step boundaries, finished requests
leave and queued requests may enter. Chunked prefill and mixed prefill/decode are now implemented.

## Architecture (modular from day one; borrowed parts behind narrow adapters)

```
llm_infer/
  model/          # Qwen loading, weights, tokenizer, config
  kv_cache/       # block allocator, block tables, page metadata
  scheduler/      # prefill/decode admission, continuous batching
  kernels/        # AttentionBackend protocol + implementations
  serving/        # request queue, sampler, streaming loop
  benchmarks/     # naive vs llm-infer vs vLLM
```

```
kernels/
  base.py             # AttentionBackend protocol
  torch_naive.py      # slow, readable reference (truth)
  flash_attn_paged.py # borrowed fast backend (v1)
  # later: flex_attention.py, triton_paged.py, cuda_paged.cu
```

**Design rule:** every borrowed component lives behind a narrow interface with a test asserting it matches the reference backend (logits / output shape / cache behavior). Replacing it later is a swap, not a rewrite. The engine owns scheduling, page tables, block allocation, batching, request state, sampling, benchmarking — regardless of which kernel is plugged in.

## Honesty discipline (the portfolio asset)

- **Correctness exact at the base.** Reference = HF greedy token ids. No "correct-ish."
- **vLLM is the ceiling, never the thing you beat.** Brag = "reaches X% of vLLM with Y lines and this feature subset."
- **Pin the full benchmark config:** GPU + power/clock state (or note "best-effort on spot"), **vLLM version and flags** (`--max-num-seqs`, `--gpu-memory-utilization`), seq lengths, sampling, warmup. Reproducible command.
- **Define "naive HF" out loud** — per-request `generate()` vs batched `forward()` — so a too-weak baseline can't read as sandbagging.
- **Adapter tests go beyond logits:** block-table indexing (allocate → append token → free → reallocate after finish), not just output match.
- **Validate before you brag.** A backend that fails the oracle reports no tok/s. Throughput numbers only for correct backends.
- **Benchmark equivalence across all three systems:** same prompts, `max_new_tokens`, stop tokens, greedy decoding, warmup/measurement windows.
- **No silent gaming.** The directional floor is naive-baseline-relative, not vLLM-relative.

## KV Cache Theater (primary track — trace-driven)

The engine-side trace emitter lives in `llm_infer/tracing.py` and `InferenceEngine(trace=...)`.
The visualizer replays **real schema-v3 traces** from that path, not a hand-drawn simulation —
the engine trace is the artifact; the visualizer is a debugger + proof view, not decoration. The
bundled demo fixture is a labelled synthetic sample emitted in the same schema by a standalone
generator, so the viewer can ship and run without the engine, a model, or a GPU.

Current trace contract: schema version **3**, emitted as JSON Lines from `TraceRecorder`.
Events are typed and stored in engine emission order:

`request_admitted · prefill_chunk_started · prefill_chunk_progress · decode_step · block_allocated · block_freed · request_finished · batch_size_changed · tokens_per_second_sampled`

Prefill events are **chunk-scoped**, not once-per-request. A long prompt can emit several
`prefill_chunk_started` / `prefill_chunk_progress` pairs before its first sampled token:

```json
{"event":"request_admitted","schema_version":3,"sequence":1,"step":0,"request_id":"long","prompt_tokens":5,"max_new_tokens":2,"reserved_blocks":2}
{"event":"prefill_chunk_started","schema_version":3,"sequence":4,"step":0,"request_id":"long","start_pos":0,"end_pos":2,"total_prompt_tokens":5}
{"event":"prefill_chunk_progress","schema_version":3,"sequence":5,"step":0,"request_id":"long","start_pos":0,"end_pos":2,"cached_tokens":2,"total_prompt_tokens":5,"completed":false}
{"event":"prefill_chunk_started","schema_version":3,"sequence":8,"step":1,"request_id":"long","start_pos":2,"end_pos":4,"total_prompt_tokens":5}
{"event":"prefill_chunk_progress","schema_version":3,"sequence":9,"step":1,"request_id":"long","start_pos":2,"end_pos":4,"cached_tokens":4,"total_prompt_tokens":5,"completed":false}
{"event":"prefill_chunk_started","schema_version":3,"sequence":12,"step":2,"request_id":"long","start_pos":4,"end_pos":5,"total_prompt_tokens":5}
{"event":"prefill_chunk_progress","schema_version":3,"sequence":13,"step":2,"request_id":"long","start_pos":4,"end_pos":5,"cached_tokens":5,"total_prompt_tokens":5,"completed":true}
```

Visualizer handoff contract: consume only schema-versioned JSONL events from the real engine;
order by `sequence`; group per-request work by `request_id`; use `step` only as the engine loop
tick; use `start_pos`/`end_pos`/`cached_tokens` for prompt chunk rendering; use `request_ids`,
`token_ids`, `token_source`, and `tokens_emitted` for generated-token rows; and treat absent
optional fields as absent, not zero. `decode_step` is the generated-token trace event even when
the token source is `prefill`: the first token sampled after a completed prompt prefill must be
represented there before it appears in `request_finished.token_ids`.

Block lifecycle is emitted honestly. `block_allocated` / `block_freed` come from an observer
on `BlockAllocator` — the single owner of the free pool — not from higher-level request state. A
block is `block_allocated` only when it leaves the pool (copy-on-write inside
`PagedKVCache.prepare_write()` counts, because it allocates a new physical block) and
`block_freed` only when it truly returns (refcount-0), so a prefix-shared block retained by a
sibling is never reported freed until its last owner releases it. Block deltas are attributed to
the owning request via `BlockTable.owner`; each event carries `request_id`, `block_count`,
`block_ids`, and the post-change `pool_used` / `pool_free` totals.

## Expansion path (post-v1, engine-first)

v1/v1.5 are complete and v2 (the speed pass) is a closed chapter. The engine is the goal, so
the forward plan is sequenced by **technique-completeness and legibility**, not by chasing a
throughput number. Speed is the fun side-quest; it gets its own track, last.

### Done

- **v1 / v1.5 — correct minimal engine + applied benchmark.** HF-exact greedy oracle, paged
  KV-cache + continuous batching end-to-end, the flash backend behind the `AttentionBackend`
  adapter, the three-way benchmark, and the frozen llm-rlvr-sql rollout hook + *Keeping the GPU
  Busy* writeup.
- **v2 — the speed pass (complete, with a measured ceiling).** Profiling/no-sync cleanup,
  packed KV reads, vectorized KV writes, and read-plan reuse moved the frozen llm-rlvr-sql rollout
  from the v1.5 baseline of **65.9 tok/s** to **365.8 tok/s** on A100-80GB PCIe — about
  **5.55x** over baseline (pinned: `bench-results/rollout-rollout-20260621T164905.json`). This
  is **not "paused mid-campaign"** — the cheap KV-materialization wins are harvested and the
  ceiling is named: the post-cleanup profile shows `kv_read_gather` is no longer the wall;
  remaining time is decode orchestration, projections/MLP, attention, prefill, KV writes, and
  GQA expansion, none of which the v2 toolkit (no-sync + KV vectorization) can move further.
- **Prefix caching — refcounted block sharing for rollout siblings.** The canonical prefix-cache
  technique shares full prompt blocks by pointer across known G=4 sibling completions,
  with refcounted physical ownership and copy-on-write only for the final partial prompt block.
  Frozen rollout evidence preserved the accepted **3026** sampled tokens and improved the
  engine from **365.8 tok/s / $0.18 per 1k rollouts** to **411.5 tok/s / $0.16 per 1k
  rollouts**, with prompt prefill token-ops reduced from **15,416** to **3,854** (pinned:
  `bench-results/rollout-rollout-20260622T173732.json`).
- **Chunked prefill / mixed prefill-decode.** Active decode requests keep advancing between
  bounded prompt chunks for not-yet-prefilled requests, so a long prompt prefills in bounded
  chunks instead of fully blocking decode, and the cached-token path is preserved.
- **Speculative decoding v1 — prompt-lookup draft + greedy verifier.** The engine can use an
  n-gram / prompt-lookup draft source and verify `last_token + draft` in one cached forward,
  accepting only the greedy-matching prefix and falling back safely. This is correctness and
  technique evidence only: it is off by default, greedy-only, uses no second model, and makes
  no speed claim.
- **KV-cache-theater trace visualizer.** The engine has an opt-in typed recorder for
  schema-versioned runtime events emitted from the real `InferenceEngine` path, plus a static
  local visualizer in `visualizer/` that renders schema-v3 JSONL as request lanes, prefill
  chunks, decode emissions, batch/throughput signals, event inspection, playback, and a paged
  KV wall driven by **real block allocation/free** (filled blocks physically held, the
  scheduler's reservation shown as headroom). The committed fixture at
  `docs/assets/kv_trace_schema_v3.jsonl` is a labelled **synthetic sample** produced by a
  standalone generator that emits the engine's schema-v3 event shapes (no engine/model/GPU
  dependency), so the visualizer ships on its own; the viewer can equally replay real
  `InferenceEngine(trace=...)` traces.
- **Block allocation/free trace hooks.** `block_allocated` / `block_freed` are emitted from an
  observer on `BlockAllocator` — the physical free-pool boundary — so they stay honest for
  refcounted prefix sharing (a retained block is not a free) and copy-on-write (a new physical
  block is an allocation). This is the prerequisite the preemption demo below needs.

### Measured dead-end — the CUDA-graph / static-bucket axis is closed for this workload

The decode-graph / static 32-slot bucket idea is a measured dead-end, not deferred work. A
static bucket was measured against the eager path and lost on two independent counts:

1. **Slower.** The static bucket pays for fixed worst-case occupancy every step; the real
   workload has variable occupancy, so the bucket does device compute the eager path skips.
   Measured 256 tok/s (SXM4) vs 299 tok/s (eager, PCIe) — capture cannot recover this because
   the cost is device compute, not host-launch overhead.
2. **It shifted the sampled token count** — 3036 vs the accepted 3026 — from bf16
   batch-composition drift, breaking the correctness gate.

Two forward gates any future decode-graph attempt **must** clear before it earns time: (a)
token-identical on GPU bf16, not just CPU fp32; (b) it must attack variable-occupancy *device*
compute, not host-launch overhead (the thing CUDA graphs remove, which this workload is not
bound by). Absent both, the CUDA-graph axis stays closed.

### Primary forward track — engine technique-completeness + legibility

The point of the project. Each item is a canonical inference-engine technique the engine does
not yet have, in rough dependency order:

- **Request preemption / eviction** *(done)* — opt-in (`InferenceEngine(preemption=True)`).
  Admission over-commits the pool on each request's current footprint instead of reserving its
  worst case; when a running request then needs a block the pool cannot give, the engine evicts
  the most-recently-admitted request (LIFO — it has decoded the fewest tokens, so recompute waste
  is minimal), frees its KV honestly, keeps its generated tokens, and resumes it later by
  recomputing prompt-plus-generated. Forward progress is guaranteed because admission already
  rejects any request that cannot fit the empty pool alone. A forced-preemption oracle pins
  token-for-token identity with the uninterrupted run; the visualizer renders the eviction and
  recompute resume, and the bundled fixture shows one. Default behavior (preemption off) is the
  unchanged strict-reservation scheduler.
- **Serving depth** *(done)* — OpenAI-compatible HTTP (`python -m llm_infer.serve`), streaming,
  `/metrics`, and `scripts/loadgen.py`.
- **Sampling completeness** *(partial)* — `top_k`, frequency/presence penalties, and HTTP-layer
  stop strings are done; `repetition_penalty` and engine-level stop remain open.
- **Expand *Keeping the GPU Busy*** — grow the writeup into a narration of the architecture and
  the honest dead-ends (the v2 ceiling and the decode-graph rejection above), so the doc
  teaches the engine, not just the one rollout number.

### Secondary track — speed, later, for fun

Picked up only when the primary track wants a breather. **Quantization is the real remaining
lever** (gpt-fast style: the projection/MLP matmuls plus KV bandwidth), and it is the one
technique plausibly able to clear the next checkpoint. **650 tok/s is a checkpoint quantization
*may* clear, not a goal to grind toward** — the engine's value is its completeness and clarity,
not that number.

Backend/kernel depth (FlexAttention, Triton, a true no-gather paged backend) stays parked here
too: the v2 loop already rejected direct FlashAttention GQA, projection/MLP fusion, step-local
prompt-prefix KV copying, and no-gather paged KV attention for changing the sampled token path
or regressing speed — do not retry those shapes without new profile evidence and a stricter
acceptance story.

### Release

Release when the primary track has filled out the engine's technique set and the writeup
narrates the architecture and the measured ceilings honestly — the speed number is whatever it
is by then.

## Repos (hub-and-spoke, not a monorepo)

- Engine name: **`llm-infer`** (descriptive and unambiguous). v1.5 is a **standalone engine + thin llm-rlvr-sql adapter script** (not a package import), preserving hub-and-spoke.
- `kv-cache-theater` — only if trace-driven; separate.
- `keeping-the-gpu-busy` — writeup/hub, later.

## Cost

Anchored to $130 for GRPO+RLVR (~$1.5/hr spot). v1 retrains nothing — runs on existing llm-rlvr-sql checkpoints at inference prices. Develop on a cheap GPU, benchmark on target, measure throughput on short runs, stay 3B/single-GPU. v1 ≈ **$30–55**. Full expansion incl. RL convergence re-runs ≈ up to ~$300 (one convergence run at the end, not per-experiment).

## References

- Build target: nano-vLLM (github.com/GeeeekExplorer/nano-vllm) — study, don't fork.
- Kernel-from-scratch: MinivLLM (github.com/Wenyueh/MinivLLM).
- FlexAttention path: flex-nano-vllm (github.com/changjonathanc/flex-nano-vllm).
- Concepts: Inside vLLM anatomy (blog.vllm.ai/2025/09/05/anatomy-of-vllm.html); KV cache from scratch — Raschka (magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms).
- Latency/quant reference: gpt-fast (github.com/pytorch-labs/gpt-fast).
- Model: [Qwen2.5-Coder-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct).
