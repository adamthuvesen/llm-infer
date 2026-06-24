# llm-infer — scoping doc

A minimal, honest paged LLM inference engine for **Qwen2.5-Coder-3B-Instruct**, measured as an **rlvr-sql rollout backend**. The engine itself is the goal — learning and showing how a small paged inference engine is built. Beautiful explanation second; speed and the rlvr-sql hook are the *fun side-quest*, not the point.

## The claim (capability statement, not a number)

> A small paged-inference engine for Qwen2.5-Coder-3B — exact greedy decoding vs HF, paged KV-cache + continuous batching running end-to-end, benchmarked against naive HF and vLLM on one GPU, with an early rlvr-sql rollout timing hook.

Not "beats vLLM." Not "production serving." Honest systems evidence. The differentiator is the RL-rollout measurement — that is what makes it *yours* rather than a generic vLLM clone.

## Hard stop (evidence-based, ship v1 when ALL hold)

1. Qwen2.5-Coder-3B **greedy decoding is exact** (same token ids as HF) on the unit correctness suite (fp32 fallback if bf16 tie-breaks get noisy — see Correctness).
2. Paged KV-cache + continuous batching run **end-to-end**.
3. Benchmarks compare **naive HF vs llm-infer vs vLLM** on one documented GPU, config pinned.
4. The optimized path **beats the naive baseline by a clear, honestly-reported margin** (directional — no target %, no vLLM-relative goal).
5. **rlvr-sql rollout timing measured once** (ugly/unoptimized is fine).
6. Short writeup.

Result is whatever it is. 35% of vLLM, well-measured and explained, is still valuable. Everything beyond this list is expansion, not v1.

Stops 1–4 are the **internal v1 engine milestone**; **public release waits for #5–6 (v1.5)** — the rlvr-sql hook is the unique angle, so it ships *with* the engine, not after.

## v1 build order (sequenced for fast falsification)

| Phase | Tasks | Why |
|-------|-------|-----|
| **A — trusted oracle** | HF greedy baseline · `torch_naive` reference backend · **single-request exact-match suite** | get a truth oracle before anything fast or batched |
| **B — systems milestone** | paged KV allocator · continuous scheduler → **two-request vertical slice** | prove the loop before the table |
| **C — speed** | `flash_attn_paged` behind the adapter, gated by the oracle | plug the fast kernel only once the reference is trusted |
| **D — evidence** | batch-correctness suite · three-way benchmark table | naive HF vs llm-infer vs vLLM, config pinned |
| **E — differentiator (v1.5)** | one frozen rlvr-sql rollout timing comparison · writeup *Keeping the GPU Busy* | anchors the claim; survives an early stop |

**Correctness gate is Phase A**, not after the fast kernel — every backend passes the same oracle from day one, so you never debug "scheduler or kernel?" without a trusted reference.

**The Phase B vertical slice is proof-of-life — ship it before the benchmark table:**
```
admit 2 requests → prefill both → decode N steps → one finishes → admit a third → continue
```
If that loop runs and stays correct, v1 is real.

**Day-one gates:** only the repo skeleton (AttentionBackend protocol + `torch_naive` + unit oracle) and the correctness-fixture format block code-start. The rlvr-sql workload spec below is a **week-1 parallel task** that gates Phase E, not branching.

## Correctness (two suites, precise bar)

"Exact" has surface area — chat template, RoPE position-ids (advance **per request** under paged decode, not per batch row), GQA head mapping, padding/mask, dtype. Split it:

1. **Unit path** — single request, no batching, **token-for-token vs HF** greedy. This is the oracle.
2. **Batch path** — same prompts batched vs run serially; batched output must equal serial greedy.

**The bar, honestly:** exact token-ids on the unit path. Pursue exactness, but bf16 greedy can diverge from HF on genuine tie-break steps — acceptable **only** if each divergence is traced to a numerical tie (equal-to-tolerance logits) and documented, never waved off as "close enough." If ties get noisy, run the unit oracle in **fp32** where they near-vanish. This keeps hard-stop #1 falsifiable instead of an infinite debug. Pin as fixtures: `transformers`, `torch`, dtype, `do_sample=False`/`temperature=0`, and **the exact rlvr-sql prompt builder**.

## rlvr-sql timing hook — workload spec (draft week 1)

One frozen workload replaying rlvr-sql's *actual* inference call pattern (its Modal/vLLM eval path / `forced_revision_generate`), not a synthetic microbench:
- **Checkpoint:** which rlvr-sql SFT/GRPO weights.
- **Slice:** a fixed prompt set (e.g. 32 Spider prompts, or one GRPO rollout batch shape).
- **Primary metric:** one of — rollout tokens/s · rollout-batch wall-clock · $/1k rollouts.

Without this, the timing number won't connect to GRPO economics.

## v1 non-goals (the scope firewall)

Explicitly **out** of v1 — naming them keeps the spec from quietly expanding:

- no custom Triton/CUDA kernel · no quantization · no prefix caching · no chunked prefill
- no OpenAI-compatible server · no streaming API · no multi-GPU
- no production-readiness claim

**v1 continuous batching, narrowly:** at decode-step boundaries, finished requests leave and queued requests may enter. v1 does **not** require chunked prefill or mixed prefill/decode optimization.

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

The engine-side trace emitter MVP lives in `llm_infer/tracing.py` and
`InferenceEngine(trace=...)`. Build the visualizer only if it replays **real engine events**,
not a simulation. The engine trace is the artifact; the visualizer is a debugger + proof view,
not decoration. Separate repo / app, fed by llm-infer traces.

Current trace contract: schema version **2**, emitted as JSON Lines from `TraceRecorder`.
Events are typed and stored in engine emission order:

`request_admitted · prefill_chunk_started · prefill_chunk_progress · decode_step · request_finished · batch_size_changed · tokens_per_second_sampled`

Prefill events are **chunk-scoped**, not once-per-request. A long prompt can emit several
`prefill_chunk_started` / `prefill_chunk_progress` pairs before its first sampled token:

```json
{"event":"request_admitted","schema_version":2,"sequence":1,"step":0,"request_id":"long","prompt_tokens":5,"max_new_tokens":2,"reserved_blocks":2}
{"event":"prefill_chunk_started","schema_version":2,"sequence":4,"step":0,"request_id":"long","start_pos":0,"end_pos":2,"total_prompt_tokens":5}
{"event":"prefill_chunk_progress","schema_version":2,"sequence":5,"step":0,"request_id":"long","start_pos":0,"end_pos":2,"cached_tokens":2,"total_prompt_tokens":5,"completed":false}
{"event":"prefill_chunk_started","schema_version":2,"sequence":8,"step":1,"request_id":"long","start_pos":2,"end_pos":4,"total_prompt_tokens":5}
{"event":"prefill_chunk_progress","schema_version":2,"sequence":9,"step":1,"request_id":"long","start_pos":2,"end_pos":4,"cached_tokens":4,"total_prompt_tokens":5,"completed":false}
{"event":"prefill_chunk_started","schema_version":2,"sequence":12,"step":2,"request_id":"long","start_pos":4,"end_pos":5,"total_prompt_tokens":5}
{"event":"prefill_chunk_progress","schema_version":2,"sequence":13,"step":2,"request_id":"long","start_pos":4,"end_pos":5,"cached_tokens":5,"total_prompt_tokens":5,"completed":true}
```

Visualizer handoff contract: consume only schema-versioned JSONL events from the real engine;
order by `sequence`; group per-request work by `request_id`; use `step` only as the engine loop
tick; use `start_pos`/`end_pos`/`cached_tokens` for prompt chunk rendering; use `request_ids`,
`token_ids`, and `tokens_emitted` for decode rows; and treat absent optional fields as absent,
not zero.

Block lifecycle remains a deliberate follow-up: `block_allocated` / `block_freed` need a clean
request-aware cache hook around `BlockTable.reserve()` / `BlockTable.free()` (and copy-on-write
inside `PagedKVCache.prepare_write()`). Do not fake those events from higher-level request
state, because prefix-shared blocks can be retained or released without returning to the free
pool.

## Expansion path (post-v1, engine-first)

v1/v1.5 are complete and v2 (the speed pass) is a closed chapter. The engine is the goal, so
the forward plan is sequenced by **technique-completeness and legibility**, not by chasing a
throughput number. Speed is the fun side-quest; it gets its own track, last.

### Done

- **v1 / v1.5 — correct minimal engine + differentiator.** HF-exact greedy oracle, paged
  KV-cache + continuous batching end-to-end, the flash backend behind the `AttentionBackend`
  adapter, the three-way benchmark, and the frozen rlvr-sql rollout hook + *Keeping the GPU
  Busy* writeup.
- **v2 — the speed pass (complete, with a measured ceiling).** Profiling/no-sync cleanup,
  packed KV reads, vectorized KV writes, and read-plan reuse moved the frozen rlvr-sql rollout
  from the v1.5 baseline of **65.9 tok/s** to **365.8 tok/s** on A100-80GB PCIe — about
  **5.55x** over baseline (pinned: `bench-results/rollout-rollout-20260621T164905.json`). This
  is **not "paused mid-campaign"** — the cheap KV-materialization wins are harvested and the
  ceiling is named: the post-cleanup profile shows `kv_read_gather` is no longer the wall;
  remaining time is decode orchestration, projections/MLP, attention, prefill, KV writes, and
  GQA expansion, none of which the v2 toolkit (no-sync + KV vectorization) can move further.
- **Prefix caching — refcounted block sharing for rollout siblings.** The canonical prefix-cache
  technique now shares full prompt blocks by pointer across known G=4 sibling completions,
  with refcounted physical ownership and copy-on-write only for the final partial prompt block.
  Frozen rollout evidence preserved the accepted **3026** sampled tokens and improved the
  engine from **365.8 tok/s / $0.18 per 1k rollouts** to **411.5 tok/s / $0.16 per 1k
  rollouts**, with prompt prefill token-ops reduced from **15,416** to **3,854** (pinned:
  `bench-results/rollout-rollout-20260622T173732.json`).
- **Chunked prefill / mixed prefill-decode.** Active decode requests now keep advancing between
  bounded prompt chunks for not-yet-prefilled requests, preserving the cached-token path while
  removing the old "prefill fully, then decode" split for long prompts.
- **Speculative decoding v1 — prompt-lookup draft + greedy verifier.** The engine can use an
  n-gram / prompt-lookup draft source and verify `last_token + draft` in one cached forward,
  accepting only the greedy-matching prefix and falling back safely. This is correctness and
  technique evidence only: it is off by default, greedy-only, uses no second model, and makes
  no speed claim.
- **KV-cache-theater trace visualizer MVP.** The engine now has an opt-in typed recorder for
  schema-versioned runtime events emitted from the real `InferenceEngine` path, plus a static
  local visualizer in `visualizer/` that renders schema-v2 JSONL as request lanes, prefill
  chunks, decode emissions, batch/throughput signals, event inspection, playback, and honest
  scheduler-reservation/cache-pressure views. The committed fixture at
  `docs/assets/kv_trace_schema_v2.jsonl` is generated through `InferenceEngine(trace=...)`.

### Measured dead-end — the CUDA-graph / static-bucket axis is closed for this workload

The decode-graph / static 32-slot bucket idea (once framed as "v3 next") was **built and
rejected**, not deferred. A static bucket was measured against the eager path and lost on two
independent counts:

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

- **Block allocation/free trace hooks** — extend KV-cache-theater only after the cache has a
  clean request-aware lifecycle hook (see the *KV Cache Theater* section above).
- **Serving depth** — streaming, an OpenAI-compatible endpoint, metrics, and a load generator,
  so the engine is drivable as a real server rather than only through the frozen harness. This
  is the final engine piece — it makes the engine drivable as a real server.
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

- Engine name: **`llm-infer`** (settled — descriptive and unambiguous; the earlier working names `paged` and `nano-infer` are retired). v1.5 is a **standalone engine + thin rlvr-sql adapter script** (not a package import), preserving hub-and-spoke.
- `kv-cache-theater` — only if trace-driven; separate.
- `keeping-the-gpu-busy` — writeup/hub, later.

## Cost

Anchored to $130 for GRPO+RLVR (~$1.5/hr spot). v1 retrains nothing — runs on existing rlvr-sql checkpoints at inference prices. Develop on a cheap GPU, benchmark on target, measure throughput on short runs, stay 3B/single-GPU. v1 ≈ **$30–55**. Full expansion incl. RL convergence re-runs ≈ up to ~$300 (one convergence run at the end, not per-experiment).

## References

- Build target: nano-vLLM (github.com/GeeeekExplorer/nano-vllm) — study, don't fork.
- Kernel-from-scratch: MinivLLM (github.com/Wenyueh/MinivLLM).
- FlexAttention path: flex-nano-vllm (github.com/changjonathanc/flex-nano-vllm).
- Concepts: Inside vLLM anatomy (blog.vllm.ai/2025/09/05/anatomy-of-vllm.html); KV cache from scratch — Raschka (magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms).
- Latency/quant reference: gpt-fast (github.com/pytorch-labs/gpt-fast).
- Model: Qwen2.5-Coder-3B (huggingface.co/Qwen/Qwen2.5-Coder-3B).
