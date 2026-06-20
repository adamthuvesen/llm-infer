> Seeded from the HQ scoping doc (`llm-infer-plan.md`) at Phase A start. The engine is
> named **`llm-infer`** in this repo; the source doc uses the working name `nano-infer`/
> `paged` interchangeably for the same engine. Scope is authoritative; the name is not.

# nano-infer — scoping doc

A minimal, honest LLM inference engine for **Qwen2.5-Coder-3B**, measured as an **rlvr-sql rollout backend**. Engine first; beautiful explanation second; expansion only if it serves the claim.

## The claim (capability statement, not a number)

> A small paged-inference engine for Qwen2.5-Coder-3B — exact greedy decoding vs HF, paged KV-cache + continuous batching running end-to-end, benchmarked against naive HF and vLLM on one GPU, with an early rlvr-sql rollout timing hook.

Not "beats vLLM." Not "production serving." Honest systems evidence. The differentiator is the RL-rollout measurement — that is what makes it *yours* rather than a nano-vLLM clone.

## Hard stop (evidence-based, ship v1 when ALL hold)

1. Qwen2.5-Coder-3B **greedy decoding is exact** (same token ids as HF) on the unit correctness suite (fp32 fallback if bf16 tie-breaks get noisy — see Correctness).
2. Paged KV-cache + continuous batching run **end-to-end**.
3. Benchmarks compare **naive HF vs nano-infer vs vLLM** on one documented GPU, config pinned.
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
| **D — evidence** | batch-correctness suite · three-way benchmark table | naive HF vs nano-infer vs vLLM, config pinned |
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
nano_infer/
  model/          # Qwen loading, weights, tokenizer, config
  kv_cache/       # block allocator, block tables, page metadata
  scheduler/      # prefill/decode admission, continuous batching
  kernels/        # AttentionBackend protocol + implementations
  serving/        # request queue, sampler, streaming loop
  benchmarks/     # naive vs nano-infer vs vLLM
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

## KV Cache Theater (deferred — trace-driven or skip)

Build only if it replays **real engine events**, not a simulation. The engine emits a trace; the visualizer is a debugger + proof artifact, not decoration. Separate repo / app, fed by nano-infer traces. Event schema:

`request_admitted · prefill_started · block_allocated · decode_step · request_finished · block_freed · batch_size_changed · tokens_per_second_sampled`

## Expansion path (post-v1, only if it serves the claim)

- v2: FlexAttention backend (more ownership, no raw CUDA/Triton).
- v3: **Triton paged-attention kernel** — the real systems-depth milestone; separate repo. Acceptance: matches reference within tolerance, faster than torch reference, documented limits.
- v4: scheduling depth — prefix caching, chunked prefill, priority/fairness, cancellation, token budgeting.
- v5: serving depth — OpenAI-compatible endpoint, streaming, metrics, load generator.
- v6: optimized rlvr-sql rollout backend + **wall-clock decomposition writeup** (the hub that ties nano-infer + quantization + kernel repos together).

## Repos (hub-and-spoke, not a monorepo)

- Engine name candidates: **`paged`** (preferred — names the core idea, distinctive, memorable) > `llm-infer` (descriptive but generic/collision-heavy) ≈ `nano-infer` (derivative echo of nano-vLLM). Pick `paged` if the GitHub name is free. v1.5 is a **standalone engine + thin rlvr-sql adapter script** (not a package import), preserving hub-and-spoke.
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
