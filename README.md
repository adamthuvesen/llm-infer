# llm-infer

A minimal, honest paged LLM inference engine for **Qwen2.5-Coder-3B-Instruct**,
sequenced so the correctness oracle comes before anything fast or batched, and
benchmarked as an **rlvr-sql GRPO rollout backend**.

The differentiator is not "fast" — it is that every backend is proven exact against
HuggingFace greedy decoding before it reports a single tok/s, and the throughput is
measured on rlvr-sql's *actual* rollout workload, not a synthetic microbench. vLLM is
the ceiling, never the thing we beat; the gap to it is named, not hidden.

## Status — v1 complete, v2 speed tranche paused

| Phase | What | State |
|-------|------|-------|
| **A** trusted oracle | `torch_naive` reference + single-request token-for-token vs HF | ✅ |
| **B** systems milestone | paged KV allocator + continuous scheduler, two-request vertical slice | ✅ |
| **C** speed | `flash_attn_paged` behind the `AttentionBackend` adapter, gated by the oracle | ✅ |
| **D** evidence | batch-correctness suite + three-way benchmark ([`docs/benchmark.md`](docs/benchmark.md)) | ✅ |
| **E** differentiator | one frozen rlvr-sql rollout-timing comparison ([`docs/keeping-the-gpu-busy.md`](docs/keeping-the-gpu-busy.md)) | ✅ |
| **v2** performance foundation | profiling, no-sync cleanup, packed KV reads/writes, read-plan reuse | paused at 365.8 tok/s |
| **v3** decode execution architecture | 32-slot decode graph / static bucket slice from [`docs/decode-graph-plan.md`](docs/decode-graph-plan.md) | planned |

Out of v1 scope (the firewall): no custom Triton/CUDA kernel, no quantization, no
prefix caching, no chunked prefill, no OpenAI-compatible server / streaming, no
multi-GPU. See [`docs/scoping.md`](docs/scoping.md) for the full plan and non-goals.

## Layout

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

**v2 speed campaign — current best** (same frozen rlvr-sql rollout, A100-80GB PCIe,
run 2026-06-21): `llm_infer` now reaches **365.8 tok/s** and **$0.18 / 1k rollouts**
after profiling/no-sync cleanup, packed KV reads, vectorized KV writes, and read-plan reuse.
That is about **5.55×** over the 65.9 tok/s v1.5 baseline, but still short of the
**>=650 tok/s** campaign target. The next planned tranche is design-first decode execution:
static decode buckets and CUDA graph capture, starting with the 32-slot frozen rollout shape.

Rejected v2 follow-ups are part of the roadmap, not footnotes: direct FlashAttention GQA,
projection/MLP fusion, step-local prompt-prefix KV copying, and no-gather paged KV attention
all either changed the sampled token path or regressed speed. Do not retry those shapes
without a new profile-backed reason.

## Roadmap

1. **v1 — Correct minimal engine.** Complete: HF oracle, paged KV, continuous batching,
   benchmark evidence, and frozen rlvr-sql rollout proof.
2. **v2 — Performance foundation.** Paused at 365.8 tok/s: the obvious KV materialization
   waste is mostly harvested, and every accepted speed claim has pinned rollout evidence.
3. **v3 — Decode execution architecture.** Next: prove whether a graph-safe 32-slot decode
   path can close the remaining gap toward 650 tok/s without changing the 3026-token frozen
   sampled path.
4. **v4 — Scheduler and serving depth.** Generalize beyond the frozen rollout with dynamic
   buckets, prefix-cache ownership/refcounts, cancellation, token budgeting, streaming, and
   an OpenAI-compatible surface.
5. **v5 — Backend/kernel depth.** Revisit FlexAttention, Triton, or a true no-gather paged
   backend only after decode orchestration is cleaner and the acceptance gates are stable.
6. **v6 — Release-quality story.** Release when the engine either hits the speed target or
   has a measured ceiling and a clear writeup explaining the remaining vLLM gap honestly.

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
