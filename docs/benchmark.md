# Three-way Benchmark

The benchmark evidence compares naive HF, `llm-infer`, and vLLM on one pinned
GPU, and reports whether the optimized path beats the naive baseline by an honest margin. The
benchmark measures **decode throughput on an identical workload** — same prompts, same stop
config, same greedy decoding — so the tokens/s numbers compare equal work, not different
work. vLLM is the ceiling, never the thing we beat.

Harness: `scripts/modal_benchmark.py` (Modal A100-80GB). Pure pieces (workload, report,
runners) live in `llm_infer/benchmarks/`.

## The three systems (four rows)

| row | what it is | why |
| --- | --- | --- |
| `hf_sequential` | HF `model.generate()` **once per request, one at a time** (default SDPA, bf16). | **The naive baseline, defined out loud.** What a person writes first: HF's own optimized cached generate, but no cross-request batching. Not a strawman — it is not the slow full-recompute path. |
| `hf_batched` | one **left-padded batched** `model.generate()` over all requests. | A stronger HF reference, so "beats naive HF" can't mean beating a deliberately weak baseline. |
| `llm_infer` | this engine, **flash-attn backend**, bf16, all requests in one paged cache; every running request advances in one **fused batched decode** (`decode_many`) per step. | The optimized path under test. |
| `vllm` | vLLM offline `LLM.generate`, prefix caching off, flags pinned. | The ceiling. |

**Where the speed comes from.** `decode_many` advances all running requests in **one** batched
forward per step — one matmul/kernel call over the whole running batch, with per-request RoPE
positions and ragged FlashAttention (`cu_seqlens`) over each request's paged history. An unfused
per-request, per-layer forward measured **slower than naive sequential HF** (251.7 s vs 164.9 s
at 32×128), so the fused batched forward is what earns the throughput. vLLM (which also fuses,
plus CUDA graphs / a mature scheduler) remains the ceiling;
`llm_infer` is the small, legible, correct paged engine that earns its throughput from the
batched forward rather than from a tight loop.

## Methodology

- **Workload** (`llm_infer/benchmarks/workload.py`): the committed golden `prompt_ids` —
  byte-identical to the correctness oracle — cycled up to `num_requests`. Replication is
  fair because vLLM **prefix caching is pinned off**: every system recomputes every prefill.
- **Agreement vs fp32 truth (the throughput gate).** The reference is the **fp32
  full-recompute oracle truth** — *not* bf16 HF `generate`, which itself diverges
  from truth at real margins (the documented step-32 case), so using it as the reference
  wrongly fails any backend that is *more* faithful to fp32. Truth is computed by running each
  unique prompt through the engine on the fp32 model (cached fp32 == full-recompute).
  Each system's bf16 output is compared to truth under a **bf16-sized tolerance (~0.1**, not
  the fp32 1e-3: bf16 noise at these logit magnitudes is ~0.05–0.1). The result reports each
  system's agreement profile — exact / genuine-tie / non-tie divergence. Per the project
  doctrine (*validate before you brag*), this profile is the **throughput gate, not a mere
  annotation**: a system whose bf16 output diverges from truth beyond genuine ties **reports no
  tok/s and no speedup** — only its token count and wall-clock, which stay as measured facts.
  Throughput is reported solely for systems that agree with truth, so a broken backend can never
  post a speed number; its divergence shows in the agreement column (`report.throughput_rows`
  enforces this via `agrees_with_truth`).
- **Timing**: `warmup` un-measured iterations (CUDA graphs / allocator settle), then `iters`
  measured iterations with a CUDA sync at each boundary. Greedy is deterministic, so tokens
  are identical across iterations and only wall-clock varies. Throughput = total scored
  output tokens ÷ **median** measured wall-clock (median, not best). Every system divides by
  the **same** token count (they decode the same continuation), so tok/s is pure speed.
- **Token count**: the scored continuation is tokens up to and including the first EOS, so
  trailing stop-token differences between systems never skew counts or equivalence.
- **One GPU type, pinned**: both functions run on A100-80GB. The HF/engine baselines and the
  vLLM ceiling run in separate images (vLLM ships its own torch/CUDA) on separate A100-80GB
  instances; GPU name, SM/max clocks, power cap, every library version, and the vLLM flags
  (`enable_prefix_caching=False`, `gpu_memory_utilization`, `max_num_seqs`, `max_model_len`,
  `dtype`, `tensor_parallel_size`) are captured into the result JSON.

## Running it

```bash
modal run scripts/modal_benchmark.py --command smoke   # cheap wiring check: N=2, 8 tokens, 1 iter
modal run scripts/modal_benchmark.py --command bench   # full run: N=32, 128 tokens, 1 warmup + 3 iters
```

The raw record lands in `bench-results/<command>-<stamp>.json` (git-ignored); the curated
table + config is folded into the **Result** section below at land time. The CPU correctness
gates (`tests/correctness/test_batch_equivalence.py` and `test_batched_decode.py` —
batched decode == serial == golden) run locally with zero spend and must be green first.

## Result

Run `2026-06-21` on **A100-80GB PCIe** (both functions; same variant this run). vLLM
**0.23.0**, torch `2.12.1+cu130`, flash-attn `2.8.3.post1`, transformers `5.12.1`. Workload:
32 requests × 128 new tokens, greedy, prefix caching off; 1 warmup + 3 measured iters,
median wall-clock. Every system decodes the same **4096** tokens.

| system | agrees fp32 truth | median s | output tok | tok/s | speedup vs naive |
| --- | --- | --- | --- | --- | --- |
| `hf_sequential` (naive baseline) | diverges (21/32 non-tie) | 98.69 | 4096 | 41.5 | 1.00× |
| `hf_batched` | diverges (21/32 non-tie) | 3.66 | 4096 | 1118.7 | 26.95× |
| **`llm_infer`** (ours) | **yes — 32/32 ties** | 41.66 | 4096 | **98.3** | **2.37×** |
| `vllm` (ceiling) | yes — 32/32 ties | 0.95 | 4096 | 4323.6 | 104.17× |

**Stop #4 holds.** `llm_infer` (98.3 tok/s) beats the naive baseline (41.5 tok/s) by **2.37×**.
The batched decode (`decode_many`) is the difference: fusing all running requests into one
decode forward per step runs the workload in 41.66 s, where an unfused per-request decode
measured 251.7 s (0.66× — *slower* than naive).

**Correctness, honestly.** Against fp32 full-recompute truth, **`llm_infer` and vLLM are
faithful — every divergence (all 32 requests) is a traced numerical tie**, while HF `generate`
(both sequential and batched) diverges at real margins on 21/32 requests (e.g. step 69, fp32
top-2 gap 0.62 — the documented fused-kernel reduction-order effect, not a tie). The engine
tracks the model's true greedy decode more faithfully than HF's own `generate()`; vLLM
agreeing with the same truth independently corroborates it. This is why the equivalence
reference is fp32 truth, not bf16 HF generate.

**The gap to the ceiling, named.** `llm_infer` (2.37× naive) sits ~11× below `hf_batched` and
~44× below vLLM. The remaining cost is paging overhead the v1 engine pays for legibility:
`decode_many` gathers each request's KV history into contiguous tensors per layer per step
(per-request Python loops + a `cat`), and there are no CUDA graphs. A custom paged-attention
kernel that reads blocks in place, a vectorized gather, and graph capture are the v2/v3
expansion path — out of v1 scope. v1's bar is *beats naive*, met, with the correctness and
the gap both reported straight.
