# Three-way benchmark (Phase D)

The evidence behind hard-stops **#3** (compare naive HF vs llm-infer vs vLLM on one pinned
GPU) and **#4** (the optimized path beats the naive baseline by an honest margin). The
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
| `llm_infer` | this engine, **flash-attn backend**, bf16, all requests in one paged cache under the continuous-batching loop. | The optimized path under test. |
| `vllm` | vLLM offline `LLM.generate`, prefix caching off, flags pinned. | The ceiling. |

**Honest limitation of the v1 engine.** The continuous-batching loop advances each running
request with its *own* forward inside a step — it does **not** yet fuse the batch into one
matmul. So `llm_infer`'s edge over `hf_sequential` comes from the fused kernel + paged cache
+ a tight Python loop, **not** from batched matmuls. vLLM (which fuses the batch) is
therefore expected to sit far above `llm_infer`; that gap is the honest cost of the v1
scope, reported, not hidden. If `llm_infer` does not clear `hf_sequential` on stop #4, the
fix is a batched-forward decode step — a scoped engine change, decided on the data.

## Methodology

- **Workload** (`llm_infer/benchmarks/workload.py`): the committed golden `prompt_ids` —
  byte-identical to the correctness oracle — cycled up to `num_requests`. Replication is
  fair because vLLM **prefix caching is pinned off**: every system recomputes every prefill.
- **Equivalence before throughput** ("validate before you brag"): `hf_sequential` is the
  reference; `hf_batched`, `llm_infer`, and `vllm` are each checked against it under the
  oracle's tie policy (`tests/correctness/tie_tolerance.py`) — exact tokens, or a first
  divergence the **fp32 reference** proves is a genuine numerical tie (top-2 gap ≤ 1e-3). A
  non-tie divergence marks that system non-equivalent and it reports **no tok/s**.
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
table + config is folded into the **Result** section below at land time. The CPU batch-
correctness gate (`tests/correctness/test_batch_equivalence.py`) runs locally with zero
spend and must be green first.

## Result

_Pending the first `bench` run — the table, the pinned config, and any traced ties land here._
