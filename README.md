# llm-infer

A from-scratch paged LLM inference engine with **pluggable model backends**, built to do the
core things a real inference engine does — paged KV cache, continuous batching, chunked
prefill, prefix caching, speculative decoding, honest benchmarking, and trace-driven
observability — and to stay legible while doing them. Qwen2.5-Coder is the pinned HF-oracle
backend; exported llm-pretrain DenseBackbone bundles are a sibling backend.

The methodology is the differentiator, not a tok/s number. Every backend is proven **exact
against HuggingFace greedy decoding before it reports a single tok/s**; every speed claim is
measured and config-pinned; vLLM is the named ceiling, never the thing we beat. Throughput is
benchmarked on real workloads — including one frozen llm-rlvr-sql GRPO rollout — rather than a
synthetic microbench. (Serving llm-rlvr-sql rollouts is a fun applied benchmark, not the point.)

## Status — v1/v1.5 complete, v2 speed pass complete (measured ceiling)

| Phase | What | State |
|-------|------|-------|
| **A** trusted oracle | `torch_naive` reference + single-request token-for-token vs HF | ✅ |
| **B** systems milestone | paged KV allocator + continuous scheduler, two-request vertical slice | ✅ |
| **C** speed | `flash_attn_paged` behind the `AttentionBackend` adapter, gated by the oracle | ✅ |
| **D** evidence | batch-correctness suite + three-way benchmark ([`docs/benchmark.md`](docs/benchmark.md)) | ✅ |
| **E** differentiator | one frozen llm-rlvr-sql rollout-timing comparison ([`docs/keeping-the-gpu-busy.md`](docs/keeping-the-gpu-busy.md)) | ✅ |
| **v2** speed pass | profiling, no-sync cleanup, packed KV reads/writes, read-plan reuse | ✅ 365.8 tok/s, ceiling named |
| **prefix caching** | refcounted KV-block sharing for G=4 rollout siblings | ✅ 411.5 tok/s, 3026-token path preserved |
| **chunked prefill** | bounded prompt chunks interleaved with active decode work | ✅ |
| **speculative decoding v1** | prompt-lookup n-gram draft + greedy verifier, no second model | ✅ technique/correctness evidence, no speed claim |
| **KV trace visualizer** | static local KV-cache-theater viewer fed by real schema-v3 engine traces, incl. block lifecycle | ✅ |
| **request preemption** | recompute eviction under KV pressure (LIFO victim), resumed token-for-token | ✅ opt-in, forced-preemption oracle |

The engine itself is the goal — a small, legible paged inference engine to learn from and show.
Speed and the llm-rlvr-sql hook are the fun side-quest. The forward plan is **engine-first**:
block-lifecycle trace hooks → request preemption/eviction (done) → serving depth (streaming + an
OpenAI-compatible endpoint) → sampling completeness → an architecture writeup, with quantization
as a later speed lever.
The decode-graph / static-bucket idea was built and rejected (a measured dead-end — see
[`docs/scoping.md`](docs/scoping.md)). See [`docs/scoping.md`](docs/scoping.md) for the
full plan and non-goals.

## Layout

Start with [`docs/architecture.md`](docs/architecture.md) for a guided map of modules, flows,
and diagrams.

- `kernels/` — the `AttentionBackend` protocol, a slow readable `torch_naive` reference
  (the truth every other backend is validated against), and `flash_attn_paged` (the fast
  GPU-only backend).
- `model/` — backend interface/registry, Qwen loading, DenseBackbone bundle loading, and
  backend-independent greedy decode.
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

**Phase E — the real llm-rlvr-sql rollout** (32-completion GRPO batch from llm-rlvr-sql's own prompt
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

**v2 speed pass — complete, with a measured ceiling** (same frozen llm-rlvr-sql rollout,
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

1. **v1 / v1.5 — correct minimal engine + applied benchmark.** HF oracle, paged KV, continuous
   batching, three-way benchmark, and the frozen llm-rlvr-sql rollout proof + writeup.
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
6. **KV-cache-theater trace visualizer.** `InferenceEngine(trace=...)` emits typed,
   schema-versioned events from the real request path, and
   [`visualizer/`](visualizer/) renders those traces as local request lanes, prefill chunks,
   decode emissions, batch/throughput signals, event inspection, playback, and a paged KV wall
   driven by real block allocation/free.
7. **Block-lifecycle trace hooks.** `block_allocated` / `block_freed` are emitted from an
   observer on `BlockAllocator` — the physical free-pool boundary — so they stay honest for
   refcounted prefix sharing and copy-on-write. The prerequisite for the preemption demo below.
8. **Request preemption / eviction.** Opt-in (`InferenceEngine(preemption=True)`): admission
   over-commits the pool on current footprint, and when a running request needs a block the pool
   cannot give, the engine evicts the most-recently-admitted request (LIFO) — freeing its KV,
   keeping its tokens — then resumes it by recomputing prompt-plus-generated. A forced-preemption
   oracle pins that the resumed request is token-for-token identical to the uninterrupted run; the
   visualizer renders the eviction + recompute resume, and the bundled demo shows one.
9. **Serving depth.** Streaming token output → an OpenAI-compatible HTTP server
   (`/v1/chat/completions`, `/v1/completions`, a stateless `/v1/responses` subset) → a hand-rolled
   Prometheus `/metrics` endpoint read from real engine state (request/token/preemption counters,
   live KV-pool and queue gauges, TTFT + latency histograms) → a concurrent `httpx` load generator
   (`scripts/loadgen.py`). The engine is drivable as a real server — `curl` it, point the
   OpenAI SDK at it, scrape its metrics, and load-test it — not only through the frozen harness.
   See [Serving](#serving).

**Primary forward track — finish the engine's technique set (dependency order)**

10. **Sampling completeness.** top-k, repetition/frequency penalties, and stop-strings, rounding
   out the decode surface beyond greedy / temperature / top-p.
11. **Expand *Keeping the GPU Busy*.** Narrate the architecture and the honest dead-ends (the v2
   ceiling, the decode-graph rejection) so the doc teaches the engine, not just one number.

**Secondary track — speed, later, for fun**

12. **Quantization** (gpt-fast style: projection/MLP matmuls + KV bandwidth) is the real remaining
   speed lever. **650 tok/s is a checkpoint quantization may clear, not a goal to grind toward.**
   Backend/kernel depth (FlexAttention, Triton, no-gather paged) stays parked here.

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

## Serving

The engine runs behind an OpenAI-compatible HTTP server (optional `serving` extra), so it is
`curl`-able like OpenAI and continuous batching shows under concurrent load — many clients
stream off one background `step()` loop. It speaks **Chat Completions** (`/v1/chat/completions`,
the primary surface) and **Completions** (`/v1/completions`), plus a stateless **Responses**
subset (`/v1/responses`, text generation only). Streaming and non-streaming both supported;
unsupported features (tools, logprobs, `n>1`, server-side state) return an honest 4xx.

### Start the server

```bash
uv sync --extra serving
python -m llm_infer.serve            # default: --backend qwen on 127.0.0.1:8000
python -m llm_infer.serve --backend dense --bundle exports/pretrain-214m-b200
# --backend / --bundle / --host / --port / --device / --dtype / --block-size / --num-blocks
```

### Chat completions — `curl`

Non-streaming returns one JSON body with `usage`:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "messages": [{"role": "user", "content": "Write a haiku about paged attention."}],
    "max_tokens": 64
  }'
```

Streaming emits OpenAI SSE chunks (`data:` deltas, terminated by `data: [DONE]`):

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "messages": [{"role": "user", "content": "Stream me a limerick about KV cache."}],
    "max_tokens": 64,
    "stream": true
  }'
```

### Responses API — `curl`

The stateless text-generation subset of `/v1/responses` (a bare string is continued raw;
add `instructions` to route through the chat template):

```bash
curl -s http://127.0.0.1:8000/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "input": "Explain a block allocator in one sentence.",
    "max_output_tokens": 64
  }'
```

### Point the official OpenAI SDK at it

It is wire-compatible, so the stock `openai` client works with `base_url` flipped (the API
key is unused — this server does no auth):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-Coder-3B-Instruct",
    messages=[{"role": "user", "content": "Write a haiku about paged attention."}],
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Metrics — `GET /metrics`

A hand-rolled Prometheus exposition (no `prometheus_client` dependency), read from real
server/engine state — counters for requests, completions by finish reason, generated tokens,
and preemptions; live gauges for running/waiting requests and the KV-cache block pool
(used / free / total / utilization); and TTFT + end-to-end latency histograms.

```bash
curl -s http://127.0.0.1:8000/metrics
```

```text
# HELP llm_infer_generated_tokens_total Output tokens generated across all requests.
# TYPE llm_infer_generated_tokens_total counter
llm_infer_generated_tokens_total 192
# HELP llm_infer_kv_blocks_used Physical KV-cache blocks currently allocated.
# TYPE llm_infer_kv_blocks_used gauge
llm_infer_kv_blocks_used 23
# HELP llm_infer_ttft_seconds Time from request arrival to its first generated token.
# TYPE llm_infer_ttft_seconds histogram
llm_infer_ttft_seconds_bucket{le="0.25"} 14
...
```

### Load generator

`scripts/loadgen.py` fires N concurrent requests at a running server's OpenAI endpoint
(async `httpx`, streaming by default) and reports per-request and aggregate tokens/s, TTFT
(p50/p99), end-to-end latency (p50/p99), total throughput, and an error count — the numbers
that make continuous batching visible under load.

```bash
uv run python scripts/loadgen.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --concurrency 16 --num-requests 64 --max-tokens 64 \
  --prompt "Write a short haiku about paged attention."
```

```text
llm-infer load generator
------------------------------------------------
requests (ok/total)       64/64
errors                    0
concurrency               16 (streaming)
wall-clock                ...
output tokens             ...
throughput                ... tok/s
per-request tok/s (mean)  ...
TTFT p50 / p99            ... ms
latency p50 / p99         ... s
```

Add `--no-stream` to measure single blocking calls (then TTFT equals total latency).

## KV Trace Visualizer

```bash
uv run python scripts/generate_kv_trace_fixture.py
python -m http.server 8765
```

Open `http://localhost:8765/visualizer/` to inspect the committed schema-v3 fixture at
[`docs/assets/kv_trace_schema_v3.jsonl`](docs/assets/kv_trace_schema_v3.jsonl), or load another
JSONL trace in the browser. The viewer replays real `InferenceEngine(trace=...)` schema-v3
traces; the committed fixture is a **labelled synthetic sample** produced by a standalone
generator that emits the same event shapes with a deterministic clock (no engine, model, or GPU
needed), so the visualizer ships on its own. The KV wall is driven by real `block_allocated` /
`block_freed` events emitted from the allocator boundary — filled blocks are physically held,
prefix-shared blocks are counted once, and a block frees only when its last owner releases it.

## Model backends

- `qwen`: `Qwen/Qwen2.5-Coder-3B-Instruct` at HF revision
  `488639f1ff808d1d3d0ba301aef8c11461451ec5` (the Instruct variant — see
  `llm_infer/model/config.py`). Plain `-3B` is a different model and would be wrong.
- `dense`: `llm_pretrain_dense_v1` export bundles from `llm-pretrain`, loaded through
  `llm_infer/model/runtime.py`. The v1 DenseBackbone path is correctness-first and full
  recompute under the shared serving engine; it is not yet the optimized paged-KV path.
