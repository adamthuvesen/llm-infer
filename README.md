# llm-infer

`llm-infer` is a small paged LLM inference engine built from first principles. It owns the
model forward pass, paged KV cache, scheduler, sampler, serving loop, benchmark harness, and
trace output. The code is meant to be read: each fast path sits next to a reference path, and
the tests prove correctness before performance numbers enter the conversation.

The main backend is `Qwen/Qwen2.5-Coder-3B-Instruct` at a pinned HuggingFace revision. A
second backend loads exported `llm-pretrain` DenseBackbone bundles through the same runtime
interface. The engine is not a wrapper around vLLM or Transformers generation; HuggingFace is
used as the oracle for Qwen weights and tokenization, while the runtime path is this repo's.

## What It Does

- Paged KV cache with block tables, lazy allocation, refcounts, copy-on-write, and real
  block lifecycle tracing.
- Continuous batching: running requests advance together through `decode_many()` while the
  scheduler admits new work at step boundaries.
- Chunked prefill, so long prompts can prefill in bounded chunks without fully blocking active
  decode work.
- Prefix caching for known sibling prompts, sharing full prompt blocks by refcount.
- Prompt-lookup speculative decoding with a greedy verifier. It is off by default and makes no
  speed claim.
- Request preemption under KV pressure: evict by freeing KV, keep generated tokens, then resume
  by recompute.
- OpenAI-compatible HTTP serving for Chat Completions, Completions, and a stateless Responses
  subset, with streaming, a small local chat UI, and Prometheus-style metrics.
- A local trace visualizer that replays schema-versioned engine events, including block
  allocation and free events emitted at the allocator boundary.

## Correctness Contract

The core rule is simple: a backend that fails the oracle does not get a throughput number.

For Qwen, the oracle is greedy HuggingFace generation at the pinned revision. The single-request
unit path must produce exact token ids. The batched and paged paths are then checked against the
same behavior. bf16 can hit genuine numerical ties; those are acceptable only when traced and
documented, not treated as "close enough."

DenseBackbone bundles are a correctness bridge. They load `llm_pretrain_dense_v1` exports,
validate the bundle, and run through the same serving interface, but the current dense path is
full recompute rather than an optimized paged-KV backend.

## Performance Snapshot

Benchmarks are pinned by workload, GPU, system versions, prompt set, decode settings, warmup,
and measurement window. vLLM is reported as the ceiling, not as a strawman to beat.

| Workload                                | System                          | Result                            |
| --------------------------------------- | ------------------------------- | --------------------------------- |
| 32 synthetic greedy requests, A100-80GB | naive HF sequential             | 41.5 tok/s                        |
| 32 synthetic greedy requests, A100-80GB | `llm_infer`                     | 98.3 tok/s                        |
| 32 synthetic greedy requests, A100-80GB | vLLM                            | 4323.6 tok/s                      |
| Frozen llm-rlvr-sql rollout, A100-80GB  | naive HF sequential             | 39.4 tok/s, $1.85 / 1k rollouts   |
| Frozen llm-rlvr-sql rollout, A100-80GB  | `llm_infer` with prefix caching | 411.5 tok/s, $0.16 / 1k rollouts  |
| Frozen llm-rlvr-sql rollout, A100-80GB  | vLLM                            | 2170.6 tok/s, $0.03 / 1k rollouts |

The win over naive HF comes from continuous batching and paged KV reuse. The gap to vLLM is
expected: vLLM has a mature scheduler, CUDA graphs, and custom kernels. See
[`docs/benchmark.md`](docs/benchmark.md) and
[`docs/keeping-the-gpu-busy.md`](docs/keeping-the-gpu-busy.md) for the full record, including
the optimization attempts that were measured and rejected.

## Repository Map

Start with [`docs/architecture.md`](docs/architecture.md) for diagrams and the full runtime map.

- `llm_infer/model/` - backend interface, runtime registry, Qwen forward pass, DenseBackbone
  bundle loader, shared transformer primitives.
- `llm_infer/kernels/` - the `AttentionBackend` protocol, `torch_naive` reference attention,
  and the optional GPU `flash_attn_paged` backend.
- `llm_infer/kv_cache/` - block allocator, block tables, paged K/V tensor store.
- `llm_infer/scheduler/` - waiting queue, running set, block-budget admission, optional
  preemption support.
- `llm_infer/serving/` - request lifecycle, continuous-batching step loop, sampler,
  speculative decode, trace hooks, and HTTP server.
- `llm_infer/benchmarks/` - workload definitions, runners, and report rendering.
- `tests/correctness/` - HF goldens, paged decode tests, batching checks, prefix cache,
  preemption, speculative decode, and dense bundle correctness.
- `scripts/` - Modal GPU oracles, benchmark harnesses, rollout timing, load generation, and
  fixture generation.
- `visualizer/` - static local KV trace replay UI.

## Quickstart

```bash
uv sync --extra dev
uv run ruff check
uv run pytest tests/correctness -q
uv run pytest tests/correctness -q -m slow  # optional 3B CPU oracle
```

The default local gate needs no GPU and keeps the slow 3B CPU oracles deselected; those
oracles load the pinned Qwen model and are available with `-m slow` when you need the full
HF-golden check.
The flash-attn backend is the one GPU-only path; its oracle runs on the target GPU via
`scripts/modal_oracle.py`. The benchmark and rollout harnesses (`scripts/modal_benchmark.py`,
`scripts/modal_rollout.py`) run on Modal A100-80GB. Regenerating goldens
(`scripts/generate_goldens.py`) loads the 3B model in fp32 on CPU.

Development conventions live in [`AGENTS.md`](AGENTS.md).

## Serving

The engine runs behind an OpenAI-compatible HTTP server (optional `serving` extra). Concurrent
clients stream from one background `step()` loop, so the same continuous-batching path is used
from tests, benchmarks, and HTTP. It speaks **Chat Completions** (`/v1/chat/completions`, the
primary surface) and **Completions** (`/v1/completions`), plus a stateless **Responses** subset
(`/v1/responses`, text generation only). Streaming and non-streaming are both supported;
unsupported features (tools, logprobs, `n>1`, server-side state) return a clear 4xx.

It also serves a small, self-contained **chat UI** at `/`: same origin as the API, no build
step, no framework. Tokens stream live with measured `tok/s` and time-to-first-token.

### Start the server

```bash
uv sync --extra serving
python -m llm_infer.serve --open          # qwen on 127.0.0.1:8000, opens the chat UI
python -m llm_infer.serve --backend dense --bundle exports/pretrain-214m-b200 --open
# Set $LLM_INFER_BUNDLE once and drop --bundle:
export LLM_INFER_BUNDLE=exports/pretrain-214m-b200
python -m llm_infer.serve --backend dense --open
# --backend / --bundle / --host / --port / --device / --dtype / --block-size / --num-blocks / --open
```

Then open <http://127.0.0.1:8000/> (or use `--open`) and chat with the loaded model.

### Chat completions - `curl`

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

### Responses API - `curl`

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

### Use the official OpenAI SDK

The stock `openai` client works with `base_url` pointed at the local server. The API key is
unused because this server does no auth.

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

### Metrics - `GET /metrics`

Prometheus text exposition (no `prometheus_client` dependency), read from real server/engine
state: counters for requests, completions by finish reason, generated tokens, and preemptions;
live gauges for running/waiting requests and the KV-cache block pool
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
(async `httpx`, streaming by default) and reports TTFT (p50/p99), end-to-end latency
(p50/p99), and an aggregate output rate. In **streaming** mode the rate is **deltas/s** (SSE
content chunks), not tokens/s; see `scripts/loadgen.py` for why. Use `--no-stream` for
blocking calls where usage block token counts are available.

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
traces. The committed fixture is a **labelled synthetic sample** produced by a standalone
generator that emits the same event shapes with a deterministic clock (no engine, model, or GPU
needed), so the visualizer ships on its own. The KV wall is driven by real `block_allocated` /
`block_freed` events emitted from the allocator boundary: filled blocks are physically held,
prefix-shared blocks are counted once, and a block frees only when its last owner releases it.

## Model backends

- `qwen`: `Qwen/Qwen2.5-Coder-3B-Instruct` at HF revision
  `488639f1ff808d1d3d0ba301aef8c11461451ec5` (the Instruct variant; see
  `llm_infer/model/config.py`). Plain `-3B` is a different model and would be wrong.
- `dense`: `llm_pretrain_dense_v1` export bundles from `llm-pretrain`, loaded through
  `llm_infer/model/runtime.py`. The v1 DenseBackbone path is correctness-first and full
  recompute under the shared serving engine; it is not yet the optimized paged-KV path.

## Further Reading

- [`docs/architecture.md`](docs/architecture.md) - module ownership and runtime flows.
- [`docs/benchmark.md`](docs/benchmark.md) - benchmark setup and results.
- [`docs/keeping-the-gpu-busy.md`](docs/keeping-the-gpu-busy.md) - rollout timing and profiling notes.
- [`docs/fixture-format.md`](docs/fixture-format.md) - golden fixtures and correctness rules.
