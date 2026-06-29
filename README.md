# llm-infer

`llm-infer` is a small paged LLM inference engine with pluggable model backends.

It owns the runtime path: model forward pass, paged KV cache, scheduler, sampler,
serving loop, benchmark harness, and trace output. HuggingFace is used for Qwen
weights and tokenization, not for generation.

The project is deliberately correctness-first. A backend that fails the reference
check does not get a throughput number.

## What Is Here

- Paged KV cache with block tables, lazy allocation, refcounts, copy-on-write, and
  allocator-level block lifecycle tracing.
- Continuous batching, where running requests advance together through one
  batched decode step.
- Chunked prefill, so long prompts do not fully block active decode work.
- Prefix caching for sibling prompts with shared prompt blocks.
- Prompt-lookup speculative decoding with a greedy verifier. It is off by
  default and does not make a speed claim.
- Request preemption under KV pressure: free KV, keep generated tokens, then
  resume by recompute.
- OpenAI-compatible HTTP serving for Chat Completions, Completions, and a small
  stateless Responses subset.
- Prometheus-style metrics and a local KV trace visualizer.

For the longer architecture map, see [docs/architecture.md](docs/architecture.md).

## Backends

- `qwen`: `Qwen/Qwen2.5-Coder-3B-Instruct` at revision
  `488639f1ff808d1d3d0ba301aef8c11461451ec5`. Use the Instruct model and its
  chat template. Plain `Qwen2.5-Coder-3B` is a different model.
- `dense`: `llm_pretrain_dense_v1` export bundles from `esme-pretrain`,
  including ESME checkpoints trained there and post-trained in `esme-posttrain`.
  This is a correctness bridge, not a fast paged-KV backend yet; it uses full
  recompute and rejects prefix caching, speculative decoding, and preemption.

## Correctness Contract

For Qwen, the reference is greedy HuggingFace generation at the pinned revision.
The single-request unit path must produce exact token IDs. Batched and paged
paths are then checked against that behavior.

`bf16` can hit real numerical ties. Those are acceptable only when traced and
documented. "Close enough" fails the reference check.

For DenseBackbone bundles, the reference is the source bundle's logits and
generation contract. Fixture details live in
[docs/fixture-format.md](docs/fixture-format.md); the evidence bar lives in
[docs/scoping.md](docs/scoping.md).

## Install

Python 3.11+ is required. Dependencies are managed with `uv`.

```bash
uv sync --extra dev
uv run ruff check
uv run pytest tests/correctness -q
uv run pytest -q
```

The default test gate is CPU-runnable. Slow 3B CPU reference tests are opt-in:

```bash
uv run pytest tests/correctness -q -m slow
```

GPU-only checks, including the flash-attn reference check, run through the Modal scripts in
[scripts/](scripts/).

## Serve Locally

Install the serving extra:

```bash
uv sync --extra serving
```

Start Qwen on `127.0.0.1:8000`:

```bash
python -m llm_infer.serve --open
```

Start a DenseBackbone bundle:

```bash
python -m llm_infer.serve --backend dense --bundle exports/pretrain-214m-b200 --open
```

The server exposes:

- `POST /v1/chat/completions`
- `POST /v1/completions`
- `POST /v1/responses`
- `GET /metrics`
- `/` for the small local chat UI

The official OpenAI Python client works by setting `base_url` to
`http://127.0.0.1:8000/v1`. The API key is ignored because this server has no
auth. For load testing, use [scripts/loadgen.py](scripts/loadgen.py).

## Benchmarks

Benchmarks compare the same greedy workload across naive HF, `llm-infer`, and
vLLM on a pinned A100-80GB setup. vLLM is the ceiling, not the opponent this repo
claims to beat.

| Workload                      | System                          | Result       |
| ----------------------------- | ------------------------------- | ------------ |
| 32 synthetic greedy requests  | naive HF sequential             | 41.5 tok/s   |
| 32 synthetic greedy requests  | `llm_infer`                     | 98.3 tok/s   |
| 32 synthetic greedy requests  | vLLM                            | 4323.6 tok/s |
| Frozen `llm-rlvr` rollout | naive HF sequential             | 39.4 tok/s   |
| Frozen `llm-rlvr` rollout | `llm_infer` with prefix caching | 411.5 tok/s  |
| Frozen `llm-rlvr` rollout | vLLM                            | 2170.6 tok/s |

The win over naive HF comes from continuous batching and KV reuse. The gap to
vLLM is expected: vLLM has a mature scheduler, CUDA graphs, and custom kernels.

Full methodology and measured caveats are in [docs/benchmark.md](docs/benchmark.md)
and [docs/keeping-the-gpu-busy.md](docs/keeping-the-gpu-busy.md).

## KV Trace Visualizer

The static viewer in [visualizer/](visualizer/) replays schema-versioned JSONL
traces emitted by `InferenceEngine(trace=...)`. It can use the committed
synthetic fixture or a real engine trace.

See [visualizer/README.md](visualizer/README.md) for the two-command local
setup.

## Where To Look

- [llm_infer/](llm_infer/) - engine code.
- [tests/correctness/](tests/correctness/) - reference and equivalence tests.
- [scripts/](scripts/) - goldens, Modal runs, loadgen, and trace fixtures.
- [visualizer/](visualizer/) - static trace replay UI.

## Further Reading

- [docs/scoping.md](docs/scoping.md) - scope, non-goals, and evidence boundary.
- [docs/architecture.md](docs/architecture.md) - runtime flows and module ownership.
- [docs/benchmark.md](docs/benchmark.md) - benchmark setup and result record.
- [docs/keeping-the-gpu-busy.md](docs/keeping-the-gpu-busy.md) - rollout timing and profiling notes.
- [docs/fixture-format.md](docs/fixture-format.md) - golden fixtures and tie handling.
