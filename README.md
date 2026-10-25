# llm-infer

![License](https://img.shields.io/github/license/adamthuvesen/llm-infer) ![Python](https://img.shields.io/badge/python-3.13-blue)

`llm-infer` is a small Python inference engine for `Esme-214M-Chat`, a model trained from scratch
in [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain). It owns the whole serving
path: paged KV cache, continuous batching, batched prefill, CUDA graphs, FlashInfer attention,
and an OpenAI-compatible server. Every speed claim first matches a known-good reference output.

On one A100 it serves **3.1x the tokens per second of batched HuggingFace `generate()`**. vLLM on
the same host is 1.2x to 1.6x faster, and the batch-1 gap is the main open question. The
[two-page write-up](docs/writeup.md) explains what each technique bought and what did not work.

## Quickstart

Python 3.13 on Linux or Apple-silicon macOS, with [`uv`](https://docs.astral.sh/uv/). You need
an Esme export bundle, for example from
[`esme-posttrain`](https://github.com/adamthuvesen/esme-posttrain).

```bash
uv sync --extra serving
```

```bash
uv run python -m llm_infer.serve --backend esme --bundle /path/to/esme-214m-chat --open
```

`--open` starts the local chat UI in a browser. The same server answers the OpenAI API at
`http://127.0.0.1:8000/v1`, so the official OpenAI Python client works with `base_url` set there.
`GET /v1/models` returns the model id to pass in requests. The server has no auth, so the API key
is ignored.

## What It Does

- **Paged KV cache** with block tables, lazy allocation, refcounts, copy-on-write, and lifecycle
  tracing.
- **Continuous batching** with ragged batched prefill and optional chunked prefill.
- **Prefix caching** for prompts that share leading blocks, including follow-up chat turns.
- **CUDA graphs** for decode: piecewise graphs per batch bucket, and grouped graphs that capture
  every layer, including KV writes and FlashInfer attention, as one graph per exact batch.
- **Fused elementwise kernels**: RMSNorm and RoPE compiled with `torch.compile`, which cut 37–44%
  of each graphed decode step ([findings entries 23 and 24](docs/findings.md)).
- **Prompt-lookup speculative decoding** with a greedy verifier. Local CPU serving turns it on.
  CUDA keeps it off, because it measured 3.5x to 4.2x slower per request there
  ([findings entry 22](docs/findings.md)).
- **Request preemption** under KV pressure: free the KV, keep the generated tokens, and resume by
  recompute.
- **OpenAI-compatible HTTP serving** for Chat Completions, Completions, and a small stateless
  Responses subset, with Prometheus-style metrics and a KV trace visualizer.

## Results

A system reports tok/s only after its tokens match the fp32 `PretrainBundleModel.logits()`
reference, with traced numerical ties the only allowed difference. Every number below comes from
a committed record and passed that gate.

**Against batched HuggingFace.** `2026-10-24`, one A100-SXM4-80GB container, chat prompts, up to
256 new tokens, real EOS, greedy, median of 5 runs:

| Concurrent requests | HF sequential tok/s | HF batched tok/s | `llm_infer` tok/s | vs HF batched |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 59.9 | 249.4 | 777.7 | **3.1x** |
| 64 | — | 2,079.3 | 5,867.8 | **2.8x** |
| 256 | — | 6,070.9 | 18,784.9 | **3.1x** |

HF sequential is the naive floor: one `generate()` call per request. Plain batching already gives
a 4.2x gain over it, so the engine is measured against batched `generate()`. The figure shows where
the remaining gain comes from. The ladder was measured before fused elementwise kernels, which
the vLLM table below includes.

![Esme ablation ladder: tok/s added by each engine technique](assets/fig-esme-ablation-ladder.svg)

**Against vLLM.** `2026-10-25`, vLLM 0.24.0 and `llm_infer` (fused elementwise kernels, with
grouped and piecewise CUDA graphs captured at each measured batch size) on one reserved A100 80GB
PCIe host, 256 cached prompt tokens, 128 new tokens per request, EOS ignored, median of 10 runs:

| Concurrent requests | `llm_infer` tok/s | vLLM tok/s | vLLM / `llm_infer` |
| ---: | ---: | ---: | ---: |
| 1 | 316.4 | 506.3 | 1.60x |
| 8 | 1,969.0 | 3,068.4 | 1.56x |
| 64 | 13,080.5 | 16,457.3 | 1.26x |
| 256 | 26,187.1 | 31,366.1 | 1.20x |

vLLM's lead shrinks with batch size, from 1.6x at batch 1 to 1.2x at batch 256
([findings entry 24](docs/findings.md)). In the same run, the unfused server path trailed vLLM by
2.6x at batch 1, so fused kernels closed most of that gap. The HTTP server captures graphs up to 16 concurrent
requests by default. Pass `--decode-graph-buckets` to serve larger batches from graphs.

## How the Claims Are Checked

- **Reference before speed.** The fp32 full-recompute oracle defines correct output. bf16 may pick
  a different token only at a near-tie, where the fp32 top-two margin is under 0.1 logits
  ([scoping.md](docs/scoping.md), [findings entry 20](docs/findings.md)).
- **Same-container A/Bs.** Identical code moves about ±20% between containers, so both arms of an
  A/B run in one container in alternating order, and ratios never cross containers.
- **Predictions before the run.** Recent experiments commit their predictions and decision rule
  first. `scripts/judge_paired_ab.py` applies the rule to records from two containers.
- **Reproducible without a GPU.** `make reproduce` checks the committed records against these
  tables, regenerates the figures, and requires each registered A/B to give its documented
  verdict.

The [research log](docs/findings.md) has every experiment's question, method, result, and
decision, including the negative results.

## Serving Details

The server picks its defaults from the device:

| Device | Weights | Attention | Decode | Prefix cache and speculation |
| --- | --- | --- | --- | --- |
| CPU (default) | fp16 | `torch_sdpa` | eager, with prompt-lookup speculation | on |
| CUDA with `--dtype bfloat16` | bf16 | FlashInfer | 8-step decode windows on piecewise and grouped CUDA graphs, fused RMSNorm and RoPE | off |

- On CPU, fp16 is about 12% faster than fp32, with byte-identical output on the 8-turn chat
  benchmark. `--dtype float32` gives the exact fp32 reference config.
- On CUDA, the fp32 default is the reference config. `--dtype bfloat16` (or `float16`) selects
  FlashInfer, which needs the `gpu` extra (`uv sync --extra gpu`). Pass
  `--attention-backend torch_naive` to opt out.
- On CUDA bf16 or fp16, the fused kernels add about 40 s of compile time to a cold start.
  `--no-fused-elementwise` skips it.
- Every automatic choice has an explicit flag or a `--no-*` override.

Engine flags: `--block-size`, `--num-blocks`, `--decode-window-size` (decode steps per EOS host
sync, where 1 syncs every step), `--prefill-chunk-size`, `--preemption-policy`,
`--prompt-lookup-speculative`, and `--prefix-cache`. See
`uv run python -m llm_infer.serve --help`.

Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/responses`,
`GET /v1/models`, `GET /metrics`, and `/` for the chat UI. The UI has a throughput bench mode
that streams 8 or 16 concurrent prompts and compares concurrent against sequential wall time. Its
base-URL field can point at a remote deployment.

To serve from a GPU on your own Modal account, run `modal deploy scripts/modal_esme_serve.py`.
It scales to zero and enables CORS, so a locally served bench page can point at it. The Modal
scripts find the bundle through `--bundle-path`, then `ESME_BUNDLE_PATH`, then the sibling
checkout `../esme-posttrain/exports/esme-214m-chat`.

## Backends

- `esme`: Esme export bundles. It serves through the paged KV engine path and is checked against
  the full-recompute `PretrainBundleModel.logits()` reference. `dense` is an alias for the same
  loader.
- `qwen`: an independent HuggingFace reference backend for `Qwen/Qwen2.5-Coder-3B-Instruct` at
  revision `488639f1ff808d1d3d0ba301aef8c11461451ec5`, used to catch correctness regressions.

## Development

```bash
uv sync --extra dev --extra serving
make check
```

`make check` runs lint, format check, types, and the CPU test suite. The slow 3B Qwen reference
checks are opt-in: `uv run pytest tests/correctness -q -m slow`. `make reproduce` re-derives the
committed benchmark verdicts without a GPU. GPU benchmarks run on Modal A100s through the
`scripts/modal_esme_*` harnesses. Findings entries 12 onward name the command that produced
their record.

## Documentation

- [docs/writeup.md](docs/writeup.md): two-page account of the experiments and what they decided.
- [docs/findings.md](docs/findings.md): research log, one entry per experiment.
- [docs/architecture.md](docs/architecture.md): runtime flows and module ownership.
- [docs/benchmark.md](docs/benchmark.md): benchmark method and every row's reference status.
- [docs/scoping.md](docs/scoping.md): scope, non-goals, and the reference policy.
- [docs/serving-eval.md](docs/serving-eval.md): the HTTP serving evaluation harness.
- [docs/fixture-format.md](docs/fixture-format.md): the golden fixture format.
- [visualizer/README.md](visualizer/README.md): the KV trace replay UI, which replays traces from
  `InferenceEngine(trace=...)`.

## Related Repositories

`llm-infer` is one part of the Esme chain. The
[full-lifecycle retrospective](https://github.com/adamthuvesen/esme-pretrain/blob/main/docs/retrospective.md)
walks the whole chain, with every claim linked to a committed artifact.

- [`esme-pretrain`](https://github.com/adamthuvesen/esme-pretrain): trains `Esme-214M-Base` from
  scratch.
- [`esme-posttrain`](https://github.com/adamthuvesen/esme-posttrain): adapts the base checkpoint
  with SFT, DPO, and verifier-backed RLVR.
- [`llm-rlvr`](https://github.com/adamthuvesen/llm-rlvr): a reusable RLVR harness with text-to-SQL
  as the reference task.
- [`grpo-decomp`](https://github.com/adamthuvesen/grpo-decomp): measures where GRPO gains come
  from, separating reliability from new capability.

## References

- Kwon et al., [_Efficient Memory Management for Large Language Model Serving with PagedAttention_](https://arxiv.org/abs/2309.06180), 2023.
- Yu et al., [_Orca: A Distributed Serving System for Transformer-Based Generative Models_](https://www.usenix.org/conference/osdi22/presentation/yu), 2022.
- Agrawal et al., [_Efficient LLM Inference via Chunked Prefills_](https://dl.acm.org/doi/10.1145/3759441.3759444), 2025.
- Leviathan et al., [_Fast Inference from Transformers via Speculative Decoding_](https://arxiv.org/abs/2211.17192), 2023.
- Dao, [_FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning_](https://arxiv.org/abs/2307.08691), 2023.
- OpenTelemetry, [_Traces_](https://opentelemetry.io/docs/concepts/signals/traces/).
