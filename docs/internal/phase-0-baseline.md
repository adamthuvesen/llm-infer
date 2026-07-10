# Phase 0 Performance Baseline

This is the decision record for Phase 0 of the
[performance roadmap](performance-roadmap.md). The harness and evidence runs are complete. They
found two correctness blockers: the engine's batch-8/context-768 row has a non-tie divergence,
and seeded sampled serving fails batch invariance at batches 8, 64, and 256.

A row reports speed only after the fp32 oracle accepts every request as exact or a traced bf16
tie. Raw benchmark JSON is gitignored; this file is the curated record.

## Current FlashInfer Decode Profile

Source: local raw record
`bench-results/esme-decode-capture-20260707T102505.json`, produced on 2026-07-07 from the
`capture` command. Raw benchmark output is gitignored; the table below is its committed summary.

Configuration:

- NVIDIA A100 80GB PCIe, 1,410 MHz SM clock, 300 W power limit.
- Esme-214M-Chat bf16 fast path with FlashInfer paged decode and piecewise CUDA graphs.
- Eight-token planned decode windows, 64 maximum new tokens, one warmup, three measured runs.
- fp32 `PretrainBundleModel.logits()` oracle with the Esme tie rule.

| Batch | GPU busy/pass | Profiler CPU/pass | Ordinary launches/pass | Graph launches/pass | Reference result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8 | 49.2 ms | 50.5 ms | 1,207.2 | 217 | 8 exact, 0 ties, 0 non-ties |
| 64 | 49.9 ms | 54.6 ms | 1,212.2 | 217 | 48 exact, 16 ties, 0 non-ties |
| 256 | 58.1 ms | 69.6 ms | 1,233.7 | 217 | 192 exact, 64 ties, 0 non-ties |

Launch counts remain nearly flat from batch 8 through batch 64 while useful work grows eightfold.
That is direct evidence of fixed launch overhead at low and medium batch sizes. At batch 8,
`cudaGraphLaunch` accounts for 17.4 ms of self CPU time per profiled pass, about 35% of the
profiler's CPU attribution. CPU and GPU work can overlap, so this ratio is not a wall-time speedup
estimate. It is large enough to justify measuring wider graph capture after page metadata becomes
stable.

The profile also shows about 151 ordinary launches and 27 graph launches per generated-token
step across the whole batch-8 row. These are not per-request token counts. Page-plan work is not
isolated in this record, but the current native-page path still
builds packed read indices that FlashInfer does not consume and rebuilds page metadata before each
token. That is code evidence of duplicated work even before a timing attribution is available.

## Measurement Contract

The Phase 0 harness records these workloads:

- Batch sizes 1, 8, 64, and 256.
- Cached contexts 32, 256, and 768 tokens.
- 128 steady-state decode tokens after two warmups.
- Ten measurements per matrix cell.
- Cold engine/KV-pool startup outside persistent-engine request timing.
- Prefill, decode, page planning, attention, sampling, and window-flush timing.
- Greedy and sampled HTTP TTFT, p50/p95 ITL, request latency, queue time, and output tok/s.
- vLLM in a separate process on the same reserved GPU host, gated by the same fp32 oracle.

## Engine Matrix

Source: `bench-results/esme-measurement-baseline-20260710T001422.json`, produced on 2026-07-10.
The run used an A100-SXM4-80GB at 1,410 MHz and a 400 W power limit. It generated exactly 128
tokens per request, ignored EOS, used two warmups and ten measured iterations, and reused one
persistent engine. Synthetic contexts repeat tokenized prompts to reach the exact requested
length; they are useful for scaling, not a natural-text workload.

| Context | Batch | Median | p95 | Output tok/s | Reference |
| ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 1 | 0.799 s | 0.805 s | 160.2 | 1 exact |
| 32 | 8 | 1.195 s | 1.362 s | 856.7 | 4 exact, 4 ties |
| 32 | 64 | 3.668 s | 3.714 s | 2,233.6 | 24 exact, 40 ties |
| 32 | 256 | 12.292 s | 12.865 s | 2,665.9 | 160 exact, 96 ties |
| 256 | 1 | 0.841 s | 0.856 s | 152.3 | 1 tie |
| 256 | 8 | 1.385 s | 1.530 s | 739.2 | 4 exact, 4 ties |
| 256 | 64 | 3.927 s | 4.148 s | 2,085.9 | 32 exact, 32 ties |
| 256 | 256 | 13.394 s | 16.851 s | 2,446.4 | 128 exact, 128 ties |
| 768 | 1 | 0.821 s | 0.823 s | 155.8 | 1 exact |
| 768 | 8 | 1.295 s | 1.552 s | — | 6 exact, 1 tie, **1 non-tie** |
| 768 | 64 | 4.500 s | 4.699 s | 1,820.3 | 40 exact, 24 ties |
| 768 | 256 | 15.607 s | 15.794 s | 2,099.6 | 192 exact, 64 ties |

The failing batch-8/context-768 request diverged at decode step 11. The fp32 oracle preferred
token 458 over token 46 by 0.108 logits, outside the 0.1 tie rule. The same context passes at
batches 1, 64, and 256, so this points to a batch-dependent numeric path. It needs a root-cause
check before that workload can support a speed claim.

Engine and KV-pool construction took 0.2–5.9 ms across the matrix. Graph capture took 58.0 s and
was recorded separately. Model loading was outside both measurements. These costs must not be
folded into steady-state request timing.

## Phase Attribution and Bottleneck Decision

The phase profile is a separate diagnostic pass. Timings are nested and are not additive. CUDA
event recording also perturbs the run: profile wall is 29–36% above the uninstrumented median at
batch 1, 3–20% at batch 8, 5–16% at batch 64, and 4–8% at batch 256. Ratios below are useful for
finding large regions, not exact shares of uninstrumented wall time or available speedups.

| Context | Batch | Profile wall | Serial prefill | Decode | Page planning | Attention | Flush |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 1.090 s | 45 ms | 1.025 s | 31 ms | 198 ms | 5 ms |
| 32 | 8 | 1.431 s | 352 ms | 1.040 s | 27 ms | 81 ms | 19 ms |
| 32 | 64 | 4.240 s | 2.833 s | 1.031 s | 26 ms | 80 ms | 302 ms |
| 32 | 256 | 13.294 s | 11.217 s | 1.116 s | 28 ms | 135 ms | 690 ms |
| 768 | 1 | 1.060 s | 50 ms | 0.992 s | 25 ms | 191 ms | 3 ms |
| 768 | 64 | 4.719 s | 3.252 s | 1.074 s | 21 ms | 228 ms | 100 ms |
| 768 | 256 | 16.306 s | 13.168 s | 1.539 s | 25 ms | 601 ms | 414 ms |

Measured facts:

- Prefill still runs once per request. It is 25% of perturbed profile wall at batch 8/context 32
  and roughly 67–84% at batches 64–256 across the matrix. Ragged batched prefill is therefore
  the first high-batch performance experiment and clearly clears Phase 0's 20% target rule at
  batches 64 and 256. The batch-8 share needs an uninstrumented A/B rather than a precise claim.
- Page planning costs 33–46 ms over 127 decode calls, about 2–3% of the nested decode event. The
  native-page path still does duplicate packed-index work, but timing does not support making it
  the first optimization.
- At batch 1, attention accounts for about 18–20% of profile wall inside decode. Together with the
  nearly fixed launch counts in the earlier FlashInfer profile, this keeps wider graph capture or
  larger captured regions as the leading batch-1 decode hypothesis.
- Window flushing reaches 407–690 ms at batch 256. This is host timing inside a nested scope; it
  merits an isolated A/B but cannot be read as an additive wall-time saving.
- Sampling is 3–18 ms per diagnostic pass and is not an engine-level priority on the greedy path.

## Persistent HTTP Serving

Source: `bench-results/esme-serving-baseline-20260710T005318.json`, produced on 2026-07-10 on an
A100-80GB PCIe. The run keeps two warmup bursts and ten measured bursts on the same persistent
engine/server, uses an explicit connection limit equal to batch size, copies finished token ids
once per request for the reference gate, and measures HTTP-visible ITL from client event times.
Each request generates 128 tokens.

| Mode | Batch | TTFT p50/p95 | ITL p50/p95 | Output tok/s | Reference |
| --- | ---: | ---: | ---: | ---: | --- |
| Greedy | 1 | 48.4/50.1 ms | 0.03/52.7 ms | 142.3 | 10/10 pass |
| Sampled | 1 | 47.6/48.3 ms | 32.6/33.6 ms | 30.5 | 10/10 pass |
| Greedy | 8 | 401/415 ms | 0.02/71.6 ms | 653.9 | 80/80 pass |
| Sampled | 8 | 395/424 ms | 44.8/46.9 ms | — | **0/80 pass** |
| Greedy | 64 | 2.85/3.19 s | 0.02/188 ms | 1,357.8 | 640/640 pass |
| Sampled | 64 | 2.88/3.38 s | 117/130 ms | — | **10/640 pass** |
| Greedy | 256 | 11.81/12.29 s | 0.02/612 ms | 1,480.9 | 2,560/2,560 pass |
| Sampled | 256 | 11.82/12.08 s | 392/466 ms | — | **2,527/2,560 pass** |

Greedy uses the fp32 full-recompute oracle with traced bf16 ties. Sampled uses a seeded
single-request run with the same bf16 backend and page geometry, so it tests batching, HTTP
dispatch, and per-request RNG invariance without claiming fp32 and bf16 sampling are identical.
Any failing sampled row suppresses tok/s; raw observed rates remain diagnostic only.

Greedy HTTP ITL is bursty because deferred eight-token windows deliver several tokens together:
p50 is about 0.02–0.03 ms, while p95 grows from 52.7 ms at batch 1 to 612 ms at batch 256. Sampled
requests do not use that fast path: even at batch 1, the accepted sampled row is 4.7x slower than
greedy in output tok/s. Fix sampled correctness before using its timing to guide optimization.

The first HTTP run was discarded. Tracing disabled deferred decode windows, a step observer
duplicated token materialization, warmups used fresh servers, and httpx's default 100-connection
cap polluted batch 256. No number from that run is used here.

## vLLM Comparison

Source: `bench-results/esme-vllm-baseline-full-20260709T235733.json`, produced on 2026-07-09 on
one reserved A100-80GB PCIe host. The fp32 oracle, `llm_infer`, and vLLM ran sequentially in
separate processes with matching hostname and GPU UUIDs. Each row generated 128 tokens per
request after two warmups and used the median of ten measurements. All 24 rows passed the oracle
gate; accepted bf16 ties are shown in the raw record.

| Context | Batch | `llm_infer` tok/s | vLLM tok/s | vLLM / `llm_infer` |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 154.0 | 492.6 | 3.20x |
| 32 | 8 | 869.0 | 3,335.7 | 3.84x |
| 32 | 64 | 2,336.2 | 19,040.0 | 8.15x |
| 32 | 256 | 2,802.4 | 38,283.1 | 13.66x |
| 256 | 1 | 151.2 | 485.1 | 3.21x |
| 256 | 8 | 842.8 | 3,191.6 | 3.79x |
| 256 | 64 | 2,168.8 | 17,492.0 | 8.07x |
| 256 | 256 | 2,609.8 | 30,802.9 | 11.80x |
| 768 | 1 | 150.5 | 470.9 | 3.13x |
| 768 | 8 | 799.4 | 2,994.9 | 3.75x |
| 768 | 64 | 1,890.5 | 12,426.1 | 6.57x |
| 768 | 256 | 2,211.0 | 17,828.1 | 8.06x |

Measured fact: vLLM 0.24.0 is 3.1–3.2x faster at batch 1 and the same-run gap grows with batch,
reaching 6.6–13.7x at batches 64 and 256. That pattern is consistent with `llm_infer` leaving
substantial batching and fixed-overhead gains on the table; it does not identify one responsible
kernel by itself.

Comparison limits:

- Both systems used the same converted Esme weights and physical GPU, but isolated dependency
  stacks: `llm_infer` used Torch 2.8/CUDA 12.8 and vLLM used Torch 2.11/CUDA 13.0.
- Prefix caching was disabled. One synthetic prompt was repeated within each context row to keep
  the fp32 oracle tractable. This comparison therefore does not exercise the eight-prompt pool
  that exposed the engine matrix's batch-8/context-768 failure.
- vLLM model construction took 121.9 s. `llm_infer` graph capture took 18.2 s and engine/KV
  construction took 6.0 ms. The `llm_infer` model-load timer in this record ended at host return,
  before the synchronization added after review, so startup values are recorded separately but
  are not used for a cross-engine startup claim.

## Tie Rule Scope

The generic fused-kernel fixture uses a 1e-3 logit tolerance. The Esme throughput harness uses
0.1 because its fast row converts the whole model to bf16, so rounding is not limited to one
attention kernel. The headline prompt pool has eight unique prompts; batches above eight repeat
them. Current accepted divergences include fp32 gaps up to 0.0463 logits.

This is evidence that the 1e-3 fixture threshold is too narrow for whole-model bf16. It is not a
measured error distribution supporting every gap below 0.1. The harness still recomputes the first
divergence with the fp32 oracle, records both candidate tokens and their gap, and suppresses tok/s
for any divergence above the threshold.
