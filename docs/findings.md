# Research Log

This log records what the `llm-infer` experiments asked, measured, and decided. It keeps the
negative results, because they explain the current design as much as the wins do.

Each entry has four parts: the question, the method, the result, and the decision. Numbers come
from Modal A100-80GB runs of `Esme-214M-Chat` unless an entry says otherwise. Raw JSON records
live in the gitignored `bench-results/` directory. Each entry names its record so a maintainer can
find it. The committed public records are [`assets/esme-ablation-ladder.json`](../assets/esme-ablation-ladder.json),
[`assets/esme-vllm-same-host.json`](../assets/esme-vllm-same-host.json), and
[`assets/esme-batch-curve.json`](../assets/esme-batch-curve.json).

Two rules apply to every entry:

- **Reference before speed.** A row reports tok/s only when its tokens qualify against the fp32
  full-recompute `PretrainBundleModel.logits()` oracle. See [scoping.md](scoping.md).
- **Same-run ratios only.** A/B arms run back to back in one container. Raw tok/s from different
  containers is not compared (entry 1 explains why).

## Summary

| # | Question | Answer | Decision |
| ---: | --- | --- | --- |
| 1 | Are cross-run numbers comparable? | No: identical code moves about ±20% across containers | Same-container A/B only |
| 2 | Where does 214M decode time go? | Kernel launches: about 2,100 per step | Attack launch count, not KV bookkeeping |
| 3 | Do piecewise CUDA graphs pay? | +177% / +45% / +16% at batch 8 / 64 / 256 | Shipped as the CUDA default |
| 4 | Does `torch.compile` do better? | Same launch cut, slower wall: 118.7 vs 425.0 tok/s at batch 8 | Rejected and deleted |
| 5 | What should the next experiment target? | Serial prefill: 67–84% of profile wall at batch 64–256 | Build ragged batched prefill |
| 6 | Does batched prefill pay? | Batch-64 TTFT 11.2x–65.6x; end-to-end 3.65x–5.32x | On by default |
| 7 | Do attention-inclusive graphs pay? | +18% to +30% at batch 1, but only while the batch holds its size | Server default; engine opt-in |
| 8 | Is op fusion worth it now? | Declined on a wrong premise; entry 14 corrects it | Reopened by 14 |
| 9 | What is the right sampled-output contract? | Same-shape seeded replay, not cross-shape exact tokens | Contract changed; sampled serving 3–5x faster |
| 10 | Do chunked prefill, prefix caching, preemption, and speculation work? | Mixed; see the entry | Defaults follow the measurements |
| 11 | Is the 0.1-logit bf16 bound measured? | No: it is an audited acceptance bound | Open; listed below |
| 12 | How far is the engine from vLLM? | Now 3.1x at batch 1 and 1.7–2.6x at batch 64/256 (was 6.6–13.7x) | Batch 1 is the open gap |
| 13 | What does each technique buy on one GPU? | 3.1x over HF batched `generate()`; graphs are the largest lever | Replace the naive-HF headline |
| 14 | Where does a batch-1 decode step go now? | 91% GPU-busy; 67% of the step is small non-GEMM kernels | Reopen fusion (entry 8 was wrong) |

## 1. Cross-container variance sets the comparison rule

**Question.** Can a change be judged by comparing its run against an earlier run?

**Method.** The same code ran in several Modal A100-80GB containers. Each record captures the GPU
SKU, SM clock, and power limit.

**Result.** Identical code moved about ±20%. Containers mixed A100 PCIe (300 W) and SXM4 (400 W)
parts, and SM clocks ranged from 1140 to 1410 MHz. The 214M decode loop is bound by the host CPU,
and the host CPU cannot be selected. One per-commit series showed a "regression" that was only a
1170 MHz container.

**Decision.** Every decision-grade comparison runs its arms back to back in one container, with
alternating order where possible. Cross-date raw tok/s is never used to justify a change.

Record: `esme-decode-profile-*` and `esme-decode-ablate-20260702T112802`.

## 2. At 214M, decode is launch-bound

**Question.** Where did the roughly 43 ms decode step go?

**Method.** `torch.profiler` and `cProfile` on the eager path at batch 8, 32, and 128.

**Result.** The step was CPU-bound, but not in the paged-KV code. It issued about 2,100
`cudaLaunchKernel` calls per step. `rms_norm` alone ran 121 times per step at about 7 ops each.
The GPU spent its "busy" time on about 9 µs kernels for about 3 GFLOP of decode math. Block-table
Python did grow with batch size: 594k `physical_slot` calls per run at batch 128.

Same-GPU ablation of the fixes (A100 PCIe, one container):

| Change | Effect |
| --- | --- |
| Deferred decode window (one host sync per 8 steps) | 0–3% alone |
| Planned window buffers (write slots and read layout built once per window) | +4.7% / +5.6% / +7.0% at batch 8 / 32 / 128 |
| All changes, like-for-like against the old baseline | +39% / +41% / +38% |

**Decision.** Keep all changes. The deferred window is worth little alone, but it is the
precondition for the planned buffers and for graph capture. The per-token host sync was never the
bottleneck, which the profile predicted.

## 3. Piecewise CUDA graphs remove the launch wall

**Question.** Can graph capture remove the launch overhead while the KV history changes size
every step?

**Method.** Capture 31 graph segments per decode step: the dense layer work between attention
calls. Attention and the paged-KV write stay eager, so no graph holds a ragged shape or a cache
address. Batches pad to a captured bucket. Pad rows skip attention and are never written to KV.
All buckets are captured before timing.

**Result.** Same container, reference-gated, zero non-tie divergences on every row:

| tok/s | Batch 8 | Batch 64 | Batch 256 |
| --- | ---: | ---: | ---: |
| Eager window + planned buffers | 152.9 | 606.5 | 987.2 |
| + piecewise CUDA graphs | 424.0 | 878.7 | 1,142.1 |
| Change | +177% | +45% | +16% |

Kernel launches per token step at batch 8 fell from 1,623 to 196 plus 27 graph replays. The gain
shrinks with batch because the launch count does not depend on batch size, while useful GPU work
does.

**Decision.** Shipped as the CUDA serving default.

Record: `esme-decode-capture-20260702T191632`.

## 4. Negative result: `torch.compile` loses to manual graphs

**Question.** Does vLLM's design, `torch.compile` with paged attention as an opaque custom op,
beat the hand-captured runner?

**Method.** The same step contract, buckets, and fallbacks, compiled with
`mode="reduce-overhead"` and graph partitioning at the attention op. Same container as the manual
runner. A recompile audit showed 0 new graphs across all 15 gated rows.

**Result.**

| Config | Batch 8 tok/s | Launches per pass at batch 8 | Startup |
| --- | ---: | ---: | ---: |
| Eager window | 155.2 | 12,986 | none |
| Manual piecewise graphs | 425.0 | 1,569 + 217 graph | 43.3 s |
| `torch.compile` | 118.7 | 1,576 + 217 graph | 407.2 s |

The GPU work is nearly the same. The loss is host-side Python per step: guard evaluation,
cudagraph-tree bookkeeping, and functionalization wrappers around 30 mutable custom-op calls. At
214M the step budget is a few milliseconds, so the framework overhead takes the whole graph win.
The design targets models whose steps take tens of milliseconds.

**Decision.** Rejected. The compiled runner and its tests were deleted.

Record: `esme-decode-capture-20260702T204006`.

An earlier static full-graph experiment on the Qwen backend was rejected for related reasons. A
fixed 32-slot bucket did extra work for inactive rows, ran slower, and changed the sampled
continuation.

## 5. A measurement baseline picks the next target

**Question.** Which bottleneck clears a 20%-of-wall bar for the next experiment?

**Method.** A fixed matrix: batch 1 / 8 / 64 / 256, cached context 32 / 256 / 768, 128 decode
tokens, two warmups, ten measured runs, persistent engine. Engine construction, KV-pool zeroing,
and graph capture were timed separately from steady state. A nested phase profile ran as a
separate diagnostic pass.

**Result.**

- Serial prefill took 67–84% of profile wall at batch 64–256.
- Page planning took 2–3% of decode.
- Window flushing reached 407–690 ms at batch 256, but inside a nested scope.
- Sampling took 3–18 ms per pass.
- Event recording itself added 4–36% to wall time, so the shares locate large regions only.

**Decision.** Ragged batched prefill was the first experiment. Page planning did not clear the bar
on timing, but the code showed duplicated work. That fix became a small separate A/B: −7.3% /
−7.3% / −11.9% decode wall at batch 1 / 8 / 64.

Records: `esme-measurement-baseline-20260710T001422`, `esme-page-plan-ab-20260710T140557`.

## 6. Ragged batched prefill

**Question.** Does packing burst admissions into one ragged prefill call amortize per-layer
overhead?

**Method.** Pack prompt tokens with no padding to the longest prompt. Write each request's K/V into
its own page table and use native GQA. Serial prefill stays as the reference and fallback. The
matrix ran two warmups plus ten alternating pairs per cell.

**Result.**

| Batch | TTFT speedup | End-to-end speedup (64 tokens) |
| ---: | --- | --- |
| 1 | about 0% (the serial path runs for one request) | about 0% |
| 8 | 8.9x–9.5x | 1.52x–1.55x |
| 64 | 11.2x (ctx 512) to 65.6x (ctx 16) | 3.65x–5.32x |

Under mixed load (8 steady decoders plus a burst), the decode stall across the burst step fell 5x
to 20x. For example, a 64-request ragged burst stalled 2,291 ms serial and 117 ms packed.

Divergence review: every divergence was a deterministic near-tie at a fixed step and token pair.
Packed and serial K/V agreed within 1–2 bf16 ulp per layer. The serial baseline itself diverged
from fp32 on other requests.

**Decision.** On by default, with no size cap. The mixed-load table is the data to size a cap if a
workload ever needs one. This change is the main reason the public batch-256 number rose from
3,776.8 to 13,490.5 tok/s.

Records: `esme-prefill-ab-20260710T113123`, `esme-mixed-load-20260710T125108`.

## 7. Attention-inclusive graph groups: the win depends on batch stability

**Question.** Can one graph cover several complete layers, including their KV writes and
FlashInfer attention calls?

**Method.** First a probe: does a fixed-batch FlashInfer wrapper replay correctly with stable page
buffers? It did, across page boundaries, with zero error and zero allocation delta. Then exact-batch
graphs of two and four layers ran against the piecewise runner, with fixed output length and
alternating pairs.

**Result.** With a fixed output length (the batch never shrinks):

| Group | Batch 1 | Batch 8 | Batch 64 | Batch 256 |
| --- | ---: | ---: | ---: | ---: |
| 2 layers | +13.4% | withheld | — | — |
| 4 layers | +29.8% | +28.1% (relative) | +4.2% | +22.0% |

Under real EOS termination, the gain fell to +0.5% to +6.3%. Grouped graphs exist only for exact
batch sizes. After the first request finishes, the batch drains below the captured size and the
run falls back to piecewise graphs. Grouped coverage was about 25% of decode steps. Capturing every
drain size would cost 6–10 s per bucket.

The HTTP server keeps the batch at its captured size more often:

| Workload | Piecewise tok/s | Grouped tok/s | Change |
| --- | ---: | ---: | ---: |
| Greedy, batch 1 | 143.1 | 168.9 | +18.0% |
| Greedy, batch 8 | 704.4 | 754.7 | +7.1% |
| Greedy, batch 64 | 1,838 | 1,960 | +6.6% |

**Decision.** Grouped graphs are the HTTP-server CUDA default. Single-request chat cannot drain,
so it clears the 10% bar. The raw engine keeps them opt-in, so fresh-engine benchmarks never pay
capture inside a timed region. Padded grouped buckets were measured at 2.5–6.0% and rejected.

The lesson generalizes: a fixed-length benchmark made this optimization look several times larger
than it is under real EOS termination.

Records: `esme-decode-four-layer-group-ab-*`, `esme-batch-curve-grouped-ab-20260710T174221`,
`esme-serving-grouped-ab-20260710T175637`.

## 8. Negative result: operator fusion declined

**Question.** Should the next phase pack QKV and gate/up GEMMs and fuse residual-add with RMSNorm?

**Result.** The evidence for it had expired. Graphs had already removed the launch cost that
fusion mostly targets. `torch.compile` had already tried the same elementwise fusions and lost.
On the graphed path, GPU kernel time is about 1–2 ms of a 5.5 ms batch-1 step. Even a 30% kernel
cut gives 5–10% of the step, below the 10% bar, before the cost of a dual reference path.

**Decision.** Closed without implementation. Reopen only if a profile of the graphed path shows
GPU-busy above half the step, with projection GEMMs dominating.

**Correction (2026-10-24).** The premise was wrong. A kernel-level profile of the graphed path
(entry 14) shows 5.0 ms of GPU kernel time in a 5.5 ms batch-1 step, not 1–2 ms. The reopen
condition was also aimed at the wrong target: GEMMs take 1.1 ms, and about 3.7 ms goes to small
elementwise, norm, and copy kernels, which is exactly what fusion removes.

## 9. The sampled-output contract was wrong, then sampled serving got 3–5x faster

**Question.** What should "correct" mean for seeded sampling under bf16?

**Method.** The first gate required each sampled request to reproduce its seeded single-request
stream exactly. A100 probes then compared RNG states and pre-sampling logits across batch shapes.

**Result.** The sampled batch-8 and batch-64 rows failed that gate from the first run. All
batchmates and all measured runs still agreed token for token with each other. RNG states were
identical per request. bf16 logits differed by up to 0.25 between batch shapes, and one flipped
multinomial draw forks the whole continuation. That is shape numerics, not an RNG bug.

**Decision.** The sampled gate is now same-shape seeded replay. Every record of one request
signature must carry an identical token stream. Agreement with the single-request anchor is stored
as evidence, not required. Any disagreement inside a replay group withholds tok/s. See
[serving-eval.md](serving-eval.md).

With a valid gate in place, the sampled path moved onto decode windows and graphs, with a batched
GPU sampler and burst token dispatch:

| Sampled HTTP | Before tok/s | After tok/s | Change |
| --- | ---: | ---: | ---: |
| Batch 1 | 30.3 | 148.6 | +390% |
| Batch 8 | 173 | 659 | +280% |
| Batch 64 | 516 | 1,811 | +251% |

Request latency p50/p95 fell 71–80%. Engine-only greedy did not regress (+0.9% to +2.9%). ITL p95
rose 52–84%, because tokens now arrive in window bursts. That is a deliberate trade:
`decode_window_size` tunes it.

Record: `esme-serving-phase5-ab-20260711T081847` (two revisions in one container, candidate first).

## 10. Serving techniques, measured one at a time

These rows ran on the former flash-attn backend. Each is a same-run on/off pair with every output
reference-gated.

| Technique | Workload | Result |
| --- | --- | --- |
| Prefix caching | 16 siblings sharing a 513-token prompt | 8,208 → 513 prompt tokens prefilled; 1.40x end to end |
| Preemption (recompute) | 12 requests in a 40-block pool that needs about 96 | 7 real evictions, 12/12 complete and exact or tie; 1.31x faster than strict reservation |
| Prompt-lookup speculation | One repetition-heavy request | 1.47x batch-1 latency, 2.0 tokens per verify step |
| Chunked prefill (128) | 8 decoders plus 4 long-prompt arrivals | About 21% slower total, and no visible stall protection |

Chunked prefill is a negative result at this model size. Prefill at 214M is launch-bound, so
splitting it adds passes without shortening a stall that matters. Batched prefill (entry 6) solved
the measured burst problem instead.

Speculation is off on CUDA. The positive row covers one repetition-heavy request, and it predates
the FlashInfer baseline. On local CPU serving, speculation and the cross-turn prefix cache are on by
default. fp16 on CPU is about 12% faster than fp32 with
byte-identical output on the 8-turn chat benchmark.

## 11. Numerical contract: what the bf16 gate does and does not prove

The fast path runs the whole model in bf16, so it can choose a different token than fp32 where
the fp32 top-two margin is small. The harness recomputes the first divergence on the fp32 oracle.
It accepts the row automatically when the margin is within 0.1 logits.

Findings so far:

- The generic fused-kernel fixture tolerance (1e-3) is too narrow for whole-model bf16.
- The accepted gaps observed so far reach 0.0463 logits. The headline pool has only 8 unique
  prompts. The 0.1 bound is therefore an audited acceptance boundary, not a measured percentile.
- Several divergences exceed 0.1: `esme-001` at context 768 (0.108), `esme-007` step 22 (0.130),
  and batched-prefill rows at 0.136, 0.142, and 0.146. None was caused by the change under test.
  At `esme-007`, baseline and candidate chose the same token. At `esme-001`, graphs, decode
  windows, attention backends, page planning, and KV layout all chose the same token. The
  batched-prefill rows were deterministic near-ties, with K/V within 1–2 bf16 ulp of serial. The
  difference accumulates in whole-model bf16 cached decode, not in one kernel.
- These rows keep raw timing but report no public tok/s. No fp32 fallback was added to force token
  identity for one marginal decision.

## 12. The comparison that matters: vLLM on the same host

**Question.** How far is this engine from a mature serving engine on the same model and GPU?

**Method.** One reserved A100-80GB host per run. The fp32 oracle, two `llm_infer` arms, and vLLM
0.24.0 ran as sequential child processes, with matching hostname and GPU UUID checked. The
`llm_infer` arm uses the raw-engine defaults (piecewise graphs). The `llm_infer_grouped` arm adds
exact-batch grouped graphs, as the HTTP server does on CUDA. Both arms capture graphs at every
measured batch size; the server's default buckets stop at 16. Each row generated 128 tokens per
request with EOS ignored, after two warmups, as the median of ten runs.

**Result** (2026-10-24, A100-SXM4-80GB). All 36 rows are headline-eligible with zero non-tie
divergences: 14 rows are exact, and 22 contain accepted bf16 ties.

| Context | Batch | `llm_infer` tok/s | `llm_infer_grouped` tok/s | vLLM tok/s | vLLM / `llm_infer_grouped` | July ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 186.0 | 184.7 | 564.7 | 3.06x | 3.20x |
| 32 | 8 | 1,208.1 | 1,208.1 | 3,416.3 | 2.83x | 3.84x |
| 32 | 64 | 9,196.4 | 9,168.7 | 24,142.2 | 2.63x | 8.15x |
| 32 | 256 | 25,836.8 | 26,188.6 | 44,863.5 | 1.71x | 13.66x |
| 256 | 1 | 180.8 | 180.9 | 563.1 | 3.11x | 3.21x |
| 256 | 8 | 1,173.4 | 1,188.6 | 3,298.7 | 2.78x | 3.79x |
| 256 | 64 | 7,743.5 | 7,858.5 | 20,000.4 | 2.55x | 8.07x |
| 256 | 256 | 18,298.8 | 18,911.1 | 35,181.0 | 1.86x | 11.80x |
| 768 | 1 | 177.1 | 178.7 | 550.8 | 3.08x | 3.13x |
| 768 | 8 | 1,131.4 | 1,138.4 | 3,080.9 | 2.71x | 3.75x |
| 768 | 64 | 6,186.0 | 6,197.1 | 13,359.6 | 2.16x | 6.57x |
| 768 | 256 | 11,741.0 | 11,851.3 | 19,876.7 | 1.68x | 8.06x |

The July ratio column comes from the 2026-07-09 run (A100 PCIe, same harness, same vLLM version,
`llm_infer` arm only). Compare ratios across the two runs, not raw tok/s: the hosts differ, and
vLLM itself measured 2–27% faster on the new host.

**What changed.**

- **The high-batch gap closed from 6.6–13.7x to 1.7–2.6x.** In July, serial prefill took 67–84% of
  `llm_infer` wall at batch 64–256. Batched prefill (entry 6) removed most of it.
- **The batch-1 gap did not move: 3.1x then, 3.1x now.** Batch 1 has no batching effect and no
  prefill packing, so this is now the cleanest open question.
- **Grouped graphs give no gain in this harness: −0.7% to +3.4%.** They ran on every decode step
  (4,572 steps per bucket), and EOS is ignored, so batch drain does not explain it. In July a
  fixed-length A/B measured +29.8% at batch 1. A plausible cause is the native-GQA decode change in
  PR #27, which removed per-layer eager KV expansion between graph segments. That eager work is
  what grouped graphs absorbed. This is a hypothesis, not a measurement. The HTTP-server default
  needs a fresh serving A/B before anyone relies on its +18% claim.

**Limits.**

- The two systems use separate dependency stacks: Torch 2.8 / CUDA 12.8 against Torch 2.11 /
  CUDA 13.0.
- Each context row repeats one synthetic prompt, and EOS is ignored. The ladder (entry 13) covers
  the natural chat workload with real EOS.
- Startup is not part of the comparison. vLLM took 62.6 s to build; `llm_infer` took 14.4 s for
  piecewise capture plus about 13 s for grouped capture.

Record: [`assets/esme-vllm-same-host.json`](../assets/esme-vllm-same-host.json).
Reproduce: `modal run scripts/modal_esme_vllm_baseline.py --command full`.

## 13. Ablation ladder: what each technique buys, on one GPU

**Question.** Entries 2–7 ran in different containers, so their gains cannot be added. What does
each technique contribute when all of them run on one GPU, and how does the engine compare with
plain batched HF?

**Method.** One A100-SXM4-80GB container on the headline workload: chat prompts, up to 256 new
tokens, real EOS, greedy. Two HF baselines run first: sequential `generate()` (the naive floor)
and left-padded batched `generate()`. Then five cumulative engine rungs each add one technique.
Every engine rung is a persistent engine built before timing, and rung order rotates each
iteration. Medians come from five runs after one warmup. All 19 rows passed the fp32 gate with
zero non-tie divergences.

**Result** (2026-10-24, raw tok/s):

| System | Batch 8 | Batch 64 | Batch 256 |
| --- | ---: | ---: | ---: |
| HF sequential `generate()` | 59.9 | — | — |
| HF batched `generate()` | 249.4 | 2,079.3 | 6,070.9 |
| Paged KV + continuous batching | 259.9 | 1,602.6 | 3,173.5 |
| + batched prefill | 283.5 | 2,084.7 | 6,460.1 |
| + decode window | 309.3 | 2,402.8 | 8,407.7 |
| + piecewise CUDA graphs | 778.7 | 5,865.9 | 18,637.1 |
| + grouped graphs | 777.7 | 5,867.8 | 18,784.9 |
| **Full engine / HF batched** | **3.12x** | **2.82x** | **3.09x** |

**Findings.**

- **Against an honest baseline, the engine is about 3x, not 610x.** Most of the old headline
  multiple was plain batching: HF batched alone is 4.2x faster than HF sequential at batch 8.
- **Serial prefill, not paging, is the first rung's deficit.** That rung prefills one request at
  a time and syncs every decode step. It runs at 0.77x of HF batched at batch 64 and 0.52x at
  batch 256. Adding batched prefill alone brings it level (1.00x and 1.06x). Paged KV buys memory
  capacity (entry 10, and the KV evidence in the batch-scaling curve); speed comes from the
  techniques above it.
- **CUDA graphs are the largest single lever: 2.53x / 2.44x / 2.24x.** This is the launch-bound
  diagnosis of entry 2, measured on the full current stack.
- **Batched prefill scales with batch: 1.05x / 1.30x / 2.03x.** That is the shape entry 5
  predicted from the prefill share of wall time.
- **The decode window adds 1.10x / 1.15x / 1.29x**, more than the 0–3% it gave alone in July.
  It now also carries the planned buffers and stable page metadata.
- **Grouped graphs add 1.00x / 1.00x / 1.01x** under real EOS, consistent with entries 7 and 12.
- **The naive-HF floor is host-sensitive:** 59.9 tok/s here, 22.1 tok/s in the July headline run.
  A multiple over that floor mostly measures the host CPU.

**Limits.** Different rungs resolve bf16 ties differently, so output lengths differ by up to 4.4%
(1,263 to 1,318 tokens at batch 8). The adjacent-rung speedups above are raw median wall-time
ratios at batch 8 / 64 / 256. Exact output parity holds for only 5 of the 12 rung transitions, so
the record withholds the parity-qualified ratio for the other 7; the figure plots tok/s, which
differs from the wall-time ratio by the token-count difference. HF batched runs before the
engine rungs, outside their rotation, so the engine-over-HF ratio is not drift-controlled the
way adjacent rungs are.

Record: [`assets/esme-ablation-ladder.json`](../assets/esme-ablation-ladder.json).
Reproduce: `modal run scripts/modal_esme_batch_curve.py --command ladder`.

## 14. Batch-1 roofline: the step is GPU-bound on kernel count

**Question.** vLLM decodes 3.1x faster at batch 1 (entry 12), where batching cannot explain the
gap. Where does one batch-1 token step go, and how far is it from the hardware floor?

**Method.** One A100-SXM4-80GB container, default CUDA path: bf16 weights, FlashInfer, piecewise
CUDA graphs, decode window 8. A persistent engine decodes 128 fixed tokens per request, so the
batch never shrinks. An uninstrumented run gives wall time per token step (median of 3 after 2
warmups). A separate `torch.profiler` run over decode only gives each kernel's device time from
CUPTI, including kernels inside graph replays. The floor is the time to stream every weight
once at the A100's 2.039 TB/s spec bandwidth: 427.9 MB of bf16 weights, plus the KV reads.

**Result** (ms per token step):

| Batch | Wall | GEMM | Attention | Other kernels | Idle gap | GPU busy | HBM floor | Wall / floor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.51 | 1.13 | 0.19 | 3.68 | 0.51 | 91% | 0.21 | 26x |
| 8 | 6.88 | 1.76 | 0.20 | 4.39 | 0.53 | 92% | 0.22 | 31x |
| 64 | 7.74 | 1.64 | 0.30 | 4.54 | 1.26 | 84% | 0.29 | 27x |

**Findings.**

- **The step is GPU-bound, not host-bound.** With graphs on, the GPU is busy 84–92% of each
  step. The idle gap is wall time minus kernel time, so it also holds the gaps between graph nodes;
  host dispatch is at most that 0.5 ms at batch 1. The host-dispatch story of entries 2–4 was true before graphs. It is not the current
  bottleneck.
- **Kernel count is the bottleneck.** At least 1,870 kernels run per token step at every batch
  size (counting only the 20 largest kernel types). That gives an upper bound of 2.6–3.5 µs of
  device time per kernel. "Other" kernels take 67% of the batch-1 step: RMSNorm as separate
  mean, pow, rsqrt, multiply, and cast kernels (121 norm calls per step), residual adds, dtype
  copies, index gathers, and small memory copies. Graphs remove the CPU cost of
  a launch, but each kernel still costs GPU time to start and drain.
- **GEMMs are not bandwidth-efficient either.** At batch 1 the GEMV kernels stream the weights at
  0.38 TB/s, 19% of spec bandwidth. That is the second lever, after kernel count.
- **The whole step is 26x the HBM floor.** Even vLLM, at about 1.8 ms per step, is near 8x the
  floor. At 214M parameters, decode latency is set by kernel count and size, not by bytes moved.

**What this predicts.** If each layer's elementwise chain (residual add with RMSNorm, the norm
itself, RoPE with the QK norm, SiLU with the gate multiply) ran as a few kernels instead of
dozens, most of the 3.7 ms would go. If "other" fell to about 0.7 ms, a batch-1 step would take
about 2.5 ms, about 400 tok/s against vLLM's 563. That number is an estimate from this profile,
not a measurement.

There is evidence against the easy route. Entry 4's `torch.compile` runner did not lower GPU
work: 59.9 ms of GPU busy time per pass against 55.2 ms for manual graphs. So Inductor, as
configured then, did not fuse these chains in a way that saved device time. Why is unknown; that
record is not in the repository.

**Decision.** Reopen fusion (entry 8). The next experiment counts kernels before it times
anything. It first tests whether a candidate actually cuts the per-step kernel count and device
time: compiled segments captured inside the manual graphs, or a fused RMSNorm and residual add.
Only a candidate that does goes on to the usual gate: fp32 reference, then a same-container A/B.

Record: [`assets/esme-decode-roofline.json`](../assets/esme-decode-roofline.json).
Reproduce: `modal run scripts/modal_esme_decode_profile.py --command roofline --batch-sizes 1,8,64`.

## Open questions

- Does compiled elementwise fusion inside the manual graphs close the batch-1 gap that entry 14
  predicts?
- Do grouped graphs still pay anywhere? Entries 12 and 13 show no gain on the current engine. The
  HTTP serving A/B that justified the server default predates PR #27.
- What logit-error distribution supports the 0.1 bf16 bound? This needs many more unique prompts
  than the 8 in the headline pool.
