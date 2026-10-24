# Research Log

This log records what the `llm-infer` experiments asked, measured, and decided. It keeps the
negative results, because they explain the current design as much as the wins do.

Each entry has four parts: the question, the method, the result, and the decision. Numbers come
from Modal A100-80GB runs of `Esme-214M-Chat` unless an entry says otherwise. Raw JSON records
go to `bench-results/`, which is not committed. Entries 12 onward commit the records behind their
claims to [`assets/`](../assets/) and link them. Earlier entries name records that are not
published.

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
| 8 | Is op fusion worth it now? | Declined on a wrong premise; entry 14 corrects it | Reopened by 14; entry 15 tested it |
| 9 | What is the right sampled-output contract? | Same-shape seeded replay, not cross-shape exact tokens | Contract changed; sampled serving 3–5x faster |
| 10 | Do chunked prefill, prefix caching, preemption, and speculation work? | Mixed; see the entry | Defaults follow the measurements |
| 11 | Is the 0.1-logit bf16 bound measured? | No: it is an audited acceptance bound | Measured in entry 20 |
| 12 | How far is the engine from vLLM? | Now 3.1x at batch 1 and 1.7–2.6x at batch 64/256 (was 6.6–13.7x) | Batch 1 is the open gap |
| 13 | What does each technique buy on one GPU? | 3.1x over HF batched `generate()`; graphs are the largest lever | Replace the naive-HF headline |
| 14 | Where does a batch-1 decode step go now? | 91% GPU-busy; 67% of the step is small non-GEMM kernels | Reopen fusion (entry 8 was wrong) |
| 15 | Does compiling the elementwise chains speed up decode? | Kernels halve; raw wall does not follow | Not shipped; see 16 |
| 16 | Where does the compiled arm's idle time go? | Inconclusive: node tracing reverses the effect it should explain | Measure host issue time without CUPTI next |
| 17 | Is the compiled arm host-bound or GPU-bound? | Gate failed: compiled was faster here. FlashInfer `plan()` syncs every step | Remove the sync; then target eager attention |
| 18 | Do grouped graphs still pay in the server? | Depends on the host: +10.3% at batch 1 with exact parity on one class, −1.3% on another | Keep the server default; record the host CPU |
| 19 | What does removing the per-step `plan()` sync buy? | Pilot rule failed (underpowered); the registered confirmatory run passed: grouped −1.6% to −3.8%, exact parity | Shipped, with a fence for a race the sync had hidden |
| 20 | What bf16 logit error does the 0.1 tie rule have to cover? | Measured: p99.9 shrink 0.25; 7 of 807 flips exceed 0.1; none where the margin is 0.3+ | Bound stays at 0.1; reviews of near-ties cite entry 20 |
| 21 | Does grouping all 30 layers beat the server's four-layer group? | Yes: −2.9% to −4.6% per step at batch 1–64, exact parity, qualified | Full depth becomes the server default |
| 22 | Does prompt-lookup speculation pay on the CUDA engine? | No: 3.5x to 4.2x slower per request; an eager verify step costs about ten graphed decode steps | Stays off; a graphed verify path is open |

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
  fixed-length A/B measured +29.8% at batch 1. A plausible cause is the native-GQA decode change,
  which removed per-layer eager KV expansion between graph segments. That eager work is
  what grouped graphs absorbed. This is a hypothesis, not a measurement. The HTTP-server default
  needed a fresh serving A/B before anyone relied on its +18% claim. Entry 18 ran it three times:
  the batch-1 gain is +10.3% or −1.3%, depending on the host class.

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
not a measurement. Entry 15 tested it: the kernel count halved, but batch 1 did not get faster.

There is evidence against the easy route. Entry 4's `torch.compile` runner did not lower GPU
work: 59.9 ms of GPU busy time per pass against 55.2 ms for manual graphs. So Inductor, as
configured then, did not fuse these chains in a way that saved device time. Why is unknown; that
record is not in the repository.

**Decision.** Reopen fusion (entry 8). The next experiment counts kernels before it times
anything. (Entry 15 ran it: fewer kernels did not mean a faster step, so kernel count alone is not
the whole bottleneck.) It first tests whether a candidate actually cuts the per-step kernel count and device
time: compiled segments captured inside the manual graphs, or a fused RMSNorm and residual add.
Only a candidate that does goes on to the usual gate: fp32 reference, then a same-container A/B.

Record: [`assets/esme-decode-roofline.json`](../assets/esme-decode-roofline.json).
Reproduce: `modal run scripts/modal_esme_decode_profile.py --command roofline --batch-sizes 1,8,64`.

## 15. Fusion probe: half the kernels, no faster

**Question.** Entry 14 predicted that fusing the elementwise chains would cut most of the 3.7 ms of
small kernels and bring batch-1 decode near 400 tok/s. Does it?

**Method.** Two arms in one container, each measured with the entry-14 roofline protocol. The
`eager` arm is the default path. The `compiled` arm replaces `rms_norm` and `apply_rope` with
`torch.compile` versions (no compiler CUDA graphs), runs them once on the eager decode path, then
captures the usual manual piecewise graphs around the compiled kernels. Capture took 19.3 s
against 9.4 s for the eager arm, so some compilation likely continued during the capture warmup;
capture safety relies on that warmup, which runs every segment before recording it. Both arms are
gated against the fp32 oracle, which runs with the original functions. A second batch-1 run adds
host-side counters: graph launches, kernel launches, and the most expensive host operations.

**Result** (A100 80GB PCIe, ms per token step):

| Arm | Batch | Kernel launches | Kernel time | Other kernels | Idle gap | Wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 1 | 2,008 | 5.02 | 3.68 | 0.04 | 5.06 |
| compiled | 1 | 921 | 2.73 | 1.42 | 2.77 | 5.51 |
| eager | 8 | 2,099 | 6.39 | 4.41 | 0.21 | 6.60 |
| compiled | 8 | 1,012 | 3.70 | 1.75 | 2.24 | 5.94 |

Both arms have zero non-tie divergences against fp32, and both decode the same fixed number of
tokens. They resolve some bf16 ties differently, so direct parity is `review_required`, not
exact. Under the reference policy that withholds any qualified relative speed claim: the wall
times above stay raw diagnostic timing, and the percentages below describe raw timing only.

**Findings.**

- **The kernel-level prediction holds.** Compiling two functions cuts kernel launches per step by
  54% and kernel time by 45%. RMSNorm becomes one fused Triton reduction instead of about eight
  kernels.
- **The wall-time prediction fails on raw timing.** Raw batch-1 wall is 9% higher; a repeat run
  measured 17% higher. Raw batch-8 wall is 10% lower. The idle gap grows by 2.7 ms at batch 1
  and 2.0 ms at batch 8. None of these is a qualified A/B result until the parity mismatches are
  reviewed; the decision below does not depend on one.
- **Both host and GPU cost fell, so the compiled arm added time somewhere else.** Both arms replay
  the same 31 graphs and issue the same 156 ordinary kernel launches per step, with the same host
  operations on top. The CPU cost of `cudaGraphLaunch` halves (3.29 to 1.58 ms per step under the
  profiler), because a graph with fewer nodes is cheaper to launch. If the step were simply the
  longer of host time and GPU time, it would have got faster. It did not, so the compiled arm
  adds a cost that neither counter sees.
- **The unexcluded suspect is GPU-side gaps between graph nodes around the Triton kernels.** Every
  compiled call runs inside a graph segment, so it adds no host work at replay. The profiler's
  host times are inflated (the top host operations sum to more than the uninstrumented wall), so
  they cannot rank host against GPU. A timeline trace (Nsight Systems) of one eager and one
  compiled step can show where the extra time sits.

**Decision.** Do not ship the compiled path. Entry 14's 400 tok/s estimate assumed the step would
shrink with GPU time; this probe disproves that assumption. The next measurement is that timeline
trace, and the next fusion attempt depends on what it shows. Entry 16 ran the trace.

Records: [`assets/esme-decode-fusion-probe.json`](../assets/esme-decode-fusion-probe.json)
(batches 1 and 8) and
[`assets/esme-decode-fusion-probe-host.json`](../assets/esme-decode-fusion-probe-host.json)
(batch 1 with host counters). Reproduce:
`modal run scripts/modal_esme_decode_profile.py --command fusion-probe --batch-sizes 1,8`.

## 16. Timeline trace: inconclusive, and why

**Question.** Entry 15's compiled arm halved kernel time but not wall time. Where does its extra
idle time sit?

**Method.** One A100 80GB PCIe container. Each arm runs a fresh batch-1 process under Nsight
Systems 2024.6 with CUDA graph node tracing, so every kernel inside a graph replay appears on the
timeline. The trace covers only 63 decode steps after warmup. Every GPU activity carries the
correlation id of the host API call that issued it, which lets the analysis split each idle gap
by cause.

**Result** (ms per token step, under tracing):

| Arm | GPU activities | GPU busy | Total idle | Span under tracing | Uninstrumented wall (entry 15) |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | 2,004 | 4.91 | 5.50 | 10.41 | 5.06 |
| compiled | 917 | 2.67 | 4.97 | 7.64 | 5.51 |

**Findings.**

- **The trace does not reproduce the effect.** Uninstrumented, the compiled arm is slower than
  eager. Under node tracing it is faster, with less total idle time. A trace that reverses the
  effect cannot explain it.
- **Tracing cost sits on the host.** GPU busy time matches the uninstrumented kernel time in both
  arms (4.91 against 5.04 ms, 2.67 against 2.73 ms), and gaps inside a graph replay average about
  55 ns. The extra 4–5 ms per step lands at launch boundaries: about 226 per step (31 graph
  launches, 156 eager kernel launches, and the host copies), about 22–24 µs each. Node tracing
  therefore makes both arms host-bound on the tracer itself.
- **A first reading of this trace was wrong.** It read the rise in GPU time spent waiting on eager
  `cudaLaunchKernel` calls (0.14 to 1.88 ms per step) as the cause. That metric counted only the
  time before each API call started, so time spent inside a long `cudaGraphLaunch` counted as
  launch latency, which fell by about the same amount (4.49 to 2.25 ms). The shift is relabeled
  host time, not a new cost. The analysis now splits each gap into time before the call, time
  inside the call, and the remainder; the committed record predates that split. The same reading
  also claimed host waits were lower bounds. That is backwards: tracing slows the host, so it
  inflates them.

**Decision.** Stop tracing this question with CUPTI. Any CUPTI trace adds host cost at every
launch, the quantity under test. The discriminating measurement is uninstrumented: time from the
start of a decode step until the host issues its last launch. If that host issue time is close to
the compiled arm's 5.5 ms wall, the step is host-bound and some host cost that no CUDA counter
sees has grown. If it is far below the wall, the idle time is on the GPU side. The predictions
for each case are written down before that run.

Record: [`assets/esme-decode-timeline.json`](../assets/esme-decode-timeline.json) (gap buckets
use the earlier metric). Reproduce:
`modal run scripts/modal_esme_decode_profile.py --command timeline --batch-sizes 1`.

## 17. Host issue time without CUPTI (predictions registered before the run)

**Question.** Entry 15's compiled arm cut kernel time from 5.0 to 2.7 ms per batch-1 step, yet
its step got slower. Is that step host-bound, or does the GPU sit idle for a reason on its side?

**Method.** One A100 container, batch 1, both arms, the roofline protocol (persistent engine,
fixed 128-token output). Three passes per arm. `uninstrumented` measures wall time only. `issue`
adds two `perf_counter` timers: around `engine.step()`, and around `Event.synchronize()`, the
window flush's only explicit wait. Host issue time is step time minus that wait. `regions` adds
host timers around each graph replay, each eager attention region, each FlashInfer `plan()` call,
and each window open. No CUPTI tracing runs, because entry 16 showed it changes the answer.

**Gate (decided before the run).** Proceed to interpretation only if the compiled arm's
uninstrumented wall exceeds the eager arm's in this container, and the `issue` pass adds at most
5% to each arm's wall. Otherwise the result is that the effect did not reproduce, or that this
instrumentation is too costly, and the question stays open.

**Predictions (written before the run).**

- **Host-bound:** in the compiled arm, host issue time is at least 90% of wall, and
  `Event.synchronize` is small. The host has not finished issuing when the GPU runs dry. Some host
  cost that no CUDA counter sees has grown, and the `regions` pass should show where.
- **GPU-side:** in the compiled arm, host issue time is below 70% of wall, and `Event.synchronize`
  absorbs most of the rest. The host finishes early and waits, so the idle time sits on the GPU
  side: between graph nodes or at the boundaries between graphs and eager kernels.
- **Either way:** the eager arm's host issue time stays below its wall, because entry 14 measured
  that arm as 91% GPU-busy.
- **Regions:** an average graph-replay call above 200 µs means a launch blocks on the host.

**Result.** The gate failed on its first condition, so entry 15's question stays open. This
container was an A100-SXM4 (500 W). Entries 15 and 16 ran on A100 PCIe. Here the compiled arm was
faster, not slower (raw timing, no speed claim):

| Pass (ms per token step) | Eager | Compiled |
| --- | ---: | ---: |
| Uninstrumented wall | 5.105 | 3.100 |
| `issue` overhead vs uninstrumented | +0.01% | −0.03% |
| Host issue time / wall (means over 3 runs) | 99.6% | 99.9% |
| `Event.synchronize` | 0.018 | 0.002 |

Every instrumented pass added at most 0.5% to wall, so the method is cheap enough to repeat on a
PCIe host. Two corrections to this record: the harness ran six warmup decodes per arm, not the
two it declares, and it stored host time as a mean over the three runs while wall is their
median. The table compares means to means. The harness now keeps host time per run and
reports the median run's.

**Outside the gate: the planner syncs on every step.** The `regions` pass found where the eager
arm's wait hides. It is not in `Event.synchronize`. It is in FlashInfer's `plan()`:

| Region (ms per token step) | Eager | Compiled | Calls per step |
| --- | ---: | ---: | ---: |
| FlashInfer `plan()` | 2.297 | 0.256 | 1 |
| Eager attention | 2.417 | 2.493 | 30 |
| Graph replay | 0.178 | 0.195 | 31 |
| Window open | 0.027 | 0.021 | 0.13 |

FlashInfer 0.6.14's `BatchDecodeWithPagedKVCacheWrapper.plan()` copies `indptr` and
`last_page_len` to the CPU. The engine passed device tensors, so every call was a blocking
device-to-host copy. The host waited there for the previous step's GPU work to drain. This is why
the eager arm's host issue time was about 100% of wall and the third prediction failed: the wait
sat inside `engine.step()`, not in the window flush. The compiled arm's GPU work drains sooner, so
the same call waited only 0.26 ms. What remains is host time for the 30 eager attention regions:
about 2.5 ms per step, or 83 µs per layer. On this host that is the floor for a compiled batch-1
step.

Both consequences are untested. Host-side metadata removes the sync, which lets the host issue
step n+1 while the GPU runs step n. Capturing attention inside graphs removes most of the 2.5 ms.
Grouped graphs already capture attention. Entry 18 finds their batch-1 gain only on the slower
host class, which fits host time limiting the piecewise step there.

Record: [`assets/esme-decode-host-timeline.json`](../assets/esme-decode-host-timeline.json).
Reproduce: `modal run scripts/modal_esme_decode_profile.py --command host-timeline --batch-sizes 1`.

## 18. Grouped graphs in the HTTP server, rerun

**Question.** Do grouped graphs still pay in the HTTP server? Entry 7's serving A/B predates
native-GQA decode, and the raw-engine harness of entries 12 and 13 showed no gain.

**Method.** Streaming requests over localhost HTTP, greedy and seeded sampled workloads, batch 1,
8, and 64, 128 new tokens, 2 warmups, and 5 measured runs per row. Graph capture for batch sizes 1
to 256 ran before timing. Three runs, each in its own A100-SXM4 container:

- **A** (400 W power limit): every piecewise row ran before every grouped row.
- **B** (500 W): the arm that runs first alternates from cell to cell.
- **C** (400 W): as B, plus direct parity. Each request's output tokens are hashed, and a cell
  counts as a qualified A/B only if both arms emitted the same streams.

**Result** (grouped against piecewise; C is the only run with direct parity):

| Workload | Batch | A: fixed order | B: balanced | C: balanced, parity |
| --- | ---: | ---: | ---: | ---: |
| Greedy | 1 | +12.3% | −1.3% | **+10.3%** |
| Sampled | 1 | +8.6% | −0.2% | **+5.3%** |
| Greedy | 8 | +2.5% | +2.7% | **+4.9%** |
| Sampled | 8 | −1.1% | −1.0% | **+0.9%** |
| Greedy | 64 | +7.1% | +2.2% | **+1.4%** |
| Sampled | 64 | +35.3% | +1.5% | **+3.5%** |

Every row in every run passed its reference gate. In run C every cell has exact direct parity, so
its changes are qualified relative results. A and B are raw timing.

Greedy batch-1 tok/s by run:

| Arm | A (400 W) | B (500 W) | C (400 W) |
| --- | ---: | ---: | ---: |
| Piecewise | 158.2 | 190.3 | 162.8 |
| Grouped | 177.6 | 187.8 | 179.5 |

- **Run A's +35% at sampled batch 64 came from the fixed order.** It disappears in both balanced
  runs.
- **The batch-1 gain depends on the host class.** Grouped graphs hold batch 1 at 178–188 tok/s
  in all three containers. The piecewise arm drops to 158–163 tok/s in the two 400 W containers.
  So the batch-1 gain measures how much the piecewise path suffers on that host, not a fixed
  speedup.
- **Why is untested.** Entry 17 measured about 2.5 ms of host time per step in the 30 eager
  attention regions, which grouped graphs capture. A slower host CPU would slow the piecewise
  step more than the grouped one. The power limit is only a proxy for the host class: these
  records do not name the CPU. Benchmark snapshots now record it.

**Decision.** Keep grouped graphs as the HTTP-server CUDA default. On one host class they give
+10.3% at batch 1 with exact parity. On the other they cost nothing but about 13 s of startup
capture. Run B also carries a lesson: one container is not a result, even with balanced order.

Records: [`assets/esme-serving-grouped-ab.json`](../assets/esme-serving-grouped-ab.json) (C),
[`assets/esme-serving-grouped-ab-balanced-500w.json`](../assets/esme-serving-grouped-ab-balanced-500w.json)
(B), and [`assets/esme-serving-grouped-ab-fixed-order.json`](../assets/esme-serving-grouped-ab-fixed-order.json)
(A). All three drop per-token inter-token latencies. Reproduce:
`modal run scripts/modal_esme_serving_eval.py --batch-sizes 1,8,64 --measured-runs 5 --grouped-ab`.

## 19. Host-side page metadata for FlashInfer's planner (predictions registered before the run)

**Question.** Entry 17 found that FlashInfer's `plan()` blocks on a device-to-host copy on every
decode step. The engine now passes CPU copies of `indptr` and `last_page_len` (built from the host
lists it already has), and keeps `indices` on the device. What does removing that sync buy?

**Method.** One A100 container, batch 1, 8, and 64, fixed 128-token greedy output with EOS
ignored. Three engine configurations: `piecewise` (raw-engine default), `grouped` (HTTP-server
CUDA default), and `compiled` (piecewise with `torch.compile` elementwise chains, entry 17's
host-bound case). Each configuration runs both metadata arms on one persistent engine with the
same graphs: `device` restores the old behavior by removing the host copies at both planner call
sites, and `host` is the fix. Two warmup rounds, then five measured rounds that alternate which
arm goes first. Every arm gates against the fp32 oracle, and the record stores direct
device/host parity.

**Predictions (written before the run).** The arithmetic uses entry 17's batch-1 numbers: about
2.8 ms of host issue time per eager step against about 4.6 ms of GPU time, and about 2.7 ms of
each in the compiled arm.

- **Parity:** exact in every configuration and batch. The planner sees the same values, so any
  other result is a bug.
- **Piecewise and grouped, batch 1:** the GPU is the bottleneck, so the sync costs only the gap
  before the next step's first kernel. Gain below 5%.
- **Compiled, batch 1:** host and GPU time are about equal, so overlap pays most here. Wall falls
  from about 3.1 ms toward 2.8 ms: a gain of 5–12%.
- **Batch 8 and 64:** GPU time per step grows, so the fixed sync cost matters less. Gain below 3%.

**Decision rule (set before the run).** Keep the fix if parity is exact in every row and no row
is slower by more than 2%. A gain counts as a qualified relative result only with exact direct
parity.

**Result.** The decision rule failed as written, so the pilot did not adopt host metadata. The
method called for one container. The pilot ran in two, one per host class.

- **Removing the sync exposed a race on a FlashInfer buffer.** `plan()` writes its plan info into
  one pinned host buffer and queues an async copy to the device. The device-to-host read this entry
  removes had drained the GPU first, which made reusing that buffer safe. Without it, a decode
  window runs steps ahead, the next `plan()` overwrites the buffer before the copy runs, and a
  batch-64 kernel reads another step's plan and never finishes. Before each `plan()`, the engine
  waits on an event recorded after the previous one. That waits until the GPU reaches the copy,
  not until it finishes the step. Removing a hidden sync can hand its job to a race.
- **Parity is exact in all 18 rows,** with 0 non-tie divergences against the fp32 oracle.
- **Three rows were slower by more than 2%:** piecewise batch 8 (+2.4%, SXM4), piecewise batch 1
  (+3.7%, PCIe), and compiled batch 64 (+3.4%, PCIe). Their per-round walls spread 5%, 22%, and
  13% within one arm. A 2% band on a median of five rounds cannot resolve that, so the rule was
  underpowered for the piecewise and compiled arms.
- **Grouped, the HTTP-server default, was faster in all six rows** on both containers, with tight
  per-round walls.
- **The predictions were partly wrong.** Compiled batch 1 showed no 5–12% gain, so entry 17's
  host-bound model did not transfer. Two batch-64 gains exceeded the predicted 3%.

| Config | Batch | A100-SXM4 (400 W) | A100 PCIe (300 W) |
| --- | ---: | ---: | ---: |
| Piecewise | 1 | −0.7% | +3.7% |
| Piecewise | 8 | +2.4% | −3.8% |
| Piecewise | 64 | +0.4% | −8.7% |
| Grouped | 1 | −2.4% | −3.1% |
| Grouped | 8 | −3.2% | −1.9% |
| Grouped | 64 | −11.4% | −3.6% |
| Compiled | 1 | −1.0% | −2.0% |
| Compiled | 8 | −0.0% | +0.7% |
| Compiled | 64 | −0.3% | +3.4% |

(Host against device metadata, median ms per step over five rounds; negative is faster.)

**Exploratory, not a result:** paired by round, with a bootstrap interval
(`llm_infer/benchmarks/paired_change.py`), all three failing rows have medians between −1.2% and
+0.4% and intervals that include zero. All six grouped intervals sit below zero. These two runs are
the pilot for the confirmatory test below. No speed claim is made from them.

**Confirmatory run (registered before the run).** Two A100 containers, the same method, 11
measured rounds per row that alternate arm order. For each row and container, host and device are
paired by round: `d_i = (host_i − device_i) / median(device)`, in percent. The statistic is the
median of `d`, and its 95% interval is a percentile bootstrap over the 11 pairs (10,000
resamples, seed 0), from `llm_infer/benchmarks/paired_change.py`. `scripts/judge_paired_ab.py`
applies the rule to both records, so no row is judged by eye.

- **Merge gate:** parity is exact in every row on both containers, and no row regresses. A row
  regresses if its interval's lower bound is above 0 and its median is above +2%. A row whose
  interval half-width exceeds 3% is reported as inconclusive. It cannot pass the gate on its own,
  but it still must not regress.
- **Relative result:** a grouped row is a qualified gain only if its interval's upper bound is
  below 0 on both containers. The headline is the smaller of the two medians, with both
  reported. Piecewise and compiled rows are reported, not claimed.
- **No third container:** if the two containers disagree on a grouped row's sign, that row is not
  confirmed, and it is not rerun.
- **Predictions:** grouped batch 1 and 8 gain 2–4%, and grouped batch 64 gains 3–12%, depending on
  host speed. Every piecewise and compiled interval includes 0.

**Confirmatory result.** The gate passed, and host page metadata became the default. Two A100 PCIe (300 W)
containers, 11 rounds per row, judged by `scripts/judge_paired_ab.py`. Parity is exact in all 18
rows, no row regresses, and no interval is wide enough to count as inconclusive.

| Config | Batch | Container 1 | Container 2 | Verdict |
| --- | ---: | ---: | ---: | --- |
| Grouped | 1 | −2.9% [−3.1, −2.8] | −2.5% [−2.6, −2.5] | Qualified gain, **−2.5%** |
| Grouped | 8 | −1.8% [−2.2, −1.7] | −1.6% [−1.6, −1.6] | Qualified gain, **−1.6%** |
| Grouped | 64 | −4.6% [−4.7, −4.5] | −3.8% [−3.8, −3.7] | Qualified gain, **−3.8%** |
| Piecewise | 1 | −3.8% [−3.9, −3.7] | −3.7% [−3.8, −3.4] | Reported |
| Piecewise | 8 | −2.6% [−2.7, −2.6] | −2.4% [−2.9, −2.2] | Reported |
| Piecewise | 64 | −7.7% [−7.8, −7.5] | −7.1% [−7.2, −7.0] | Reported |
| Compiled | 1 | −0.6% [−0.9, −0.4] | −0.5% [−0.7, −0.4] | Reported |
| Compiled | 8 | −0.8% [−1.8, −0.4] | −0.5% [−0.7, −0.1] | Reported |
| Compiled | 64 | −0.5% [−0.6, −0.3] | −0.3% [−0.7, −0.1] | Reported |

(Median paired change in ms per step, host against device metadata, with 95% bootstrap intervals;
negative is faster.)

- **Grouped graphs, the HTTP-server default, gain 1.6–3.8%,** a qualified relative result.
- **Two of three grouped predictions held.** Batch 1 (2.5%) and batch 64 (3.8%) fall in their
  predicted ranges. Batch 8 (1.6%) falls just below 2–4%.
- **"Every piecewise and compiled interval includes 0" was wrong, in the favorable direction.**
  Every interval sits below 0, and piecewise gains 2.4–7.7%. Under the registered rule these rows
  are reported, not claimed.
- **Eleven paired rounds fixed the pilot's noise problem.** The widest interval here is 1.4
  points, against pilot intervals of up to 28 points from five rounds.
- **Limit:** both containers landed on the same host class (Intel family 6, model 106). The
  result does not cover other host classes.
- **Known bias, in the claim's favor:** only the `device` arm runs the wrapper that strips the
  host copies. It adds one or two `dataclasses.replace` calls per step, an estimated 0.1–0.2% of
  a step (not measured). That is well below the smallest claimed gain of 1.6%.

Records: [`assets/esme-decode-host-plan-ab-confirm-1.json`](../assets/esme-decode-host-plan-ab-confirm-1.json),
[`assets/esme-decode-host-plan-ab-confirm-2.json`](../assets/esme-decode-host-plan-ab-confirm-2.json).

Records (pilot): [`assets/esme-decode-host-plan-ab-pilot-sxm4.json`](../assets/esme-decode-host-plan-ab-pilot-sxm4.json),
[`assets/esme-decode-host-plan-ab-pilot-pcie.json`](../assets/esme-decode-host-plan-ab-pilot-pcie.json).

Reproduce: `modal run scripts/modal_esme_decode_profile.py --command host-plan-ab --batch-sizes
1,8,64`.

## 20. Measured bf16 logit error (predictions registered before the run)

**Question.** Entry 11 calls the 0.1-logit bound an audited acceptance boundary, not a measured
percentile: the headline pool has 8 unique prompts. What is the measured distribution of bf16
logit error, and how often does it exceed 0.1?

**Method.** 1,024 sampled prompts (1,022 unique texts: two Dolly instructions appear twice): a
seeded sample of short, context-free instructions from
`databricks/databricks-dolly-15k` at a pinned revision. The engine decodes them in batches of 64,
128 new tokens each, EOS ignored, on its default bf16 path: FlashInfer paged decode, piecewise
graphs, and decode windows. A recorder keeps every decode step's logits. The fp32 oracle and the
bf16 model then recompute logits over exactly the contexts the engine saw, so every comparison is
numerical error on one context, not a different continuation. That gives about 130,000 positions
per comparison. Three comparisons split the error: engine against fp32, bf16 full recompute
against fp32, and engine against bf16 recompute.

The main statistic is the **competitor shrink**: the largest gain of any of the fp32 top-8
competitors over the fp32 winner, `max(err[c] − err[a])`. The engine flips to `c` when the shrink
exceeds the fp32 gap `f[a] − f[c]`. A tie tolerance has to bound this quantity. A CPU smoke on 4
prompts showed one more fact before the run: the engine stores logits in bf16, whose spacing at
logit values of 16–64 is 0.125–0.25.

**Predictions (written before the run).**

- **The tail passes 0.1.** Engine-vs-fp32 shrink has a median of 0.03–0.08, and its 99.9th
  percentile is above 0.1. Entry 11 already saw single divergences at 0.108–0.146.
- **The error is whole-model bf16, not the cached path.** Bf16 recompute against fp32 has a
  99th-percentile shrink within 20% of the engine's. Entry 11 drew this conclusion from a few
  rows; here it is a distribution.
- **Flips stay near ties.** Some flips have an fp32 gap above 0.1, every flip has a gap below 0.3,
  and no flip occurs where the fp32 margin is 0.3 or more.
- **Output rounding matters.** The median bf16 spacing at the top logit is at least 0.0625.

**Decision rule (set before the run).** If the 99.9th-percentile shrink is at most 0.1 and no flip
has a gap above 0.1, the bound becomes a measured percentile. Otherwise the bound stays as it is,
and a new tolerance is a policy decision, made with these counts.

**Result.** The decision rule keeps the 0.1 bound as it is: the 99.9th-percentile shrink is 0.25,
and 7 flips have an fp32 gap above 0.1. Three of four predictions held. 130,048 positions per
comparison on one A100-SXM4, with the row-to-request mapping checked on every batch. Each request
contributes its 127 decode steps. The first generated token comes from prefill, and the recorder
does not see it, so no claim below covers that token.

| Statistic (logits) | Engine vs fp32 | bf16 recompute vs fp32 | Engine vs bf16 recompute |
| --- | ---: | ---: | ---: |
| Competitor shrink, median | 0.039 | 0.039 | 0.039 |
| Competitor shrink, p99 | 0.160 | 0.161 | 0.156 |
| Competitor shrink, p99.9 | 0.250 | 0.248 | 0.234 |
| Competitor shrink, max | 0.636 | 0.574 | 0.606 |
| Positions with shrink above 0.1 | 7.3% | 7.2% | 7.5% |
| Flips | 807 | 767 | 867 |
| Flips with a reference gap above 0.1 | 7 | 13 | 12 |
| Largest flip gap | 0.135 | 0.245 | 0.156 |

In the third column the reference is the bf16 recompute, so its margins and gaps are measured
against that, not fp32. Its flip count may also include exact top-two ties in the bf16 reference,
counted as flips with a gap of 0. The fp32 columns are unaffected, and so is every count above
0.1.

Engine flips by the fp32 top-two margin:

| fp32 margin | Positions | Flips | Flip rate |
| --- | ---: | ---: | ---: |
| below 0.01 | 630 | 255 | 40.5% |
| 0.01–0.03 | 1,177 | 330 | 28.0% |
| 0.03–0.1 | 4,041 | 215 | 5.3% |
| 0.1–0.3 | 10,471 | 7 | 0.07% |
| 0.3 or more | 113,729 | 0 | 0 |

- **The tail passes 0.1 (held).** The median shrink is 0.039, and 7.3% of positions move a
  competitor by more than 0.1 relative to the winner. A flip needs the fp32 margin to be small as
  well, which is why only 7 of 807 flips sit above the bound.
- **The cached path does not make the error larger (held for the distribution).** Bf16 full
  recompute against fp32 matches the engine's distribution to within 1% at the 99th percentile.
  So paged KV, FlashInfer, and CUDA graphs do not widen the error that the tie rule must cover.
  They do change individual logits: engine against bf16 recompute shows an error as large as
  either against fp32. A different rounding order would do that, but this run has no control
  that separates a rounding-order effect from a small error of the cached path's own. That
  attribution stays open.
- **Flips stay near ties (held).** No flip has an fp32 gap above 0.25, and no flip occurs where
  the fp32 margin is 0.3 or more, across 113,729 such positions.
- **Output rounding does not set the floor (failed).** The median bf16 spacing at the top logit is
  0.0078, because top logits are mostly between 1 and 2. The 0.25 differences in the CPU smoke
  came from rare large logits. The error builds up through the model's 30 bf16 layers, not in the
  last rounding step.
- **Divergence from fp32 greedy is normal, not rare.** 544 of the 1,024 prompts flip at least once
  in their 127 measured decode steps. Almost every such flip sits inside the 0.1 tie band, so
  policy v2 accepts it. At most 7 prompts in 1,024 (0.7%) would reach `review_required` from a
  decode step; the prefill token is not measured.

**Decision.** The automatic bound stays at 0.1 logits. The two options considered were
keeping 0.1 or raising it to 0.25, the measured 99.9th-percentile shrink. A gate at 0.25 sits at
the ceiling of normal bf16 error, so it could not tell that error from a kernel bug that moves
logits by 0.2. Keeping 0.1 costs little: at most 0.7% of requests reach `review_required` from a
decode step, and every observed case was a near-tie (engine fp32 gap at most 0.135).

This entry is the precedent for those reviews. A `review_required` divergence is a known bf16
near-tie, and its review may cite this entry instead of a new analysis, when all three hold:

1. The fp32 gap at the first divergence is at most 0.25. No flip occurred at a margin of 0.3 or
   more across 113,729 positions.
2. The change under test did not cause it. An A/B row needs exact direct candidate/baseline
   parity. A standalone row needs the `torch_naive` reference backend to choose the same token as
   the fast path, the check entry 11 used for `esme-001`.
3. The divergence is at a decode step. This run never saw the prefill token, so a prefill
   divergence gets a full review.

The row stays `review_required` and gets no public tok/s until the review is written. The harness
keeps one threshold (`BF16_AGREEMENT_TOLERANCE = 0.1` in `llm_infer/benchmarks/esme_three_way.py`).
The envelope comes from 1,024 short Dolly prompts with the Esme chat bundle on A100. Another
bundle, a long-context workload, or a gap above 0.25 is outside it and gets a full review.

Record: [`assets/esme-bf16-error.json`](../assets/esme-bf16-error.json). It stores the Dolly line
indices and a hash, not the prompt text.

Reproduce: `modal run scripts/modal_esme_bf16_error.py`. A CPU wiring check runs with
`--command smoke`.

## 21. Full-step decode graphs (predictions registered before the run)

**Question.** The HTTP server's CUDA default groups the first four layers, with their KV writes
and FlashInfer attention, into one graph per exact batch. The other 26 layers replay piecewise
graphs, with KV writes and attention running eagerly between them. In July, grouping 4 layers
beat grouping 2 by about twice the margin, and nobody tried all of them. Does grouping all 30
layers, so that a token step is one graph launch plus its `plan()`, beat the four-layer default?

**Method.** `scripts/modal_esme_full_step_ab.py`. One persistent engine per arm and batch size:
`grouped4` (the server default) and `full_step` (`grouped_layers=30`). Batch 1, 8, and 64, fixed
128-token greedy output with EOS ignored, two warmup rounds, and 11 measured rounds that
alternate which arm goes first. Every arm gates against the fp32 oracle, and the record stores
direct `grouped4`/`full_step` token parity. Two A100 containers. `scripts/judge_paired_ab.py`
applies the rule below, with entry 19's statistic: the median paired per-round change, with a
95% percentile bootstrap interval over the 11 pairs.

**Predictions (written before the run).** Entry 14 measured a batch-1 step as 91% GPU-busy, so
at most about 9% of it is idle time that a single graph could reclaim.

- **Parity:** exact at every batch. Capture changes when kernels launch, not what they compute.
- **Batch 1:** `full_step` is 2–8% faster.
- **Batch 8:** 1–6% faster.
- **Batch 64:** 0–4% faster. GPU time per step grows with batch, so the host gaps matter less.
- **Engine build:** `full_step` capture takes longer than `grouped4`, still within seconds.

**Decision rule (set before the run).** The gate passes when parity is exact in every row on both
containers and no row regresses: a row regresses if its interval's lower bound is above 0 and its
median is above +2%. A row is a qualified gain only if its interval's upper bound is below 0 on
both containers, and the headline is the smaller of the two medians. If the gate passes and
batch 1 is a qualified gain, full-depth grouping becomes the HTTP-server default in a separate
change. Otherwise the default stays at four layers.

**Result.** The gate passed, and full-depth grouping is a qualified gain at every batch. Two A100
PCIe (300 W) containers, both on the same host class (AMD family 25, model 1). Parity is exact in
all 6 rows, with 0 non-tie divergences against the fp32 oracle in either arm.

| Batch | Container 1 | Container 2 | Verdict |
| ---: | ---: | ---: | --- |
| 1 | −3.7% [−3.7, −3.7] | −2.9% [−3.0, −2.9] | Qualified gain, **−2.9%** |
| 8 | −4.9% [−4.9, −4.8] | −4.6% [−4.7, −4.6] | Qualified gain, **−4.6%** |
| 64 | −4.4% [−4.5, −4.4] | −4.2% [−4.2, −4.0] | Qualified gain, **−4.2%** |

(Median paired change in ms per step, `full_step` against `grouped4`, with 95% bootstrap
intervals; negative is faster.)

- **Parity, batch 1, and batch 8 held.** The gains (2.9% and 4.6%) fall in their predicted ranges.
- **Batch 64 beat its prediction.** It gained 4.2% against a predicted 0–4%. Removing the 26 eager
  attention regions still pays when the GPU does more work per step.
- **Capture cost did not grow (failed).** Engine build took about 6 s per arm at every batch. The
  one larger number, 16–17 s for `grouped4` at batch 1, is the first engine built in each
  container, so it includes FlashInfer's first-use setup.
- **Limit:** both containers ran on one host class. Entry 18 found that host speed changes how
  much grouped graphs pay, so the gain on another host may differ.

**Decision.** By the registered rule, full-depth grouping becomes the HTTP-server default.
`build_app_from_runtime` in `llm_infer/serve.py` now groups every layer. The raw engine keeps four
layers, so benchmark records stay comparable.

Records: [`assets/esme-full-step-ab-1.json`](../assets/esme-full-step-ab-1.json),
[`assets/esme-full-step-ab-2.json`](../assets/esme-full-step-ab-2.json).

## 22. Prompt-lookup speculation on the CUDA engine (predictions registered before the run)

**Question.** Speculation is off on CUDA (entry 10). Its one positive row was a single
repetition-heavy request on the flash-attn backend. A speculative step verifies up to four
drafted tokens in one forward pass, but it gives up the decode window and its graph path. Does it
pay at batch 1 on the current engine, on ordinary prompts or on prompts whose answers copy their
input?

**Method.** `scripts/modal_esme_spec_decode_ab.py`. Two workloads of 64 distinct Dolly prompts each
(pinned revision, seed 1), after 4 warmup prompts: `open` (context-free instructions) and `grounded`
(summarization, extraction, and closed QA over a 200–1,200-character passage). Batch 1, 128 new
tokens, EOS ignored. Each prompt runs once on each arm, alternating which arm goes first, so each
prompt is one pair: `baseline` (the raw engine's default, with piecewise decode graphs) against
`speculative` (prompt lookup, up to 4 draft tokens from n-grams of up to 4). The statistic is the
median per-prompt change in request latency, with a 95% bootstrap interval (entry 19's method).
Every arm gates against the fp32 oracle, the record stores direct per-prompt parity, and
`scripts/judge_paired_ab.py` applies the rule. Two A100 containers.

**Predictions (written before the run).**

- **Parity is not exact.** A verify step computes several positions in one forward pass, a
  different reduction order than one-token decode, so some prompts flip at near-ties (entry 20).
  Both workloads report `review_required` direct parity, and no claim qualifies.
- **Open:** 1.0–1.4 tokens per verify step. Speculation is 20–60% slower per request, because it
  pays the lost window and graph path without accepting many drafts.
- **Grounded:** 1.5–2.5 tokens per verify step, but speculation is still 10–60% slower. At batch
  1, the graphed window step costs less than an eager verify step that emits two tokens.
- **Identical-stream subset (exploratory, registered here):** the median change over the prompts
  whose two arms emit identical tokens has the same sign as the full-workload median.

**Decision rule (set before the run).** Speculation becomes a CUDA default for a workload only if
that workload is a qualified gain: exact direct parity on both containers, and an interval upper
bound below 0 on both. Otherwise it stays off, and the timing is reported as raw. The
identical-stream subset is reported, never claimed.

**Result.** The gate failed, and speculation stays off on CUDA. Requests take about 4.1x as long
on the open workload and about 3.5x as long on the grounded workload. Two A100 PCIe (300 W)
containers ran on different host classes (AMD family 25 and Intel family 6, model 106).

| Workload | Container 1 | Container 2 | Tokens per verify step | Identical streams |
| --- | ---: | ---: | ---: | ---: |
| Open | +311.6% [+270.7, +332.0] | +315.9% [+263.1, +342.4] | 2.01 | 33 of 64 |
| Grounded | +248.9% [+208.5, +291.2] | +250.8% [+205.8, +276.2] | 2.24 | 32 of 64 |

(Median paired change in request latency, speculative against baseline, with 95% bootstrap
intervals; positive is slower.)

- **Parity is not exact (held).** About half the prompts' streams diverge between the arms, so
  neither workload can qualify. Both arms pass the fp32 gate, apart from one baseline row noted
  below.
- **Open prompts accept more drafts than predicted (failed).** The prediction was 1.0–1.4 tokens
  per verify step. The measured 2.01 comes from the model repeating its own output, which prompt
  lookup finds in the generated history.
- **Grounded acceptance held:** 2.24 tokens per verify step, inside 1.5–2.5.
- **Both slowdowns are far larger than predicted** (+249% and +312%, against 10–60%). The cost is
  the verify step, not the drafts. Dividing the median request time by the verify steps per
  request gives at most 50–77 ms per verify step, against 5–6 ms per generated token on the
  graphed baseline, prefill included. One eager verify step costs about ten graphed decode
  steps. Accepting two tokens per step cannot recover that.
- **The identical-stream subset agrees (held).** It shows +227% to +273%, the same sign as the
  full workloads.
- **Limit: the baseline is the raw engine's default.** It decodes through piecewise graphs. The
  HTTP server's CUDA default adds grouped graphs, which were as fast or faster at batch 1
  (entries 18 and 21). Against the server default the slowdown would be the same or larger, so the
  decision holds.
- **One baseline row is `review_required`.** At grounded prompt `esme-066`, step 63, the baseline
  chose a token 0.119 logits below the fp32 maximum in both containers. That is one of the rare
  cases above 0.1 that entry 20 measured. It changes no conclusion here.

**Decision.** Speculation stays off on CUDA. For it to compete at batch 1, the verify step needs
the same treatment as decode: a graphed, planned multi-token path. That is an open question.

Records: [`assets/esme-spec-decode-ab-1.json`](../assets/esme-spec-decode-ab-1.json),
[`assets/esme-spec-decode-ab-2.json`](../assets/esme-spec-decode-ab-2.json).

## Open questions

- Can a graphed, planned multi-token verify step make speculation pay on CUDA (entry 22)?
- Does entry 15's compiled slowdown on A100 PCIe hold without the per-step `plan()` sync? Entry 17
  saw the compiled arm faster on A100-SXM4.
- Does the grouped-graph gain track the host CPU (entry 18)? Benchmark snapshots record the CPU.
- Does the cached path add error of its own, or only a different rounding order (entry 20)? It
  needs a control, for example the engine against itself with a perturbed reduction order.
