# Research Log

This log records what the `llm-infer` experiments asked, measured, and decided. It keeps the
negative results, because they explain the current design as much as the wins do.

Each entry has four parts: the question, the method, the result, and the decision. Numbers come
from Modal A100-80GB runs of `Esme-214M-Chat` unless an entry says otherwise. Raw JSON records
live in the gitignored `bench-results/` directory. Each entry names its record so a maintainer can
find it. The committed public curve is [`assets/esme-batch-curve.json`](../assets/esme-batch-curve.json).

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
| 8 | Is op fusion worth it now? | GPU kernels are about 1–2 ms of a 5.5 ms step | Declined |
| 9 | What is the right sampled-output contract? | Same-shape seeded replay, not cross-shape exact tokens | Contract changed; sampled serving 3–5x faster |
| 10 | Do chunked prefill, prefix caching, preemption, and speculation work? | Mixed; see the entry | Defaults follow the measurements |
| 11 | Is the 0.1-logit bf16 bound measured? | No: it is an audited acceptance bound | Open; listed below |
| 12 | How far is the engine from vLLM? | 3.1–3.2x at batch 1, 6.6–13.7x at batch 64/256 (before entries 6–7) | Rerun prepared |

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

**Method.** One reserved A100-80GB PCIe host. The fp32 oracle, `llm_infer`, and vLLM 0.24.0 ran as
sequential child processes, with matching hostname and GPU UUID checked. Each row generated 128
tokens per request (EOS ignored), with two warmups and the median of ten runs. All 24 rows passed
the fp32 gate with zero non-tie divergences.

**Result** (2026-07-09):

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

**Limits.**

- This run predates batched prefill (entry 6) and grouped graphs (entry 7). At that time, serial
  prefill took 67–84% of `llm_infer` wall at batch 64–256, so the high-batch gap is the stale part
  of this table.
- The two systems used separate dependency stacks: Torch 2.8 / CUDA 12.8 against Torch 2.11 /
  CUDA 13.0.
- Each context row repeats one synthetic prompt, so this workload does not test prompt diversity.

**Decision.** The batch-1 gap (about 3.2x) is the cleanest open question: it is not a batching
effect. A rerun against the current engine is prepared in
[`scripts/modal_esme_vllm_baseline.py`](../scripts/modal_esme_vllm_baseline.py), with vLLM pinned
to 0.24.0 and a grouped-graph `llm_infer` arm.

Record: `esme-vllm-baseline-full-20260709T235733`.

## Open questions

- What explains the batch-1 gap to vLLM? A 214M bf16 model reads about 0.43 GB of weights per
  step, so A100 bandwidth allows several thousand steps per second. The engine runs about 150. The
  remaining host and launch cost is not yet split.
- What logit-error distribution supports the 0.1 bf16 bound? This needs many more unique prompts
  than the 8 in the headline pool.
- How much did each technique contribute on one GPU? Entries 2–7 ran in different containers, so
  their gains cannot be added. A same-container ablation ladder is prepared as
  `modal run scripts/modal_esme_batch_curve.py --command ladder`.
