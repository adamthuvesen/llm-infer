# Performance Roadmap

`llm-infer` is a small inference engine for learning how modern serving techniques work and what
they buy. The useful work is concrete: add one technique, measure it against the current engine,
find the next bottleneck, and keep the engine understandable. The project has already moved far
past its first baseline. It does not need to reproduce every feature in vLLM or SGLang to be
successful. State-of-the-art (SoTA) engines are still one of the best sources of ideas for faster
attention, batching, scheduling, memory use, and launch reduction. We should study those designs
and adapt the parts that fit this engine.

Phase 2, ragged batched prefill, is complete and enabled by default. The next useful decode work
is narrower than the old Phase 3 plan suggested. The current piecewise CUDA graphs already hold
the largest dense regions possible while attention stays eager. Any further reduction in graph
boundaries must include KV writes and attention, which first needs stable page metadata.

The priority order below guides the next experiment. Start each performance experiment from a
reference-checked baseline and compare it back to back on one GPU. Keep raw measurements, qualify
public tok/s separately, and use direct A/B parity to tell a candidate regression from numerical
behavior shared with the baseline.

## How to Choose Work

- Pick a measured bottleneck or a modern serving technique that teaches us something useful.
- Make the smallest experiment that can answer the question. Stop when the measurement answers it.
- Prefer changes that improve the shared engine path and leave clear fallbacks. Benchmark-only
  machinery has to earn its maintenance cost.
- Study vLLM, SGLang, TensorRT-LLM, and other SoTA engines for techniques we can test here. Choose
  work from the bottleneck and the engine's scale instead of copying a product checklist.
- Keep negative results when they prevent a repeat dead end. Revert code that adds complexity
  without improving the engine or producing reusable measurement tools.

## Priorities

| Phase | Work | Main result | Status |
| ---: | --- | --- | --- |
| 0 | Fix the measurement baseline | Batch-1, phase, serving, and comparison evidence | Complete; bottleneck selected |
| Gate | Clarify numerical contracts | Separate deterministic greedy from stochastic sampling | Complete |
| 2 | Batch prefill | Better TTFT and prompt throughput under burst admission | Complete; enabled by default |
| 1 | Stabilize direct-page metadata | Remove duplicate work and support a bounded graph experiment | Complete; kept by A100 A/B |
| 3 | Test attention-inclusive graph groups | Four-layer probe cuts decode wall about 22–23% | Candidate proven; serving validation next |
| 4 | Fuse measured hot operations | Fewer kernels and larger GEMMs | Pending |
| 5 | Optimize sampled and HTTP serving | Keep engine speed through the public API | Pending |

## Decision Rules

Apply these rules to every phase:

1. Deterministic greedy headline rows qualify against the fp32 reference before they report public
   speed. Raw timing is always retained. Exact or explicitly reviewed numerical
   candidate/baseline parity with equal token counts can support a same-run relative result even
   when both share a pending fp32 numerical review.
   Sampled
   rows use same-shape seeded replay, per-request RNG isolation, and teacher-forced distribution
   checks; cross-shape exact tokens are not required. Label sampled timing diagnostic until those
   checks exist for the workload.
2. Run A/B variants back to back in one container. Cross-container raw tok/s can vary by about
   20% on the current Modal setup.
3. Report prefill, decode, time to first token (TTFT), inter-token latency (ITL), and end-to-end
   wall time separately when the workload includes them.
4. Report startup, graph-capture, compilation, and KV-pool allocation costs outside steady-state
   timing.
5. A 10% same-run improvement is the normal bar for keeping added runtime complexity. A smaller
   win can stay when the change makes the engine simpler, supplies a reusable measurement tool,
   or enables the next bounded experiment. Label diagnostic paths as experimental instead of
   turning them into defaults.
6. Record token counts beside tok/s. Numerical ties and stochastic continuations can change output
   length and bias close comparisons.

## Starting Point

The current CUDA path combines paged KV, continuous batching, planned eight-token decode
windows, manual piecewise CUDA graphs, batched greedy sampling, and direct FlashInfer page
reads. The public A100-80GB curve reports:

| Concurrent requests | `llm_infer` tok/s | Sequential HF tok/s |
| ---: | ---: | ---: |
| 8 | 658.6 | 39.4 |
| 16 | 1,140.0 | 39.6 |
| 32 | 1,844.9 | 39.7 |
| 64 | 2,603.4 | 39.3 |
| 128 | 3,275.5 | 39.5 |
| 256 | 3,776.8 | 39.5 |

Every row has zero classified non-tie divergences. Peak KV use at batch 256 was about 1.5 GB,
compared with about 8 GB for a contiguous layout reserved to the 1,024-token context limit.
See [benchmark.md](../benchmark.md) and [benchmark-evidence.md](benchmark-evidence.md).

Recent performance work has already removed several sources of overhead:

- Batched greedy argmax and cached EOS tensors.
- Deferred stop checks with a pinned, non-blocking device-to-host flush.
- Decode-window plans that precompute write slots and packed read layouts.
- Multi-step scheduler passes.
- Cached RoPE rows and fp32 RMSNorm weights.
- Piecewise CUDA graphs for the dense layer work.
- Direct FlashInfer access to paged KV.

The older eager path launched about 2,100 kernels per generated-token step across the batch.
Planned buffers added 5-7%, then
manual CUDA graphs improved like-for-like throughput by 177%, 45%, and 16% at batch 8, 64,
and 256. See [esme-decode-overhead.md](esme-decode-overhead.md) and
[decode-graph-capture.md](decode-graph-capture.md).

The 2026-07-07 FlashInfer profile records about 1,207 ordinary launches and 217 graph
launches per eight-token scheduler pass at batch 8. That is about 151 ordinary launches and 27
graph launches per generated-token step across the batch. The counts stay nearly flat through
batch 64, which points to fixed overhead at low and medium batch sizes. Phase 0 turns this raw run
into durable evidence.

## Phase 0: Build the Measurement Baseline

### Goal

Make the next optimization choice from current FlashInfer measurements. The detailed public
profile predates direct paged decode, and the headline curve mixes model work with fresh engine
and KV-pool construction.

### Work

- [x] Add a curated summary of the latest FlashInfer launch and timing profile under
  `docs/internal/`.
- [x] Add batch 1 to the GPU benchmark and decode-profile harnesses.
- [x] Split persistent-engine steady state from engine construction and KV-pool zeroing.
- [x] Add phase timing for prefill, decode, page planning, attention, sampling, and window
  flushing.
- [x] Add prompt-length and context-length sweeps at 32, 256, and 768 cached tokens.
- [x] Add persistent-server workloads for greedy and sampled requests. Report TTFT, p50/p95
  ITL, request latency, queue time, and output tok/s.
- [x] Run current vLLM against the converted Esme checkpoint in a separate process on the same
  reserved GPU host. Apply the same fp32 reference check.
- [x] Correct the public HF repetition wording: only batch 8 has three HF measurements; batches
  16-256 have one.
- [x] Document the Esme bf16 tolerance of 0.1, why it differs from the generic 1e-3 fixture
  tolerance, and how many unique prompts exercise it.

### Measurement matrix

- Batch: 1, 8, 64, 256.
- Cached context: 32, 256, 768 tokens.
- Decode: 128 steady-state tokens after warmup.
- Runs: two warmups, then ten paired measurements in one A100 container.
- Record: median and p95 step time, tok/s, launch counts, H2D/D2H copies, synchronizations,
  graph capture time, graph memory, GPU busy time, and top host operations.

### Done when

- Batch-1 TTFT and ITL have a committed reference-checked record.
- Persistent and cold-engine costs are separate.
- The next code target accounts for at least 20% of the relevant wall time or has direct code
  evidence of duplicate work.
- The public benchmark method matches the committed JSON.

### Result

Phase 0 selects ragged batched prefill as the first performance experiment. Serial prefill clearly
clears the 20% decision bar at batches 64 and 256; page planning does not. The two numerical
findings change benchmark coverage and wording, not the production execution path:

1. The context-768 `esme-001` prompt differs between bf16 cached decode and full recompute at both
   batch 1 and batch 8. Add all-prompt single-request coverage and keep affected speed rows blank.
2. Seeded sampled output keeps independent RNG state, but batch-shaped bf16 logits can produce a
   different valid draw. Check same-shape replay and distributions instead of forcing serial
   decode to preserve one continuation.

See [phase-0-baseline.md](phase-0-baseline.md) for the measurements and timing caveats.

## Phase 1: Stabilize Native-Page Metadata

**Complete; keep the stable-buffer path.** The focused A100 A/B found a useful direct win as well
as the cleaner graph boundary: median decode wall fell 7.3% at batches 1 and 8 and 11.9% at batch
64. The old path built packed indices that FlashInfer did not use and recreated native-page
metadata tensors every token. The new path removes that work and supplies fixed page buffers for
the bounded attention-inclusive graph experiment.

### Hypothesis

Before this phase, the direct FlashInfer path paid for both the old packed-read plan and the
native-page plan. FlashInfer used only the page metadata, yet `DecodeWindowPlan.begin_step()` built
packed read indices. `_page_plan()` also created and uploaded `indptr`, page counts, and last-page
lengths every token. `wrapper.plan()` still runs before the 30 layer-level attention calls because
that metadata changes as the requests grow.

Relevant code:

- [`DecodeWindowPlan.begin_step()`](../../llm_infer/model/decode_plan.py)
- [`PretrainBundleModel._prepare_paged_decode()`](../../llm_infer/model/pretrain_bundle.py)
- [`FlashInferPagedAttention.plan_decode_batch_paged()`](../../llm_infer/kernels/flashinfer_paged.py)

### Work

- [x] Skip packed `idx` and cumulative-length construction when the backend consumes native pages.
- [x] Preallocate page `indptr`, indices, and last-page buffers for the whole decode
  window.
- [x] Copy each step's precomputed values into stable-address page buffers.
- [x] Keep FlashInfer planning per token. Page counts and last-page lengths change during a window,
  and the current wrapper has no safe plan-once contract for that metadata.
- [x] Measure the stable-buffer path against the previous planner on A100.
- [x] Update stale comments that claimed the planned path had no per-step H2D traffic.

### Correctness checks

- Compare old and new page plans across ragged lengths that cross page boundaries.
- Cover representative block sizes, decode windows, and low/high batches. Expand the matrix when a
  failure points to a missing boundary case.
- Exercise prefix copy-on-write, request completion, and batch changes.
- Run real Esme generation through the fp32/tie reference check before timing.

### A/B

Compare the current path with these variants in order:

1. Native-page path without packed `idx`.
2. Preallocated page buffers with per-token `wrapper.plan()`.
3. Plan-once-per-window or graph-safe fast planning, if supported.

Start with the Phase 3-relevant exact graph buckets rather than repeating the whole Phase 0 matrix.
Add `wrapper.plan` host time, page-plan device time, and metadata copy counts.

### Result

Measured 2026-07-10 on A100-80GB with context 256, 128 fixed output tokens, two warmup pairs, and
ten measured alternating legacy/stable pairs. Each mode used a persistent engine and piecewise
CUDA graphs captured before timing for exact buckets 1, 8, and 64.

| Batch | Legacy median / p95 | Stable median / p95 | Median change | Stable tok/s |
| --- | --- | --- | --- | --- |
| 1 | 1.084 s / 1.099 s | 1.005 s / 1.016 s | -7.3% | 126.4 |
| 8 | 1.505 s / 1.615 s | 1.395 s / 1.423 s | -7.3% | withheld |
| 64 | 1.924 s / 2.733 s | 1.695 s / 1.749 s | -11.9% | 4,795.0 |

The short two-step profile kept graph launches fixed at 496 and cut ordinary CUDA launches from
about 3,230 to 2,919, matching the removed per-token tensor work. The host timing bucket stayed
roughly flat at 47–58 ms per 127-step diagnostic because FlashInfer planning still dominates it;
that bucket is not the source of the measured gain.

Reference gate: batches 1 and 64 have zero non-tie divergences. At batch 8, both legacy and stable
produce the same continuation and the same single fp32 divergence at `esme-007` step 22. Its fp32
margin is 0.130 logits, outside the automatic 0.1 tie rule, so the harness correctly reports no
tok/s for that row. The identical old/new output shows the metadata change did not cause it, but
the table keeps the row as wall-time-only rather than weakening the gate.

Raw local evidence: `bench-results/esme-page-plan-ab-20260710T140557.json` (gitignored).

### Done when

- Page metadata matches the current implementation at every tested step.
- The experiment either removes the duplicate work, supplies fixed buffers for one bounded Phase 3
  probe, or shows clearly that the extra machinery is not worth keeping.
- The measured high batch does not regress; expand only if a later workload exposes a larger-batch
  concern.
- No timed run rebuilds or recaptures hidden state.

### Local result

Native-page decode now omits the packed physical-slot gather, packed cumulative-length update, and
their device buffers. Planned windows precompute each step's page metadata once, then copy it into
fixed-address buffers before the existing per-token FlashInfer plan. Packed attention and the
`torch.compile` experiment keep their old plan, including eager fallback from a compiled bucket.

CPU proof: page metadata matches the classic planner across ragged page growth, destination buffer
addresses stay fixed, the packed fallback stays exact, ruff is clean, and the full fast suite
passes. The A100 result above supplies the measured decision: keep the stable-buffer path.

## Phase 2: Add Ragged Batched Prefill

**Implementation complete; full A100 matrix measured; enabled by default.** Phase 0 measured serial
prefill above the 20% selection bar at high batch sizes.

### Hypothesis

New requests are prefilled one at a time. Each short prompt walks the full layer stack, so burst
admission repeats launches and runs small GEMMs. A packed ragged prefill should amortize layer
overhead and improve GPU occupancy. It should also avoid GQA `repeat_interleave`, since the
attention backend can consume fewer K/V heads than query heads.

Relevant code:

- [`EnginePrefillMixin._prefill_requests()`](../../llm_infer/serving/engine_prefill.py)
- [`PretrainBundleModel.prefill()`](../../llm_infer/model/pretrain_bundle.py)
- [`PagedKVCache`](../../llm_infer/kv_cache/paged_kv_cache.py)

### Work

- [x] Add a ragged batch-prefill contract to the model backend.
- [x] Pack prompt tokens and request boundaries without padding to the longest prompt.
- [x] Write each request's K/V into its own page table.
- [x] Use native GQA in the fast prefill attention path.
- [x] Keep sequential prefill as the reference and fallback.
- [x] Preserve whole-prompt and chunked-prefill scheduling policies.
- [x] Make mixed prefill/decode scheduling a separate experiment after isolated batched prefill
  passes.

### A100 results

The full matrix completed on 2026-07-10 on an A100-80GB with the resumable A/B harness. Raw
evidence: `bench-results/esme-prefill-ab-20260710T113123.json` and
`bench-results/esme-prefill-ab-rows.jsonl` — 24 rows, each cell 2 warmup plus 10 measured
alternating pairs. The packed path (`InferenceEngine(batched_prefill=True)`) is now the default;
serial prefill stays the reference and fallback.

Speedups (candidate vs. serial baseline):

- Batch 1: exact tokens with roughly 0% delta on all 8 rows — the serial fallback fires for a
  single request, so baseline and candidate run the same path.
- Batch 8: prefill/TTFT 8.9x to 9.5x across uniform 16/128/512 and the ragged mixture; end-to-end
  1.52x to 1.55x at 64 output tokens.
- Batch 64: TTFT 65.6x (ctx 16), 38.6x (ctx 128), 11.2x (ctx 512), and 24.9x (ragged); end-to-end
  3.65x to 5.32x at 64 output tokens.

All 1-token TTFT rows are exact for the candidate.

### Divergence review

Evidence: `bench-results/esme-prefill-divergence-20260710T112250.json`, `-113643.json`, and
`-113908.json`. Every 64-token divergence is a deterministic near-tie: the same step and token
pair recur across all repeats. The two rows the 0.1-logit rule flags — batch-8 uniform-128 at
step 5 and uniform-512 at step 53 — have fp32 top-2 gaps of 0.142 and 0.146, with engine-side
bf16 gaps of 0.125 and 0.031 (1 ulp at that magnitude). The serial baseline itself diverges from
the fp32 oracle on other requests with engine-side gaps up to 0.094, and one request diverges
identically in both modes (pure decode-path noise, unrelated to prefill packing). Packed-vs-serial
prefill K/V agrees within 1-2 bf16 ulp per layer (max abs diff <= 0.26, mean ~0.007). Decoded text
stays coherent on both paths.

Conclusion: these are numerical ties amplified by reduction order, not an implementation error.
Batched prefill is enabled by default with the serial path kept as fallback and reference. No
serial or fp32 fallback was added to force token identity.

### Mixed-load results

Measured 2026-07-10 on A100-80GB with the mixed-load A/B harness (`--command mixed-load`).
Raw evidence: `bench-results/esme-mixed-load-20260710T*.json` and
`bench-results/esme-mixed-load-rows.jsonl` — 8 steady ragged decoders at 256 output tokens with
`decode_window_size=1` and per-step synchronization, a burst admitted in one reserve-mode step
(worst case: the whole burst prefills in one call), 1 warmup plus 5 measured alternating pairs
per cell. Steady-state ITL baseline is ~30 ms per token in this per-step-synchronized protocol.

The decode-tail stall is the one inter-token gap that spans the burst step. Batched prefill
shrinks it 5x to 20x versus serial prefill of the same burst; burst TTFT moves identically
because both equal that step's wall time:

| Burst | Prompt tokens | Serial stall | Packed stall | Packed/serial |
| --- | --- | --- | --- | --- |
| 8 ragged | 1,434 | 309 ms | 61 ms | 0.20x |
| 32 ragged | 5,932 | 1,156 ms | 74 ms | 0.06x |
| 64 ragged | 12,778 | 2,291 ms | 117 ms | 0.05x |
| 64 uniform-512 | 32,768 | 2,446 ms | 241 ms | 0.10x |

Packed stall grows roughly linearly at ~6 ms per 1k packed prompt tokens over a ~55 ms base.
Correctness: one candidate divergence across all cells (1 of 40 requests, decode step 100, fp32
top-2 gap 0.136) — the same deterministic near-tie class as the divergence review above; the
serial baseline is itself tie-tolerant on every cell.

Decision: no packed-prefill size cap for now. Batched prefill strictly reduces the decode-tail
stall relative to the serial path at every measured burst size, including the ~32k-token worst
case (241 ms, ~8 normal token gaps, once per burst). If a future workload needs smoother tails,
the cap belongs in `_can_prefill_many`/`_prefill_many` (`llm_infer/serving/engine_prefill.py`)
and the table above is the data to size it; splitting a 32k burst into ~8k sub-batches would
trade burst TTFT for ~70 ms stall ceilings.

### Measurement matrix

- Batch: 1, 8, 64.
- Prompt lengths: uniform 16, 128, and 512; ragged mixture from 16 to 512.
- Output: 1 token for TTFT and 64 tokens for end-to-end time.
- Mixed load: active short decodes plus a burst of long prefills.
- Record: prefill tok/s, p50/p95 TTFT, launch counts, GPU busy time, peak KV use, decode-tail
  latency, and end-to-end wall time.

### Correctness checks

- First-token logits within the existing numeric tolerance and exact token IDs against
  sequential prefill.
- Layer-level K/V equality on mixed prompt lengths.
- Prefix sharing, partial-page copy-on-write, and chunk continuation.
- Full reference-checked generation after batched prefill.

### Done when

- Batch 8 or 64 improves prefill tok/s or TTFT by at least 10%.
- Batch 1 regresses by no more than 3%.
- Mixed prefill does not worsen decode-tail latency compared with the current whole-prompt path.
- Sequential fallback remains available for unsupported backends and debug runs.

## Phase 3: Test Attention-Inclusive Graph Groups

**Do not restart the old static full-graph experiment.** A static 32-slot Qwen graph was already
measured and rejected: it did extra work for inactive rows, ran slower, and changed the sampled
continuation. The current Esme runner avoids that mistake by keeping padding out of eager
attention.

### Hypothesis

The current runner replays 31 graph segments per generated-token step across the batch and enters
eager Python for every layer's KV write and FlashInfer attention call. Each middle segment already
combines one layer's post-attention work with the next layer's pre-attention work, so dense segments
cannot be combined further on their own. `cudaGraphLaunch` remains the top current host event.

After Phase 1 supplies fixed page metadata, a small engine-owned graph may be able to cover two to
four complete layers, including their KV writes and FlashInfer attention calls. Exact batch sizes
1 and 8 are the useful first test because they avoid padding work and target the measured fixed
overhead. Planning stays outside capture.

### Work

- [x] Prove that one fixed-batch FlashInfer wrapper can use Phase 1's stable metadata across
  repeated planned decode steps without hidden allocation or recapture.
- [x] Capture the smallest useful region: two to four complete layers tied to one engine cache.
- [x] Start at exact batches 1 and 8. Keep the existing piecewise runner as the reference and
  fallback.
- [x] Record graph launches, ordinary launches, steady-state latency, capture time, and graph
  memory for the experiment.
- [x] Expand the group size only when the smaller group improves the target row or clearly removes
  enough host work to justify another measurement.
- [ ] Measure padded batches 9, 17, 33, 65, 96, and 129 only after an exact-size group works. This
  separates graph-boundary savings from padding cost.
- [ ] Compare the useful bucket set with the server default, which stops at 16 and falls back to
  eager above that size.

### Result

The bounded probe passed on A100-80GB with FlashInfer 0.6.14. A graph-mode wrapper with
caller-owned page buffers replayed across lengths 63/64/65 and 127/128/129 at exact batches 1 and
8. Pointers stayed fixed, every result matched a freshly planned ordinary wrapper with zero error,
and every steady-state re-plan had zero allocated/reserved-byte delta. `plan()` remains outside
capture once per token; only `run()` is captured.

The engine-owned experiment then compared the current piecewise runner with exact-batch layer
groups at context 256 and 128 fixed output tokens, using two warmup and ten measured alternating
pairs. Capture stayed outside timing.

| Group | Batch | Piecewise median / p95 | Grouped median / p95 | Median change | Tok/s change |
| --- | ---: | --- | --- | ---: | ---: |
| 2 layers | 1 | 0.759 s / 0.769 s | 0.670 s / 0.672 s | -11.8% | +13.4% |
| 2 layers | 8 | 0.963 s / 0.972 s | 0.870 s / 0.870 s | -9.7% | withheld |
| 4 layers | 1 | 0.858 s / 0.866 s | 0.661 s / 0.665 s | -22.9% | +29.8% |
| 4 layers | 8 | 1.109 s / 1.113 s | 0.866 s / 0.869 s | -22.0% | +28.1% relative |

The four-layer graph moved graph launches from 31 to 27 and ordinary launches from about 182 to
158 per generated token. Its group capture took 0.27–0.47 seconds after the normal piecewise
capture. The experiment owns one 128 MiB FlashInfer workspace plus about 8.1 MiB of group-graph
memory and small metadata buffers per exact cache/batch runner—roughly 136 MiB total. That is fine
for the proof but too expensive to multiply across buckets without a serving design.

Policy-v2 follow-up: batch 1 has zero review-required differences. At batch 8, a correctness-only
rerun recorded every output for both modes. The candidate and baseline continuations are exactly
identical across all eight requests. Both choose token 13204 at `esme-007` step 22 while fp32
chooses 1616 with a 0.13044-logit margin. That row remains ineligible for an absolute headline,
but direct parity qualifies the existing same-run 22.0% wall reduction / 28.1% throughput ratio as
an optimization result. No timing matrix was repeated.

Raw local evidence (gitignored):

- `bench-results/esme-decode-flashinfer-graph-probe-20260710T143410.json`
- `bench-results/esme-decode-two-layer-group-ab-20260710T150034.json`
- `bench-results/esme-decode-four-layer-group-ab-20260710T150753.json`
- `bench-results/esme-decode-four-layer-parity-20260710T154124.json` (correctness only)

Decision: keep the grouped runner as benchmark-only candidate code. Do not enable it by default
until cache ownership, EOS/abort/window lifecycle, useful bucket count, and batch 64/256 regression
checks pass. Do not expand beyond four layers before those serving questions are answered; the
four-layer result already clears the target and larger groups would increase capture coupling.

### Correctness checks

- Exact-size eager versus grouped logits and tokens on the tiny bundle.
- Real Esme generation through the fp32 oracle and documented-tie rule.
- A fresh engine and cache, page growth, batch changes, and eager fallback.
- EOS, length caps, aborts, and window flushing on the grouped path.
- Proof that no graph capture occurs inside a timed region.

### Stop when

- FlashInfer needs timed recapture, changing tensor addresses, or per-step wrapper construction.
- The exact batch-1 and batch-8 experiment cannot materially reduce graph launches or latency.
- Per-engine graph memory or cache ownership makes the serving path harder to understand than the
  result is worth.
- A candidate-only token divergence or structural output failure appears.

### Done when

The experiment answers whether short attention-inclusive graph groups help this engine. The
benchmark-only four-layer candidate clears the low-batch bar: graph launches fall from 31 to 27
per generated token and batch 1 improves by more than 10%. Keep it as a default only after batch 64
and 256 regress by no more than 3% and the serving lifecycle checks above pass. Record a negative
result and remove the candidate if those remaining checks miss the bar.

## Phase 4: Fuse Measured Hot Operations

Start with the current profile and test each fusion independently. Phase 4 can move ahead of Phase
3 when stable metadata or attention capture costs more complexity than the likely gain. Keep the
unfused path as the reference.

Recommended order:

1. [ ] Pack Q/K/V projection weights into one GEMM.
2. [ ] Pack SwiGLU gate/up projection weights into one GEMM.
3. [ ] Fuse residual add with RMSNorm.
4. [ ] Consider fused QK-norm, RoPE, and KV append if those operations still account for enough
   wall time.

For each change:

- Measure kernel and graph launch counts as well as wall time.
- Compare component tensors in fp32 before full generation.
- Run the full greedy reference check and audit every changed tie margin.
- Keep added runtime complexity when it clears the 10% bar on its target workload. A smaller change
  can stay when it simplifies the model path or supplies a shared primitive used elsewhere.

## Phase 5: Optimize Sampled and HTTP Serving

### Goal

Keep model speed through the public API. The current headline covers all-greedy engine runs.
Sampled requests disable planned decode windows and CUDA graphs, while HTTP streaming adds
per-token tensor conversion, cross-thread callbacks, and full-history detokenization.

### Work

- [ ] Add an A100 sampled-decode profile before changing the sampler.
- [ ] Batch compatible greedy, top-k, and top-p rows on the GPU.
- [ ] Preserve one seeded generator and batch-independent output per request.
- [ ] Keep planned decode and CUDA graphs active for stable sampled batches when safe.
- [ ] Pass staged host token IDs to the async dispatcher instead of converting CUDA tensor views
  again.
- [ ] Dispatch one token burst per request/window instead of one callback per token.
- [ ] Replace full-history detokenization with a tokenizer-native incremental decoder or a
  bounded suffix strategy.
- [ ] Overlap CPU result handling with the next GPU step if Phase 0 shows host gaps.

### Correctness checks

- Same-shape seeded replay is reproducible, and each request keeps independent RNG state when it
  runs with batchmates.
- Teacher-forced distribution checks cover batch-shape numerical differences. Cross-shape exact
  sampled tokens are not required.
- Seeds, penalties, top-k, top-p, and mixed sampling parameters keep their current semantics.
- Streaming text, UTF-8 handling, stop strings, usage counts, and final responses stay identical.
- Client disconnects still abort work and free KV.

### Done when

- Sampled serving has its own reference and benchmark record.
- HTTP p50/p95 TTFT and ITL improve by at least 10% on the target workload.
- Engine-only greedy throughput regresses by no more than 3%.

## Work to Defer

Keep these ideas off the main path until a new measurement changes the bottleneck model:

- Generic `torch.compile` decode. The current path matched the launch reduction but lost every
  wall-clock comparison and took 407 seconds to compile and warm. Re-run only after a meaningful
  PyTorch or Inductor change. See [torch-compile-decode.md](torch-compile-decode.md).
- More chunked prefill. The current Esme workload became about 21% slower and did not improve the
  intended stall metric. Packed batched prefill now handles the measured burst workload; revisit
  chunking when a real workload needs a lower stall ceiling.
- Larger decode windows. Stop-sync removal alone added only 0-3%, while larger windows delay
  visible tokens and waste more work after EOS.
- Default speculative decoding. The positive result covers one repetition-heavy batch-1
  workload and predates the current FlashInfer baseline.
- Weight or KV quantization. The model fits easily, the 1,024-token context keeps KV use modest,
  and quantized kernels can add more overhead than they remove at small batch sizes.
- Multi-GPU serving, offload, and prefill/decode disaggregation. They add communication for a
  model that already fits on one GPU.
- Another attention backend without a profile. Decode already reads paged KV through FlashInfer,
  and attention has not been isolated as the current wall.
- More cache hierarchy. Paging already saves capacity; KV bandwidth has not been shown to limit
  performance.
- Feature parity with vLLM or SGLang. Studying and borrowing their speed techniques stays in scope.
  Their full product surface and scheduler complexity do not define this repo's roadmap.

## Open Questions

- What dominates current batch-1 TTFT and ITL?
- Can FlashInfer reuse a fixed plan for a whole decode window, or only fixed metadata buffers?
- Can a fixed-batch FlashInfer wrapper be captured across a short group of complete layers?
- How large is the performance cliff above the server's batch-16 graph limit?
- What logit-error distribution supports the 0.1 bf16 tolerance?

## References

- [vLLM CUDA graph design](https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/)
- [SGLang FlashInfer backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py)
- [TensorRT-LLM attention](https://nvidia.github.io/TensorRT-LLM/features/attention.html)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [PagedAttention](https://arxiv.org/abs/2309.06180)
- [Speculative decoding](https://arxiv.org/abs/2211.17192)
