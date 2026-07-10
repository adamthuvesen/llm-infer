# Performance Roadmap

The next performance work should batch the engine's repeated per-request prefills. Current A100
measurements attribute 24% of profiled wall at batch 8 and roughly 67–85% at batches 64–256 to
serial prefill. Page planning is duplicated work, but its measured share is only about 2–3% of
decode time. Esme-214M is still small enough that kernel launches, metadata preparation, and
Python can cost as much as useful GPU work, especially at batch 1.

Use the measured priority order below; phase numbers preserve the original plan. Each phase
starts with a reference-checked baseline and ends with a same-container A/B. A path that fails
the reference check reports no tok/s.

## Priorities

| Phase | Work | Main result | Status |
| ---: | --- | --- | --- |
| 0 | Fix the measurement baseline | Batch-1, phase, serving, and mature-engine evidence | Complete; bottleneck selected |
| Gate | Clarify numerical contracts | Separate deterministic greedy from stochastic sampling | Complete |
| 2 | Batch prefill | Better TTFT and prompt throughput under burst admission | Built; A/B checkpoint |
| 1 | Remove direct-page planning overhead | Less metadata work on every decode token | Deprioritized |
| 3 | Reduce CUDA graph boundaries | Lower batch-1 and low-batch decode latency | Pending |
| 4 | Fuse measured hot operations | Fewer kernels and larger GEMMs | Pending |
| 5 | Optimize sampled and HTTP serving | Keep engine speed through the public API | Pending |

## Decision Rules

Apply these rules to every phase:

1. Deterministic greedy rows match the fp32 reference before they report speed. Exact tokens are
   required unless the first divergence is a traced bf16 tie under the documented rule. Sampled
   rows use same-shape seeded replay, per-request RNG isolation, and teacher-forced distribution
   checks; cross-shape exact tokens are not required. Label sampled timing diagnostic until those
   checks exist for the workload.
2. Run A/B variants back to back in one container. Cross-container raw tok/s can vary by about
   20% on the current Modal setup.
3. Report prefill, decode, time to first token (TTFT), inter-token latency (ITL), and end-to-end
   wall time separately when the workload includes them.
4. Report startup, graph-capture, compilation, and KV-pool allocation costs outside steady-state
   timing.
5. Keep a change only when it clears the phase's success criteria. The normal rollback bar is a
   10% same-run improvement on the workload the change targets.
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

## Phase 1: Remove Native-Page Planning Overhead

**Priority after Phase 2.** Phase 0 measured page planning at about 2–3% of nested decode time.
Keep this phase because the packed-index work is demonstrably duplicated and stable metadata may
unlock wider graph capture, not because current timing predicts a large direct win.

### Hypothesis

The direct FlashInfer path pays for both the old packed-read plan and the new native-page plan.
FlashInfer uses only the page metadata, yet `DecodeWindowPlan.begin_step()` still builds packed
read indices. `_page_plan()` also creates and uploads `indptr`, page counts, and last-page
lengths every token, then `wrapper.plan()` runs before the 30 layer-level attention calls.

Relevant code:

- [`DecodeWindowPlan.begin_step()`](../../llm_infer/model/decode_plan.py)
- [`PretrainBundleModel._prepare_paged_decode()`](../../llm_infer/model/pretrain_bundle.py)
- [`FlashInferPagedAttention.plan_decode_batch_paged()`](../../llm_infer/kernels/flashinfer_paged.py)

### Work

- [ ] Skip packed `idx` construction when the backend consumes native pages.
- [ ] Preallocate page `indptr`, indices, counts, and last-page buffers for the whole decode
  window.
- [ ] Update only fields that change as each request grows.
- [ ] Test whether FlashInfer planning can move from every token to every window.
- [ ] If planning must stay per token, use graph-safe fixed buffers and measure a smaller fast
  planning path.
- [ ] Update stale comments that claim the planned path has no per-step H2D traffic.

### Correctness checks

- Compare old and new page plans across random ragged lengths.
- Cover every page boundary for block sizes 16, 32, 64, and 128.
- Cover decode windows of 1, 8, and 16 tokens at batches 1-256.
- Exercise prefix copy-on-write, request completion, and batch changes.
- Run real Esme generation through the fp32/tie reference check before timing.

### A/B

Compare the current path with these variants in order:

1. Native-page path without packed `idx`.
2. Preallocated page buffers with per-token `wrapper.plan()`.
3. Plan-once-per-window or graph-safe fast planning, if supported.

Use the Phase 0 batch/context matrix. Add `wrapper.plan` host time, page-plan device time, and
metadata copy counts.

### Done when

- Page metadata matches the current implementation at every tested step.
- The winning variant improves batch 1 or 8 by at least 10% on the same GPU, or improves the
  chosen workload-weighted result by the same bar.
- Batch 64 and 256 regress by no more than 3%.
- No timed run rebuilds or recaptures hidden state.

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
- [ ] Make mixed prefill/decode scheduling a separate experiment after isolated batched prefill
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

## Phase 3: Reduce CUDA Graph Boundaries

### Hypothesis

The current runner replays 31 graph segments per generated-token step across the batch and enters
eager Python for every layer's
KV write and FlashInfer attention call. `cudaGraphLaunch` remains the top current host event.
Stable page metadata from Phase 1 may allow a full decode graph or larger captured regions.

### Work

- [ ] Prototype an engine-owned full decode graph tied to one cache address.
- [ ] Keep FlashInfer planning outside capture and feed captured kernels through stable metadata
  buffers.
- [ ] If full capture is unsafe, combine several layer segments before returning to Python.
- [ ] Measure graph buckets at real batch sizes: 9, 17, 33, 65, 96, and 129.
- [ ] Compare the benchmark bucket set with the server default, which currently stops at 16 and
  falls back to eager above that size.
- [ ] Record capture time and graph memory for every bucket set.

### Correctness checks

- Different engine and cache instances.
- Batch changes, bucket padding, and eager fallback.
- Page growth, prefix sharing, and copy-on-write.
- EOS, length caps, aborts, and window flushing.
- Proof that no graph capture occurs inside a timed region.

### Done when

- Graph launches fall materially below the current roughly 27 per generated-token step across
  the batch.
- Batch 1 or 8 improves by at least 10%.
- Batch 64 and 256 regress by no more than 3%.
- Capture time and graph memory are bounded and reported.

## Phase 4: Fuse Measured Hot Operations

Start this phase only after the current profile identifies the remaining GPU work. Test each
fusion independently and keep the unfused path as the reference.

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
- Keep the change only if it clears the 10% bar on its target workload or removes enough work to
  unlock Phase 3 capture.

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

- Sampled output remains identical when a request runs alone or with batchmates.
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
  intended stall metric. Batch prefill first.
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

## Open Questions

- What dominates current batch-1 TTFT and ITL?
- Can FlashInfer reuse a fixed plan for a whole decode window, or only fixed metadata buffers?
- How much of the public wall time is fresh KV-pool allocation?
- How large is the performance cliff above the server's batch-16 graph limit?
- What logit-error distribution supports the 0.1 bf16 tolerance?
- Can current vLLM or SGLang load the converted checkpoint and pass the same fp32 reference check?

## References

- [vLLM CUDA graph design](https://docs.vllm.ai/en/v0.21.0/design/cuda_graphs/)
- [SGLang FlashInfer backend](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/flashinfer_backend.py)
- [TensorRT-LLM attention](https://nvidia.github.io/TensorRT-LLM/features/attention.html)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [PagedAttention](https://arxiv.org/abs/2309.06180)
- [Speculative decoding](https://arxiv.org/abs/2211.17192)
