# Esme Decode: Sync Sweep + Piecewise CUDA Graph Capture (2026-07-02)

Local `main`, base `bb793f5`. Follow-up to
[esme-decode-overhead.md](esme-decode-overhead.md), which left the decode wall at
kernel-launch dispatch: ~1,850 tiny kernels per token-step, roughly half CPU dispatch and
half GPU launch-overhead execution. This pass removes that wall the way vLLM/SGLang do:
bucket-padded CUDA graph capture over the planned decode window, after first making the
decode loop sync-free. Buckets are captured once and never recaptured for a new batch shape.

**Verdict: shipped.** Captured decode beats the eager window like-for-like by +177% /
+45% / +16% at batch 8 / 64 / 256, far past the 10% rollback bar, with the fp32
reference gate green on every row (`nontie == 0`, agreement identical to the eager rows).

## Slice 1: sync sweep

Inventory via code audit plus GPU probes (`--command sync`: per-pass warning counts under
`torch.cuda.set_sync_debug_mode`, per-op isolation, and per-warning Python stacks). Three
sync sources existed; all are gone:

| where | sync | fix |
| --- | --- | --- |
| `DecodeWindowPlan.begin_step` | `masked_select` queries its output count from the device: one implicit D2H sync **per step** | packed read indices rebuilt from host-known sizes (each request grows exactly one token per step), via `repeat_interleave(..., output_size=)` + index arithmetic; same request-major order, no sync |
| `_flush_decode_window` | blocking `matrix.cpu().tolist()`: one full-stream sync **per window** | the flush *stages*: one `non_blocking` D2H copy of (tokens, GPU-computed EOS mask) into pinned buffers plus a CUDA event, consumed **one window behind**; tokens decoded past a request's EOS or cap are discarded at consume, never emitted (waste, not correctness) |
| `build_decode_window_plan` | five `torch.tensor(host_list, device=cuda)` uploads: blocking pageable H2D **per window open**, stalling the CPU behind the previous window's queued GPU work (found by the stack probe; every flush primitive measured clean in isolation) | build on host, upload `non_blocking` (pageable H2D is host-synchronous per CUDA semantics, so the temporary's lifetime is safe) |

The sampled-token → next-step-input path stays entirely on GPU: within a window the
previous step's argmax feeds the next forward directly, and across pipelined windows the
staged matrix's last row feeds the next window's first step. Stop semantics (max_tokens,
EOS; stop strings live in the server's detokenizer, above the engine) are pinned identical
to the per-step engine by `tests/serving/test_decode_window.py`; `decode_window_size=1`
(the classic per-step path) is untouched.

**GPU sync probe after the fixes** (batch 8, one scheduler pass = one whole 8-step window,
passes 2-3 include the pipelined consume): **0 warnings per pass, both configs**:
`{'eager-window': [0, 0, 0], 'cuda-graphs': [0, 0, 0]}`.

## Slice 2: piecewise CUDA graphs, the vLLM shape

`llm_infer/model/decode_graph.py`, routed through
`PretrainBundleModel.decode_window_step` once `enable_decode_graphs()` is called.

* **Bucketed, padded, captured once.** Default capture sizes (1, 2, 4, 8, 16, 32, 64,
  128); the harness adds 256. A real batch pads up to the nearest bucket; static input
  buffers are `copy_`-ed into and addresses never change. All buckets are captured up
  front at enable time (44.7 s for 9 buckets × 31 segments = 279 graphs on the A100), so
  capture can never leak into a timed region. Nothing is ever recaptured.
* **Piecewise: attention and the paged-KV write stay eager.** Per step the runner replays
  31 captured segments: embed+pre-attention(0), post-attention(l−1)+pre-attention(l) per
  layer, then final norm+lm-head+soft-cap. The flash-attn varlen call, the packed
  history gather, and `write_rows` run eagerly between segments. The ragged KV history
  changes size every step; keeping it out of the graphs removes all shape dynamism.
  Keeping the KV *write* eager means no graph bakes in a `PagedKVCache` tensor address:
  one capture serves every engine built on the model (the bench builds a fresh engine per
  iteration).
* **Padding is contained.** Pad rows flow through the captured segments (row-independent
  math, garbage never mixes into real rows) and are *skipped* by the eager attention: pad
  K/V is never written, pad queries never attend, pad logits are dropped by the
  `[:batch]` slice. Deviation from the plan's "pad rows attend to a dummy block": skipping
  them is strictly less work and needs no dummy block, because the write side is eager and
  writes real rows only.
* **One shared memory pool** across all buckets' captures; every tensor crossing a segment
  boundary (hidden, residual, q/k/v, attention out, logits) is a static buffer allocated
  outside the pool, so interleaved replays across buckets cannot alias live state.
* **RoPE rows are pinned.** The runner holds strong references to the capture-time table
  and bounds every graph-path position by it; a window that could reach past it, or a
  batch above the largest bucket, falls back to the eager planned path for that window.
* **CPU-testable.** `mode="eager"` runs the same segment functions and static-buffer flow
  without capture; `tests/model/test_decode_graph.py` pins padded-vs-plain logits parity,
  buffer contents, bucket selection, and the fallbacks on the tiny bundle.

Full-graph capture (flash-attn inside the graph) is outside the current engine: the packed
varlen layout grows every step, so full capture would need a fixed-max-length padded attention
path with a different kernel shape than the parity-gated one.

## Launch counts and GPU time (same container, per 8-step scheduler pass)

`--command capture`, torch.profiler over 16 passes. Per token-step, divide by 8.

| batch | config | cudaLaunchKernel/pass | cudaGraphLaunch/pass | GPU busy/pass |
| ---: | --- | ---: | ---: | ---: |
| 8 | eager-window | 12,986 | 0 | 115.3 ms |
| 8 | cuda-graphs | 1,569 | 217 | 60.1 ms |
| 64 | eager-window | 13,210 | 0 | 151.7 ms |
| 64 | cuda-graphs | 1,576 | 217 | 75.3 ms |
| 256 | eager-window | 13,231 | 0 | 243.4 ms |
| 256 | cuda-graphs | 1,597 | 217 | 146.1 ms |

Per token-step at batch 8: **1,623 kernel launches → 196 launches + 27 graph replays**
(−88% dispatch calls), and GPU busy time per pass roughly halves. The GPU was spending
half its "busy" time executing launch overhead, exactly as the overhead profile said.

## Same-GPU bench (one container, reference-gated, 64 new tokens, greedy, bf16 flash)

`modal run scripts/modal_esme_decode_profile.py --command capture --batch-sizes 8,64,256`,
record `bench-results/esme-decode-capture-20260702T191632.json` (A100-SXM4-80GB, SM
1140 MHz, 400 W, on a slower host than the 2026-07-02 overhead-doc container; compare within
this table only). Every row passed the fp32 `PretrainBundleModel.logits()` reference with
the audited tie-tolerant rule: batch 8 = 6 exact + 2 genuine ties, batch 64 = all exact,
batch 256 = 192 exact + 64 ties, **nontie 0 everywhere**, and the graphs rows match the
eager rows' agreement exactly.

| config (tok/s) | batch 8 | 64 | 256 |
| --- | ---: | ---: | ---: |
| per-step (window=1) | 145.1 | 581.2 | 938.5 |
| window, classic decode_many | 144.1 | 578.4 | 931.3 |
| window + planned buffers | 152.9 | 606.5 | 987.2 |
| **window + planned + cuda graphs** | **424.0** | **878.7** | **1142.1** |
| graphs vs eager window | **+177%** | **+45%** | **+16%** |

Rollback rule (<10% like-for-like → revert capture) is decisively cleared at every batch
size. The gain shrinks as batch grows because the per-step launch count is
batch-independent while useful GPU math scales with batch, which is the expected shape.
An earlier identical run (`esme-decode-capture-20260702T185132.json`, before the
plan-upload sync fix) shows the same numbers within noise, so the headline gain is the
graphs; the sync sweep makes capture possible and adds
a small boundary-stall saving.

Headline `benchmark.md`/README numbers are the pinned public record; compare the tables in this
note only within the same run.
