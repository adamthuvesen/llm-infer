# Esme Decode — Graph Wiring + torch.compile Comparison (2026-07-02)

Local `main`, base `51c2602`. Follow-up to
[decode-graph-capture.md](decode-graph-capture.md), which shipped the piecewise
CUDA-graph decode runner but left it unwired. This slice (A) makes the captured path what
serving and the benchmark harnesses actually run, and (B) builds the vLLM-style
alternative — `torch.compile` over the decode step with paged attention as an opaque
custom op — and races the two on one GPU.

**Verdict: the manual piecewise capture stays the serving default.** On the same A100,
`torch.compile(mode="reduce-overhead")` reaches the *same* kernel-launch reduction as the
manual runner (~1,576 + 217 graph launches per 8-step pass vs eager's ~13,000) but loses
the wall-clock race at every batch size — at batch 8 it lands *below the eager window*
(118.7 vs 155.2 tok/s; manual graphs: 425.0). The gap is host-side per-step framework
overhead, not GPU work. The compiled runner is demoted to an opt-in measurement path.

## Part A — wiring (what runs the graphs now)

The runner is model-owned, so one capture serves every engine built on the model. One
helper, `enable_decode_graphs_if_cuda` (`llm_infer/model/decode_graph.py`), gates on
CUDA + a bundle backend and returns the capture time; every entry point logs it.

| entry point | default | escape hatch | buckets |
| --- | --- | --- | --- |
| `serve.py` / `build_app_from_runtime` | ON (capture at startup, before serving) | `--no-decode-graphs` | `(1, 2, 4, 8, 16)` — ~5 s/bucket capture, sized for a demo server's concurrency; `--decode-graph-buckets` to change |
| `scripts/esme_serving_eval.py` (local target) | ON on CUDA, no-op on CPU | `--no-decode-graphs` | `DEFAULT_CAPTURE_SIZES` (1..128) |
| `scripts/modal_esme_three_way.py` | ON for the llm_infer row | — | `DEFAULT_CAPTURE_SIZES` (covers headline batch 64) |
| `scripts/modal_esme_batch_curve.py` | ON for the llm_infer rows | — | `(1..128, 256)` so batch 256 replays, not falls back |

Batches above the largest bucket (and windows past the pinned RoPE rows) fall back to the
eager planned window per window — correct, just slower. Capture time on this container:
**43.3 s for 9 buckets** (harness spread), so the serve default's 5 buckets land around
25 s of startup. Serving-level parity is pinned by
`tests/serving/test_decode_graph_serving.py`: identical SSE streams with the runner on and
off through `build_app_from_runtime` → async engine → OpenAI endpoint, and the CPU serve
default stays a no-op.

Public `benchmark.md`/README numbers are untouched per the mission non-goal; the pinned
full-protocol rerun is its own future slice, and its harnesses now measure the graphs path
when it comes.

## Part B — the torch.compile runner

`llm_infer/model/decode_compile.py` (`CompiledDecodeRunner`, enabled via
`PretrainBundleModel.enable_decode_compile`). Same caller contract and same
`model.decode_graphs` slot as the manual runner — buckets, padding, eager fallbacks — but
the segment math goes to Dynamo/Inductor instead of hand-captured graphs:

* **Opaque attention boundary.** `llm_infer::paged_decode_attention` — a mutable custom op
  (`Tensor(a!) out`, `register_fake` for tracing, `cudagraph_unsafe` tag) that runs the
  eager per-layer slice: paged-KV `write_rows`, ragged history gather, GQA expansion,
  flash-attn varlen, writing the real rows into a passed-in output buffer. The cache
  handle, real batch, write slots, and read plan cross the boundary through a module-level
  step context, so no gather size or cache address can be traced into the graph.
  `torch._inductor.config.graph_partition` + the tag make Inductor's CUDA graphs split at
  the op — the launch counts confirm it worked (217 graph launches per pass, same as the
  31-segment manual replay shape).
* **One traced step, dynamic batch.** The whole step (embed → 30 layers → soft-capped
  logits) compiles `fullgraph=True`; `mark_dynamic` on the batch dim keeps all buckets >1
  on one artifact (bucket 1 specializes, as Dynamo always does) — **2 unique Dynamo graphs
  for 9 buckets**. Positions and tokens cross as per-bucket device buffers, never Python
  ints; sequence lengths live in the read plan's `cu_seqlens` tensor.
* **Warmup at enable time, parity-checked.** Every bucket compiles, records, and replays at
  startup against a throwaway cache, and each warmup step is compared to the eager planned
  path — a torch build that CUDA-graph-captures *through* the custom op (baking one step's
  gather into the replay) fails loudly at enable time. The harness falls back to
  `mode=None` (fusion only) on that failure; torch 2.8.0 on this container passed in
  `reduce-overhead`.
* **Recompile audit: clean.** `TORCH_LOGS=recompiles` in the container log plus counters
  around the bench rows: `unique_graphs 2 → 2`, `cudagraph_skips 0 → 0`, **0 new graphs**
  across all 15 oracle-gated rows.

Two GPU-run potholes worth recording: the shared flash image needed a host C toolchain
(Triton builds its launcher stubs with `cc` at runtime — `build-essential` added), and
`mark_static_address` on the per-bucket input buffers cost one Dynamo cache entry per
bucket via its object-identity guard, tripping `recompile_limit(8)` under `fullgraph` with
9 buckets. Dropping it is strictly better: compiler-managed graphs copy the two tiny
`(B,)` inputs themselves.

## Same-GPU bench (one container, reference-gated, 64 new tokens, greedy, bf16 flash)

`modal run scripts/modal_esme_decode_profile.py --command capture --batch-sizes 8,64,256`,
record `bench-results/esme-decode-capture-20260702T204006.json` (A100-80GB PCIe, SM
1410 MHz, 300 W — compare within this table only; ±20% across containers). Every row
passed the fp32 `PretrainBundleModel.logits()` reference with the audited tie-tolerant
rule: **nontie 0 on all 15 rows**. The compile rows resolve some genuine tie steps
differently (batch 8: 8 exact vs the other configs' 6 exact + 2 ties; batch 64: 48 exact +
16 ties) — Inductor's fusions round bf16 differently on exact-tie logits, which the tie
rule exists to absorb.

| config (tok/s) | batch 8 | 64 | 256 |
| --- | ---: | ---: | ---: |
| per-step (window=1) | 146.4 | 593.3 | 964.0 |
| window, classic decode_many | 148.2 | 589.6 | 960.2 |
| window + planned buffers | 155.2 | 623.0 | 1011.6 |
| **window + planned + cuda graphs (default)** | **425.0** | **931.8** | **1163.0** |
| window + planned + torch.compile | 118.7 | 597.0 | 967.4 |

Launch counts per 8-step scheduler pass (torch.profiler, 16 passes) and startup cost:

| config | launches/pass @8 | graph launches | GPU busy @8 | startup |
| --- | ---: | ---: | ---: | ---: |
| eager-window | 12,986 | 0 | 101.6 ms | — |
| manual cuda graphs | 1,569 | 217 | 55.2 ms | 43.3 s capture |
| torch.compile | 1,576 | 217 | 59.9 ms | **407.2 s** compile+warmup |

## Why compile loses

The GPU side is nearly identical — same dispatch reduction, GPU busy within ~10% of the
manual runner. The loss is host-side Python per step: Dynamo guard evaluation over the
traced step's inputs, the cudagraph-tree runtime's dispatch/liveness bookkeeping, and the
auto-functionalization wrappers around 30 mutable custom-op calls per step. The manual
runner's steady-state step is 31 `graph.replay()` calls plus the eager attention slice —
near-zero Python. At 214M the whole step budget is a few milliseconds, so milliseconds of
framework overhead swallow the entire graph win (batch 8: compile is 24% *below* the eager
window) and only amortize once the GPU math dominates (batch 256: compile ≈ eager-planned,
still 17% behind manual graphs). Add the 9.4× startup cost and there is no batch size at
which compile is the right default for this model class. vLLM's piecewise-compile design
targets models where step time is tens of milliseconds; at 214M the Python wrapper *is*
the workload.

## Kept / demoted

* **Kept (serving default):** the manual piecewise runner, wired per Part A.
* **Demoted, not deleted:** `CompiledDecodeRunner` stays as an opt-in measurement path —
  nothing constructs it by default, no serving entry point references it. It is kept
  because it is the only instrument for re-racing the compiler stack when torch is
  upgraded (Inductor's graph partition and cudagraph-tree overhead change release to
  release), its CPU tests (`tests/model/test_decode_compile.py`, ~2.5 s) pin the contract,
  and the harness re-measures it in one command (`--command capture`). If that re-race
  never gets exercised, deleting `decode_compile.py` + its tests + the harness config is a
  clean three-file removal.

## Spend

Three Modal A100-80GB runs (`llm-infer-esme-decode-profile`): `ap-5AJmk80RqQRPLFzjbo88SP`
(failed in ~5 min — no C compiler in the image), `ap-8pkFJJniHFtwao9tnAzW2W` (failed in
~12 min — recompile limit), `ap-acI56fduAyKlxuKlni0qlw` (full report, ~28 min GPU).
Roughly 45 GPU-minutes ≈ **$3, well under the $10 cap**.
