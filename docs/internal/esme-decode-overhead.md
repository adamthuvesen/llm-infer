# Esme Decode Overhead — Profile and Reduction (2026-07-02)

Branch `adam/esme-decode-overhead`, base `786f821`. Goal: widen the Esme headline gap over
naive HF by removing Python per-step overhead in the decode loop. Everything here ran on
Modal A100-80GB, bf16 flash backend, greedy, prefix caching off, 64 new tokens, and every
speed row passed the fp32 `PretrainBundleModel.logits()` reference with the audited
tie-tolerant rule (`nontie == 0` on every row cited below). Harness:
`scripts/modal_esme_decode_profile.py` (`profile` / `bench` / `ablate`).

## Phase A — where the ~43 ms/step went (baseline `ad1c46e`)

Profile run `bench-results/esme-decode-profile-20260702T102616.json`
(A100-SXM4, SM 1140 MHz):

| batch | wall/step | GPU busy/step | decode tok/s | kernel launches/step |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 46.2 ms | 19.3 ms | 164.2 | ~2131 |
| 32 | 54.1 ms | 20.5 ms | 523.0 | ~2162 |
| 128 | 59.8 ms | 26.7 ms | 2029.8 | ~2162 |

The step was CPU-bound, but not where the mission brief guessed. The breakdown:

* **Kernel-launch dispatch is the wall.** ~2,100 `cudaLaunchKernel` calls per step;
  15.3 ms/step of CPU in the launch call alone at batch 8. The launches come from the layer
  math, not the paged-KV bookkeeping: `rms_norm` runs 121 times/step (30 layers x
  input/post-attention/q/k norms + final) at ~7 ops each (~850 launches), `apply_rope` 60
  times (~480), 211 matmuls, plus per-call dtype casts — `weight.to(fp32)` inside `rms_norm`
  launched 121 real cast kernels per step on the bf16 model.
* **GPU "busy" is itself launch-shaped.** 19 ms/step of GPU time across ~2,100 tiny kernels
  (~9 µs each) for a model whose decode math is ~3 GFLOP. The GPU is busy executing launch
  overhead, not starved by one big sync.
* **Block-table Python scaled with batch.** At batch 128 the cProfile top was
  `physical_slot` (594k calls), `prepare_write` (237k), `capacity` (831k) — `write_many`
  re-walking tables per layer and `plan_read_many` re-listing every history per step.
* **Per-token host syncs existed but were cheap.** One `.cpu().tolist()` EOS check per step;
  since the CPU, not the GPU, was behind, removing it alone could not move the wall.

## Phase B — changes and attribution

One commit per change; the classic per-step path (`decode_window_size=1`) is preserved.

1. `c8c2ee1` — all-greedy batches sample with one batched argmax; EOS lookup tensors are
   cached per engine instead of rebuilt (H2D) per step.
2. `46600fb` — deferred decode window (`decode_window_size`, default 8): sampled tokens stay
   on device, EOS/stop is decided at one host sync per window, overshoot past EOS is
   discarded at the flush (outputs stay token-for-token identical; pinned by
   `tests/serving/test_decode_window.py`).
3. `02981c3` — planned decode windows (`llm_infer/model/decode_plan.py`): write slots for the
   whole window and the packed read layout are built once per window and advanced with ~6
   device kernels/step; `write_rows` replaces per-layer table walks; RoPE rows come from a
   cached, dtype-cast table. `plan == plan_read_many` is pinned step-by-step by
   `tests/model/test_decode_plan.py`.
4. `725d43d` — multi-step scheduling: one scheduler/admission pass runs the whole open window
   (per-pass cadence returns whenever the waiting queue is non-empty).
5. `9b1c975` — profile-driven invariant hoists: RMSNorm weights pre-cast to fp32 once
   (−121 cast kernels + allocs/step) and the decode `cu_seqlens_q` arange cached in the flash
   backend (−30 launches/step). Bit-identical values, cast/built once.

### Per-commit Modal runs (cross-container — read with the variance note below)

`modal run scripts/modal_esme_decode_profile.py --command bench`, tok/s at batch 8/32/128:

| code state | batch 8 | 32 | 128 | GPU (snapshot) |
| --- | ---: | ---: | ---: | --- |
| baseline (`ad1c46e`) | 176.3 | 485.1 | 998.4 | PCIe, 1410 MHz, 300 W |
| + change 1 | 166.2 | 462.9 | 946.2 | SXM4, 1365 MHz, 400 W |
| + change 2 | 185.5 | 478.8 | 967.7 | SXM4, 1410 MHz, 400 W |
| + change 3 | 180.2 | 488.0 | 989.4 | SXM4, 1410 MHz, 400 W |
| + change 4 | 144.2 | 399.4 | 807.1 | SXM4, **1170 MHz**, 400 W |

**These cross-container deltas are not decision-grade.** The decode loop is host-CPU-bound
and Modal A100 containers vary in GPU SKU, SM clock, and host CPU; identical code moved
±20% between runs. That finding is itself worth keeping: single-run cross-container tok/s
deltas under ~±25% say nothing here.

### Same-GPU ablation (decision-grade)

`--command ablate` measures all engine configs back to back in one container
(`esme-decode-ablate-20260702T112802.json`, A100 PCIe 1410 MHz 300 W, final code):

| config | batch 8 | 32 | 128 |
| --- | ---: | ---: | ---: |
| per-step (`decode_window_size=1`) | 234.9 | 631.6 | 1291.8 |
| window, classic `decode_many` | 234.9 | 648.7 | 1289.7 |
| window + planned buffers (default) | 245.9 | 685.1 | 1380.8 |

Every row: reference agreement `nontie 0` (batch 8: 6 exact + 2 genuine bf16 ties; 32: all
exact; 128: 96 exact + 32 ties — the tie is the known esme-001 step-22 fp32 gap 0.0119).

Attribution on one GPU: the deferred window alone (changes 2+4) is worth ~0–3% — as the
Phase A profile predicted, the per-step sync was never the wall. The planned buffers
(change 3) add +4.7% / +5.6% / +7.0% on top and require the window to exist. The
unconditional changes (1, RoPE row cache inside classic `decode_many`, and 5) are in every
row above; their effect shows in the like-for-like pair next.

### Like-for-like before/after (same GPU SKU, clocks, power)

The baseline bench run and the ablation ran on the same A100 PCIe 1410 MHz / 300 W
configuration (host CPU not controllable):

| batch | before (`ad1c46e`) | after (all changes) | change |
| ---: | ---: | ---: | ---: |
| 8 | 176.3 | 245.9 | +39% |
| 32 | 485.1 | 685.1 | +41% |
| 128 | 998.4 | 1380.8 | +38% |

### Rollback-rule accounting

* Change 1: below 10% alone (noise-level) but near-zero complexity — kept.
* Changes 2+4: ~0–3% alone; kept because they are the structural precondition for the
  planned buffers (a stable batch with no per-token host boundary) and for any future CUDA
  graph slice, and `decode_window_size=1` preserves the classic path exactly.
* Change 3: +5–7% same-GPU, and it removes the batch-scaling Python (594k `physical_slot`
  calls/run at batch 128 → gone from the profile) — kept.
* Change 5: unconditional, bit-identical, removes ~150 launches/step — kept.

## Where the time goes now (final profile, `esme-decode-profile-20260702T114417.json`)

Post-change, per token-step at batch 8: ~1,850 kernel launches (was ~2,130), `copy_` down
484 → 363/step (norm casts gone), block-table Python gone from the cProfile top at every
batch size. What remains is almost entirely per-layer math dispatch: 211 matmuls, the
7-op fp32 `rms_norm` sequence x121, `apply_rope` x60, GQA `repeat_interleave` x60, and the
flash varlen call x30. Note the profile harness's `wall_ms_per_step` now counts scheduler
*passes* (one pass = one whole window since change 4); use `decode_tokens_per_second` for
comparisons.

**Conclusion for the next slice:** the remaining decode wall is kernel-launch dispatch from
~1,850 tiny kernels/step, roughly half CPU dispatch and half GPU launch-overhead execution.
No further paged-KV/bookkeeping cut can move it much. The lever that fits is CUDA graph
capture (or torch.compile) over the planned window step — explicitly out of scope for this
slice, and the planned window (stable buffers, no per-step host work, no allocation in the
read path except `masked_select`) was shaped to be capture-friendly.

## Headline (pinned three-way) and variance

`modal run scripts/modal_esme_three_way.py --command bench`, two fresh runs, both
reference-checked (llm_infer row: 6 exact + 2 ties, nontie 0):

| run | hf_sequential | llm_infer | llm_infer / HF |
| --- | ---: | ---: | ---: |
| pinned 2026-06-30 record | 33.8 | 185.2 | 5.5x |
| 2026-07-02 run 1 | 30.3 | 190.7 | 6.3x |
| 2026-07-02 run 2 (slow host) | 22.8 | 145.7 | 6.4x |

Both new runs landed on slower hosts than the 2026-06-30 record (their own HF baselines are
10% and 33% below the pinned HF row). The host-independent movement is the same-run ratio:
5.5x → 6.3–6.4x over naive HF. The equal-dtype flash gate
(`modal run scripts/modal_esme_flash_reference_check.py --command check`) passed 8/8 exact
on this branch.

## Pre-existing issue noted, not fixed here

`Request.record()` stores speculative-decode draft tokens as Python ints (CPU tensors)
while prefill/decode tokens are device tensors; on CUDA, `Request.generated` would stack
mixed devices and raise. Esme/Qwen speculative decoding is CPU-tested only today. Same
shape as the bug fixed inside the window flush (which records device-tensor views).
