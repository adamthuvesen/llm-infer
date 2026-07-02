# Decode Graph — Archived Dead End and What the Next Capture Slice Must Respect

> **Historical — measured dead-end. Do not re-implement the static-bucket design.** The
> 32-slot static decode-graph path was built and rejected on the frozen Qwen rollout:
> slower than eager (256 vs 299 tok/s) and it shifted sampled token counts (3036 vs
> 3026). The detailed Qwen-era implementation plan is preserved in git history; this note
> keeps only what a future compile/CUDA-graph slice must not re-learn.

## Updated interpretation (2026-07-02)

The decode-overhead profile in
[esme-decode-overhead.md](esme-decode-overhead.md) settled where the wall actually is at
214M: **kernel-launch dispatch** — ~1,850 tiny kernels per token-step after the
planned-window work, roughly half CPU dispatch and half GPU launch-overhead execution —
not paged-KV bookkeeping and not the per-token host sync the old plan assumed. The planned
decode window (stable buffers, no per-step host work, no allocation in the read path except
`masked_select`) was shaped to be capture-friendly, so the next slice is CUDA graph capture
or torch.compile over the planned window step.

## Rejected shapes — do not retry as shortcuts

Measured on the frozen Qwen rollout (accepted eager baseline at the time: 365.8 tok/s,
3026 output tokens):

| experiment | tokens | tok/s | verdict |
| --- | ---: | ---: | --- |
| static 32-slot decode graph | 3036 | 256 (vs 299 eager) | slower AND changed outputs |
| direct FlashAttention GQA | 3045 | 107.1 | changed outputs, big regression |
| projection/MLP fusion | 2988 | 368.1 | changed outputs for ~nothing |
| step-local prompt-prefix KV copy | 3026 | 315.7 | exact but regressed |
| no-gather FlashAttention paged KV | 3045 | 282.4 | changed outputs, regressed |

## Rollback criteria for any future capture slice

Carried forward from the archived plan, generalized to the current Esme gates:

- The fp32 reference gate (tie-tolerant, zero non-tie divergences) must pass on every
  measured row; a capture path that changes token outputs is rejected by default.
- Speed must beat the current accepted same-container baseline; an improvement under ~10%
  does not pay for graph/scheduler complexity.
- Timed iterations must not silently include capture/recapture work.
- No disabling existing correctness tests, no silent changes to EOS inclusion, stop
  rules, or request order, and no memory growth that forces lower batch admission.

A marginal implementation is kept only as an explicitly-labeled diagnostic scaffold that
proves a ceiling, never as a headline speed path.
