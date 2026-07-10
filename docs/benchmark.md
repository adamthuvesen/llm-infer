# Benchmarks

`llm-infer` keeps raw benchmark timing, but reports public headline speed only after the generated
tokens qualify under the reference policy for the same prompt and model. The public benchmark
compares:

| system | role |
| --- | --- |
| `hf_sequential` | HuggingFace `generate()` once per request, one at a time: the naive baseline a first serving script would ship. |
| `llm_infer` | This engine: paged KV, continuous batching, batched greedy decode, and FlashInfer paged attention on the default CUDA Esme path. |

The model is `Esme-214M-Chat` (214M parameters, 1024-token context), loaded from its export
bundle. The HF baseline runs a converted `Qwen3ForCausalLM` checkpoint emitted from the same
bundle and checked against the bundle reference.

Every row below passed the fp32 `PretrainBundleModel.logits()` reference gate with zero
non-tie divergences. Genuine bf16 numerical ties are accepted only after recomputing the
divergence on the fp32 reference.

## Headline

Run `2026-07-10` on A100-80GB with the default CUDA Esme path,
`FlashInferPagedAttention`: **13,490.5 tok/s at 256 concurrent requests**, **610.8x** the
same-run naive-HF floor. At 64 concurrent requests, the engine serves **4,695.2 tok/s**.

`FlashInferPagedAttention` is the `auto` choice for CUDA fp16/bf16 Esme bundles and needs the
`gpu` extra (`uv sync --extra gpu`); the Modal harnesses bake it into the shared GPU image. To
serve or benchmark on CUDA without it, pass `--attention-backend torch_naive` or
`--attention-backend flash_attn`.

## Throughput Curve

![Esme batch-size throughput curve](../assets/fig-esme-batch-curve.svg)

Same run as the headline: chat prompts from a fixed pool, greedy decoding, up to 256 new
tokens per request, and prefix caching off. Each `llm_infer` row is the median of 3 measured
iterations after 1 warmup. The sequential HF batch-8 anchor uses the same repetition count;
the batch-16 through batch-256 HF rows are single measured flatness checks with no warmup because
each iteration runs every request sequentially. The committed record is
[`assets/esme-batch-curve.json`](../assets/esme-batch-curve.json), and the figure regenerates
from repo state with:

```bash
uv run scripts/plot_benchmark_curve.py
```

| concurrent requests | llm_infer tok/s | hf_sequential tok/s |
| ---: | ---: | ---: |
| 8 | 645.0 | 22.1 |
| 16 | 1,297.9 | 22.0 |
| 32 | 2,438.9 | 22.1 |
| 64 | 4,695.2 | 22.3 |
| 128 | 8,417.0 | 22.2 |
| 256 | 13,490.5 | 22.1 |

At 256 concurrent requests, the engine serves **610.8x** the same-row measured naive-HF floor.

Two things moved since the previous committed record (`2026-07-07`, 3,776.8 tok/s, 95.6x).
Most of the gain is engine work: batched ragged prefill is now on by default, so the 256
prompts prefill in packed calls instead of one at a time — the improvement grows with batch
exactly as that predicts (batch 8 is unchanged within noise). The naive-HF floor also
measured 22.1 tok/s on this run's container versus 39.5 on the previous one: the sequential
`generate()` loop is host-bound, and cross-container host speed is the dominant term for it.
Both sides of the multiple come from the same container in the same run, as always.

## Method

- Workload: chat-templated prompts, 15-41 prompt tokens, greedy decoding, up to 256 new tokens,
  EOS stopping, prefix caching off.
- Engine timing: median wall-clock over 3 measured iterations after 1 warmup, with a fresh engine
  per measured iteration.
- HF timing: batch 8 uses 3 measured iterations after 1 warmup. Batches 16-256 use one measured
  iteration and no warmup; these rows check that sequential HF throughput stays flat, not its
  run-to-run spread.
- Reference policy: fp32 full-recompute `PretrainBundleModel.logits()` is the oracle. Policy-v2
  records keep raw tok/s in every measured row and expose `headline_eligible` separately. Public
  tables use only eligible rows.
- bf16 tie rule: the Esme benchmark uses a 0.1-logit tolerance when it recomputes the first
  divergence with the fp32 oracle. This is wider than the generic 1e-3 fused-kernel fixture rule
  because the fast Esme row converts the whole model to bf16, not just attention. The headline
  pool contains 8 unique prompts; larger batches repeat them. The current evidence includes
  accepted gaps up to 0.0463 logits, but it does not establish a full error distribution up to
  0.1. Treat the threshold as an audited acceptance bound, not a measured noise percentile. A
  larger numerical difference is `review_required`, not automatically a bug; accepting it needs
  a durable diagnostic.
- Comparison rule: raw tok/s is compared only inside the same benchmark run; same-run ratios are
  the useful number. An A/B ratio also needs `exact` or explicitly reviewed `accepted_numerical`
  candidate/baseline parity and equal token counts. It does not need both paths to independently
  clear an fp32 review when they share the same continuation.

## Local Checks

These commands do not rerun GPU benchmarks; they check the local code, committed curve record,
and figure-generation path:

```bash
uv run ruff check
uv run pytest -q
uv run scripts/check_benchmark_evidence.py
```
