# Benchmarks

`llm-infer` reports speed only after the generated tokens match the reference output for
the same prompt and model. The public benchmark compares:

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

Run `2026-07-07` on A100-80GB with the default CUDA Esme path,
`FlashInferPagedAttention`: **3,776.8 tok/s at 256 concurrent requests**, **95.6x** the
same-run naive-HF floor. At 64 concurrent requests, the engine serves **2,603.4 tok/s**.

`FlashInferPagedAttention` is the `auto` choice for CUDA fp16/bf16 Esme bundles and needs the
`gpu` extra (`uv sync --extra gpu`); the Modal harnesses bake it into the shared GPU image. To
serve or benchmark on CUDA without it, pass `--attention-backend torch_naive` or
`--attention-backend flash_attn`.

## Throughput Curve

![Esme batch-size throughput curve](../assets/fig-esme-batch-curve.svg)

Same run as the headline: chat prompts from a fixed pool, greedy decoding, up to 256 new
tokens per request, prefix caching off, median of 3 measured iterations after 1 warmup. The
committed record is
[`assets/esme-batch-curve.json`](../assets/esme-batch-curve.json), and the figure regenerates
from repo state with:

```bash
uv run scripts/plot_benchmark_curve.py
```

| concurrent requests | llm_infer tok/s | hf_sequential tok/s |
| ---: | ---: | ---: |
| 8 | 658.6 | 39.4 |
| 16 | 1,140.0 | 39.6 |
| 32 | 1,844.9 | 39.7 |
| 64 | 2,603.4 | 39.3 |
| 128 | 3,275.5 | 39.5 |
| 256 | 3,776.8 | 39.5 |

At 256 concurrent requests, the engine serves **95.6x** the same-row measured naive-HF floor.

## Method

- Workload: chat-templated prompts, 15-41 prompt tokens, greedy decoding, up to 256 new tokens,
  EOS stopping, prefix caching off.
- Timing: median wall-clock over 3 measured iterations after 1 warmup, with a fresh engine per
  measured iteration.
- Reference gate: fp32 full-recompute `PretrainBundleModel.logits()` is the oracle. A row with
  any non-tie divergence reports no tok/s.
- Comparison rule: raw tok/s is compared only inside the same benchmark run; same-run ratios are
  the useful number.

## Local Checks

These commands do not rerun GPU benchmarks; they check the local code, committed curve record,
and figure-generation path:

```bash
uv run ruff check
uv run pytest -q
uv run scripts/check_benchmark_evidence.py
```
