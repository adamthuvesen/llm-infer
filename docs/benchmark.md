# Benchmarks

`llm-infer` reports speed only after the generated tokens match the reference output for
the same prompt and model. The public benchmark compares:

| system | role |
| --- | --- |
| `hf_sequential` | HuggingFace `generate()` once per request, one at a time: the naive baseline a first serving script would ship. |
| `llm_infer` | This engine: paged KV, continuous batching, batched greedy decode, bf16 flash-attn. |

The model is `Esme-214M-Chat` (214M parameters, 1024-token context), loaded from its export
bundle. The HF baseline runs a converted `Qwen3ForCausalLM` checkpoint emitted from the same
bundle and checked against the bundle reference.

Every row below passed the fp32 `PretrainBundleModel.logits()` reference gate with zero
non-tie divergences. Genuine bf16 numerical ties are accepted only after recomputing the
divergence on the fp32 reference.

## Headline

Run `2026-07-02` on A100-80GB: **64 concurrent chat requests, up to 256 new tokens each**,
greedy, prefix caching off, median of 3 measured iterations after 1 warmup.

| model | system | reference agreement | median s | output tok | tok/s | vs floor |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `Esme-214M-Chat` | `hf_sequential` | yes, 24 exact + 40 ties, 0 non-tie | 301.545 | 10,432 | 34.6 | 1x |
| `Esme-214M-Chat` | `llm_infer` | yes, 32 exact + 32 ties, 0 non-tie | 10.902 | 10,144 | 930.4 | **26.9x** |

The output-token totals differ slightly because a genuine bf16 tie can change when a
continuation reaches EOS. Those flips were checked against the fp32 reference before tok/s was
reported.

## Throughput Curve

![Esme batch-size throughput curve](../assets/fig-esme-batch-curve.svg)

Same workload family as the headline: chat prompts from a fixed pool, greedy decoding, up to
256 new tokens per request, prefix caching off. The committed record is
[`assets/esme-batch-curve.json`](../assets/esme-batch-curve.json), and the figure regenerates
from repo state with:

```bash
uv run scripts/plot_benchmark_curve.py
```

| concurrent requests | llm_infer tok/s | hf_sequential tok/s |
| ---: | ---: | ---: |
| 8 | 206.0 | 47.8 |
| 16 | 388.2 | 44.4 |
| 32 | 720.1 | 45.6 |
| 64 | 1,153.0 | 43.5 |
| 128 | 1,908.9 | 44.9 |
| 256 | 2,748.5 | 45.4 |

At 256 concurrent requests, the engine serves about **57x** the measured naive-HF floor in
that same run.

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
