# Benchmark Record

Esme is the primary benchmark path. `Esme-214M-Chat` is checked against the source bundle
oracle before any speed row reports tok/s, then compared against naive HuggingFace and vLLM
through a converted HF checkpoint. Qwen records are retained only as historical public-baseline
evidence.

## Current Esme Result

Run `2026-06-30` on **A100-80GB**. Workload: 8 requests x 64 new tokens, greedy, prefix
caching off, 1 warmup + 3 measured iterations, median wall-clock. vLLM is `0.23.0`; all rows
use bf16 for timed generation and are checked against the fp32
`PretrainBundleModel.logits()` oracle.

| model | system | reference agreement | median s | output tok | tok/s |
| --- | --- | --- | ---: | ---: | ---: |
| `Esme-214M-Chat` | `hf_sequential` | yes, 8/8 exact | 13.427 | 454 | 33.8 |
| `Esme-214M-Chat` | `llm_infer` | yes, 6 exact + 2 ties | 2.625 | 486 | 185.2 |
| `Esme-214M-Chat` | `vllm` | yes, 6 exact + 2 ties | 0.149 | 472 | 3163.4 |

`llm_infer` is **5.5x** the naive HF baseline on Esme. vLLM is the ceiling, not
the system this repo claims to beat.

## What Was Measured

The Esme bundle is converted to a native `Qwen3ForCausalLM` checkpoint for the HF and vLLM
baselines. The conversion is a key remap into the matching Qwen3 architecture, gated by local
parity tests that compare converted HF logits against direct bundle logits.

The three measured systems are:

| row | role |
| --- | --- |
| `hf_sequential` | HF `generate()` once per request, one at a time. This is the naive baseline, not a full-recompute strawman. |
| `llm_infer` | Esme paged-KV path on the bundle, using the validated bf16 flash-attn backend and fused batched decode. |
| `vllm` | vLLM offline generation on the converted checkpoint, prefix caching off. This is the ceiling. |

## Reference Gate

The benchmark follows the project rule: match before measuring speed.

For Esme, the oracle is fp32 greedy decode through direct `PretrainBundleModel.logits()`.
Timed bf16 systems are compared with the audited tie-tolerant agreement rule. A row reports
tok/s only when it has zero non-tie divergences. Exact token agreement is preferred; a bf16
near-tie is accepted only when the fp32 top-token margin is inside the documented tolerance.

The `llm_infer` row also depends on the equal-dtype flash gate: bf16 flash must match bf16
`torch_naive` through the same engine path before the flash-backed speed row is trustworthy.

## Esme Paged KV vs Full Recompute

The engine path is also compared against the direct bundle full-recompute baseline. Local CPU
run, fp32, 4 chat requests x 24 new tokens, both systems exact against the reference:

| system | mode | median s | output tok | tok/s |
| --- | --- | ---: | ---: | ---: |
| `llm_infer_paged` | paged KV + batched decode | 1.176 | 96 | 81.6 |
| `full_recompute` | per-request full recompute | 3.276 | 96 | 29.3 |

This CPU figure is relative evidence for the paged path. It is not the headline GPU serving
number.

## Reproducing

Local checks:

```bash
uv run ruff check
uv run pytest tests/correctness -q
uv run pytest -q
```

GPU benchmark and reference scripts live in [../scripts/](../scripts/). They require Modal,
an available Esme bundle, and A100 execution; they are intentionally not part of the default
local check path.

## Historical Qwen Public Baseline

The Qwen benchmark remains for public-model reproduction and regression context. It uses
`Qwen/Qwen2.5-Coder-3B-Instruct` at revision
`488639f1ff808d1d3d0ba301aef8c11461451ec5`, with the Instruct chat template.

Run `2026-06-21` on **A100-80GB PCIe**, 32 requests x 128 new tokens, greedy, prefix caching
off. Only systems that agreed with the fp32 reference reported throughput.

| system | reference agreement | tok/s |
| --- | --- | ---: |
| `hf_sequential` | diverged beyond tie tolerance | not reported |
| `hf_batched` | diverged beyond tie tolerance | not reported |
| `llm_infer` | yes, 32/32 ties | 98.3 |
| `vllm` | yes, 32/32 ties | 4323.6 |

The historical result is useful for reproducing older public-model evidence. New benchmark
work should use Esme unless the task is explicitly about that Qwen record.

## Historical Rollout Timing

The archived Qwen rollout timing is kept in
[internal/keeping-the-gpu-busy.md](internal/keeping-the-gpu-busy.md) for reproduction context.
Sampling means cross-system token equality is not expected, so that record is timing evidence
rather than a reference-equivalence benchmark. It is not the current repo story.
