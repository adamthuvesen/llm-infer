# Historical Qwen Rollout Timing

This is archived Qwen public-baseline evidence for the older `llm-rlvr` rollout hook. It is
kept so the past public-model result remains inspectable. The current repo story is Esme; see
[../benchmark.md](../benchmark.md).

## Workload

The benchmark times one frozen GRPO-style rollout batch:

| knob | value |
| --- | --- |
| prompts | 8 Spider-dev questions |
| generations per prompt | 4 |
| completions per batch | 32 |
| max completion length | 1024 |
| sampling | temperature 1.0, top_p 1.0, seed 0 |
| model | `Qwen/Qwen2.5-Coder-3B-Instruct` plus merged `grpo-s0` LoRA |

Prompts are byte-identical to the original `llm-rlvr` evaluation path and frozen in
`llm_infer/fixtures/rollout_grpo_s0_spider_dev.json`.

## Systems

| system | role |
| --- | --- |
| `hf_sequential` | floor: HF `generate()` once per completion, sequentially |
| `llm_infer` | this engine, bf16 flash backend, all completions in one paged cache |
| `vllm` | ceiling: vLLM offline generation with the same merged weights |

Sampling uses different engine-specific RNG implementations, so cross-system token equality is
not expected. This document is timing evidence, not a reference-equivalence benchmark.

## Result

Run `2026-06-21` on Modal A100-80GB. Served weights: pinned Qwen2.5-Coder-3B-Instruct plus
the `grpo-s0` adapter, merged to bf16. Workload: 32 completions, 1 warmup + 2 measured
iterations, median wall-clock.

| system | wall-clock s | output tok | tok/s | $/1k rollouts | vs floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hf_sequential` | 85.41 | 3366 | 39.4 | $1.85 | 1.00x |
| `llm_infer` | 45.91 | 3026 | 65.9 | $1.00 | 1.67x |
| `vllm` | 1.32 | 2864 | 2170.6 | $0.03 | 55.1x |

`llm_infer` beats the naive floor on this rollout-shaped workload. vLLM remains far ahead, as
expected from a mature serving stack with CUDA graphs, an optimized scheduler, and custom
paged-attention kernels.

## Prefix Caching Record

A later run of the same frozen workload with sibling prefix caching enabled produced:

| system | wall-clock s | output tok | tok/s | $/1k rollouts | vs floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| `hf_sequential` | 81.86 | 3366 | 41.1 | $1.78 | 1.00x |
| `llm_infer` with prefix caching | 7.35 | 3026 | 411.5 | $0.16 | 10.01x |
| `vllm` | 1.37 | 2864 | 2096.3 | $0.03 | 50.98x |

Prefix caching reduced repeated prompt prefill work for the 8 x 4 sibling shape. This remains
historical Qwen evidence; it should not be read as an Esme result.

## Reproducing

The harness is `scripts/modal_rollout.py`; adapter merge support is in
`scripts/merge_adapter.py`. The raw outputs land under `bench-results/`, which is git-ignored.

Run the local correctness checks before spending GPU time:

```bash
uv run pytest tests/correctness -q
uv run pytest -q
```

Use the Modal harness only when intentionally reproducing this archived Qwen record.
