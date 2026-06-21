# Keeping the GPU busy — the rlvr-sql rollout-timing hook (Phase E)

This is the differentiator: the one number that makes `llm-infer` an *rlvr-sql rollout
backend* rather than a nano-vLLM clone. Phase D proved the engine decodes a synthetic
workload correctly and beats naive HF (`docs/benchmark.md`). Phase E points it at the
**real** thing it exists to serve — one frozen rlvr-sql GRPO rollout batch — and times it
against vLLM (rlvr-sql's current rollout backend, the ceiling) and naive HF (the floor).

The framing is honest about scope: v1 is a small, legible, *correct* paged engine. vLLM is a
mature system with CUDA graphs, a real scheduler, and a custom paged-attention kernel. The
gap is the cost of v1's legibility, and it is named below, not hidden.

## The workload — a real rollout batch, not a microbench

One GRPO rollout batch from rlvr-sql's own `GrpoHyperparams`, anchored to **`grpo-s0`** (the
shipped main result):

| knob | value | source |
| --- | --- | --- |
| prompts | 8 Spider-dev questions | `per_device_batch_size=8` |
| generations per prompt (G) | 4 | `num_generations=4` |
| completions per batch | **32** | 8 × 4 |
| max completion length | 1024 | `max_completion_length=1024` |
| temperature | 1.0 | `temperature=1.0` (rollout exploration) |
| top_p | 1.0 | `top_p=1.0` (a no-op nucleus, supported honestly) |
| sampling seed | 0 | `GrpoHyperparams.seed` |

The prompts are **byte-identical to rlvr-sql's eval path**: built through its own
`build_messages(style="cot")` + `serialize_schema()` (the CoT system prompt + deterministic
`CREATE TABLE` blocks), then tokenized with the pinned Qwen2.5-Coder-3B-Instruct chat
template (`add_generation_prompt=True`). Eight Spider-dev questions are picked by a pinned
seed and **frozen as a committed fixture**
(`tests/fixtures/rollout_grpo_s0_spider_dev.json`), so the run reproduces exactly with no
rlvr-sql import and no re-tokenization drift. The chosen dev rows span six distinct schemas
(`pets_1`, `student_transcripts_tracking`, `tvshow`, `world_1`, `orchestra`, `dog_kennels`),
251–868 prompt tokens each.

## The served model — merged grpo-s0, bf16

rlvr-sql ships its policy as a **rank-32 PEFT LoRA adapter** over
`Qwen/Qwen2.5-Coder-3B-Instruct` (pinned revision `488639f1`). The engine stays LoRA-free:
`scripts/merge_adapter.py` folds `grpo/grpo-s0` into the base with PEFT `merge_and_unload`
and writes the **merged bf16** weights to a Modal volume, served identically to llm-infer
and vLLM (same weights → fair timing). The adapter's `base_model_name_or_path` is asserted
against the pin before merging — a wrong base would silently corrupt every number.

Why bf16: it is Qwen2.5's **native** dtype (the model is trained and shipped in bf16, and
rlvr-sql's GRPO run was bf16), it is the dtype the engine's greedy oracle already validates,
and it is the dtype Phase D uses — so all three systems and the engine run **one uniform
precision**, with no mixed fp16/bf16 path to reason about. All three produce coherent SQL
(sampled completions reason in `<think>…</think>` then emit a ` ```sql ` query), confirmed on
a smoke before the timed run.

## The three systems

| row | what it is | role |
| --- | --- | --- |
| `vllm` | vLLM offline generate, prefix caching off, flags pinned, bf16 merged weights. | **The ceiling** — rlvr-sql's *current* rollout backend. Never the thing we beat. |
| `llm_infer` | this engine: flash backend, bf16, all 32 completions in one paged cache, fused batched decode (`decode_many`), seeded sampler. | The engine under test. |
| `hf_sequential` | HF `generate()` once per completion, sequentially, sampling. | **The floor** — the naive thing written first. |

## Methodology — honest about sampling

- **No cross-engine token equivalence.** Phase D adjudicated every system against fp32 truth
  because greedy decoding is deterministic. Under sampling, llm-infer, vLLM, and HF use
  *different* RNG, so identical tokens are neither expected nor honest to require. The
  rollout is **timing-only**. (The sampler's correctness is pinned separately and exactly:
  `tests/correctness/test_sampler.py` asserts that at temperature 0 the sampler equals the
  proven greedy oracle token-for-token — the engine driven entirely through
  `Sampler(temperature=0)` reproduces the committed HF greedy goldens.)
- **Equal work per measured iteration.** Each system re-seeds at the start of every iteration
  (the engine builds a fresh seeded `Sampler`; vLLM pins `SamplingParams(seed=0)`; HF calls
  `set_seed(0)`), so every timed iteration of a given system decodes the *same* tokens and
  only wall-clock varies.
- **Which metric means what.** Under sampling each system decodes its *own* token volume
  (different RNG, different EOS points), so **tok/s is the apples-to-apples speed number**.
  Wall-clock and $/1k reflect each system's own sampled batch — the token counts are shown
  in the table so nothing is hidden.
- **Timing.** `warmup` un-measured iterations (allocator / CUDA-graph / autotune settle),
  then `iters` measured iterations with a CUDA sync at each boundary; throughput is total
  scored output tokens ÷ **median** measured wall-clock.
- **$/1k rollouts** = the cost to generate 1000 completions at the pinned **Modal A100-80GB
  rate of $2.50/hr** ($0.000694/s, modal.com/pricing, 2026-06-21), derived from this batch's
  wall-clock and its 32 completions. One rollout = one completion.
- **One GPU class, pinned.** Both functions run on A100-80GB; the engine/HF baselines and the
  vLLM ceiling run in separate images (vLLM ships its own torch/CUDA) on separate A100-80GB
  instances. GPU name, every library version, and the vLLM flags are captured into the result
  JSON. Modal picks the exact A100-80GB variant (PCIe vs SXM4) per instance and does not let
  you select it — see the note under the result.

## Result

Run `2026-06-21`. Served weights: `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1` + rlvr-sql
`grpo-s0` (rank-32 LoRA), merged **bf16**. Workload: 32 completions (8 Spider-dev prompts ×
G=4), ≤1024 tokens, temperature 1.0, top_p 1.0, seed 0; 1 warmup + 2 measured iterations,
median wall-clock.

| system | wall-clock s | output tok | tok/s | $/1k rollouts | vs floor |
| --- | --- | --- | --- | --- | --- |
| `hf_sequential` (floor) | 85.41 | 3366 | 39.4 | $1.85 | 1.00× |
| **`llm_infer`** (ours) | 45.91 | 3026 | **65.9** | **$1.00** | **1.67×** |
| `vllm` (ceiling) | 1.32 | 2864 | 2170.6 | $0.03 | 55.1× |

(Token counts differ by ~15% across systems because sampling uses different RNG per engine —
3366 / 3026 / 2864 — so the batches are close but not identical; tok/s is the speed metric
that normalizes for this, and the near-equal token volumes keep the wall-clock and $/1k
columns comparable too. grpo-s0 is a trained SQL policy that terminates cleanly, ~95 tokens
per completion on average, far short of the 1024 cap.)

**The engine beats the naive floor on the real rollout.** `llm_infer` (65.9 tok/s) decodes
the GRPO rollout batch **1.67× faster** than naive sequential HF (39.4 tok/s) and at **1.9×
lower $/1k** ($1.00 vs $1.85) — the same win as Phase D's stop #4, now on rlvr-sql's actual
workload rather than the synthetic one. The fused batched decode (`decode_many`) is what earns
it: all 32 completions advance in one forward per step instead of one-at-a-time generate.

**The gap to the ceiling, named.** vLLM (2170.6 tok/s) is **~33× faster** than `llm_infer` and
~35× cheaper per 1k rollouts ($0.03 vs $1.00). That gap is the cost of v1's scope, not a
defect: vLLM brings CUDA graphs, a custom in-place paged-attention kernel, and a mature
scheduler, while `llm_infer`'s `decode_many` still gathers each request's KV history into
contiguous tensors per layer per step (per-request Python loops + a `cat`) and captures no
graphs. A custom paged kernel that reads blocks in place, a vectorized gather, and graph
capture are the v2/v3 expansion path — explicitly out of v1 scope. v1's bar is *beats the
naive floor on the real rollout*, met, with the ceiling gap reported straight.

**One honest caveat on the hardware.** This run, Modal placed the two functions on different
A100-80GB variants: the engine/HF baselines on an **A100 80GB PCIe** (300 W), the vLLM ceiling
on an **A100-SXM4-80GB** (400 W) — the faster variant. Modal does not expose variant selection
for `A100-80GB`. This does not affect the v1 claim: `llm_infer` and `hf_sequential` are timed
on the **same** PCIe instance, so the 1.67× engine-vs-floor win is apples-to-apples. The vLLM
row is the uncatchable ceiling reported for context, and the ~33× gap dwarfs the few-percent
PCIe↔SXM4 difference either way.

What this buys the claim: rlvr-sql's GRPO loop spends most of its wall-clock in rollout
generation. This is the hook that connects `llm_infer`'s throughput to that loop's economics —
$/1k rollouts at a pinned GPU price, on a frozen, reproducible batch built from rlvr-sql's own
prompt builders and anchored to the shipped `grpo-s0` checkpoint.

### Pinned configuration

- **GPU:** A100-80GB, one per image, `tensor_parallel_size=1`, 1410 MHz SM clock. This run:
  engine/HF on **NVIDIA A100 80GB PCIe** (300 W), vLLM on **NVIDIA A100-SXM4-80GB** (400 W)
  — Modal-assigned, captured in the result JSON.
- **Served model:** base `Qwen/Qwen2.5-Coder-3B-Instruct` @ `488639f1ff808d1d3d0ba301aef8c11461451ec5`
  merged with rlvr-sql `grpo/grpo-s0` (PEFT LoRA, rank 32, α 64, all-linear), `merge_and_unload`,
  saved **bf16**.
- **Workload:** `tests/fixtures/rollout_grpo_s0_spider_dev.json` — Spider dev (HF `xlangai/spider`
  @ `0c350918`), dev indices `[82, 530, 621, 788, 829, 861, 976, 995]` (selection seed 0), built
  through rlvr-sql `build_messages(cot)` + `serialize_schema`; 8 prompts × G=4 = 32 completions,
  `max_completion_length=1024`, EOS `{151643, 151645}`.
- **Sampling:** temperature 1.0, top_p 1.0, seed 0 — every system re-seeds per measured iteration.
- **Timing:** 1 warmup + 2 measured iterations, median wall-clock, CUDA sync at each boundary.
- **`llm_infer`:** flash-attn backend (`2.8.3.post1`), bf16, `block_size=128`, `num_blocks=396`,
  fused `decode_many`; torch `2.12.1+cu130`, transformers `5.12.1`, numpy `2.4.6`.
- **`vllm`:** **0.23.0**, bf16, `enable_prefix_caching=False`, `gpu_memory_utilization=0.9`,
  `max_num_seqs=256`, `max_model_len=1892`, native-torch sampler, FLASH_ATTN backend; torch
  `2.11.0+cu130`, transformers `5.12.1`.
- **`hf_sequential`:** per-request `model.generate()`, SDPA attention, bf16, `do_sample=True`.
- **$/1k rollouts** at the pinned **Modal A100-80GB rate $2.50/hr** ($0.000694/s,
  modal.com/pricing, 2026-06-21); 1 rollout = 1 completion.
- **Repro:** `modal run scripts/modal_rollout.py --command rollout`.

## Running it

```bash
# one-time: merge the grpo-s0 adapter into the base, write bf16 weights to the Modal volume
modal run scripts/merge_adapter.py

# (re)build the frozen 8-prompt fixture from rlvr-sql's builders (local, no GPU)
uv run --with datasets --with pydantic python scripts/build_rollout_fixture.py

# cheap wiring + bf16-sanity check (2 completions, 64 tokens)
modal run scripts/modal_rollout.py --command smoke

# the frozen rollout-timing comparison (32 completions, ≤1024 tokens)
modal run scripts/modal_rollout.py --command rollout
```

The raw record lands in `bench-results/rollout-*.json` (git-ignored); the curated table and
pinned config are folded into the **Result** section above at land time. The sampler
correctness gate (`tests/correctness/test_sampler.py`, temperature 0 == greedy) and the
Phase A–D greedy oracle run locally with zero spend and must be green first.
