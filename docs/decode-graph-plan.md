# Decode Graph Execution Plan

> **Historical — measured dead-end. Do not implement.** The static decode-graph / 32-slot
> bucket path was built and rejected: slower than eager (256 vs 299 tok/s) and it shifted
> sampled token counts (3036 vs 3026). See [`docs/scoping.md`](scoping.md) § "Measured dead-end".

This document is kept for context only.

---

# Decode Graph Execution Plan (archived)

This is the design-first tranche after the local speed loop reached a clean best of
365.8 tok/s on the frozen llm-rlvr rollout. It deliberately does not implement CUDA
graphs, a new scheduler, or a new kernel. The goal is to make the next implementation
slice small enough to measure, while still plausibly moving the rollout toward the
650 tok/s target.

## Evidence Base

Accepted best:

- Commit: `d853b9c9c125aeeb060c9691b1a0746f985c9022`
- Artifact: `bench-results/rollout-rollout-20260621T164905.json`
- Command: `modal run scripts/modal_rollout.py --command rollout`
- Hardware: NVIDIA A100 80GB PCIe, 1410 MHz, 300 W, 80 GB
- Versions: torch `2.12.1+cu130`, CUDA `13.0`, transformers `5.12.1`,
  flash-attn `2.8.3.post1`, numpy `2.4.6`
- Result: 3026 output tokens / 8.271865055 s = 365.8 tok/s, $0.1795 / 1k rollouts

Latest useful profile:

- Artifact: `bench-results/rollout-rollout-20260621T165840.json`
- Command: `modal run scripts/modal_rollout.py --command rollout --profile`
- Diagnostic buckets: decode 14.3 s, projections/MLP 4.5 s, prefill 2.5 s,
  attention 2.1 s, `kv_write` 1.5 s, `gqa_expand` 1.1 s, `kv_read_gather` 0.55 s,
  CPU/GPU sync 22 ms.

Interpretation: `kv_read_gather` is no longer the wall. The remaining gap is mostly
inside repeated decode-layer orchestration: Python builds positions, lengths, tables,
read plans, launches many small kernels, checks EOS on the host, and re-enters the same
static model graph every token. A 650 tok/s target means the accepted rollout needs to
drop from 8.27 s to about 4.66 s for the same 3026 tokens. That is too large for another
metadata tweak; it needs a captured decode execution path or a measured proof that the
current architecture has hit its ceiling.

Rejected experiments this plan must respect:

- Direct FlashAttention GQA changed sampled tokens to 3045 and regressed to 107.1 tok/s.
- Projection/MLP fusion changed sampled tokens to 2988 and only reached 368.1 tok/s.
- Step-local prompt-prefix KV copy preserved 3026 tokens but regressed to 315.7 tok/s.
- No-gather FlashAttention paged KV changed sampled tokens to 3045 and regressed to
  282.4 tok/s.

Do not retry those shapes as implementation shortcuts. The graph tranche should preserve
the current math path first, then measure whether launch/orchestration removal is enough.

## Current Decode Shape

`InferenceEngine.step()` admits waiting requests, runs per-request prefill for newly
admitted requests, batches already-prefilled requests through `QwenModel.decode_many()`,
samples one token per row, syncs EOS flags to the host, records tokens, and releases
finished requests.

`QwenModel.decode_many()` is already the right high-level unit for graph work:

1. Build per-request `positions` and `new_lengths` from Python `BlockTable.length`.
2. Reserve one token in each block table.
3. Convert the last tokens into a CUDA tensor.
4. Build RoPE rows for each request's current position.
5. Build one `KVReadPlan` with packed physical slots and cumulative sequence lengths.
6. Reuse that read plan across all layers.
7. For each decoder layer:
   - project Q/K/V,
   - apply RoPE,
   - write one K/V row per request,
   - gather packed histories with the read plan,
   - expand GQA,
   - call packed FlashAttention,
   - run output projection and MLP.
8. Update `BlockTable.length` on the host.
9. Return logits for sampling.

Commit `d853b9c` is important: it moved the read-plan construction out of the per-layer
loop. The graph-safe version should keep that property by making the read-plan buffers
preallocated inputs to the graph, not by rebuilding them inside every layer.

## Design Goal

Add a decode execution plan that can replay a captured decode step for a static slot
bucket. The first implementation should target the frozen rollout shape, because that is
the campaign metric:

- 8 prompts x G=4 = 32 completions
- all requests admitted at once in the current benchmark pool sizing
- `max_completion_length=1024`
- `temperature=1.0`, `top_p=1.0`, seed `0`
- current accepted sampled length: 3026 output tokens

The first graph slice should not try to solve full production scheduling. It should add a
specialized, opt-in graph path for a stable decode bucket, then fall back to the existing
engine when the bucket contract is not met.

## Static Decode Buckets

Define a `DecodeBucket` by the tensors whose shapes must stay fixed across replay:

- `slot_count`: number of request slots represented in the graph.
- `max_total_kv_tokens`: packed-history capacity for the bucket.
- `max_seq_len`: maximum per-request history length in the bucket.
- `block_size`: current cache block size, 128.
- model dtype, backend type, device, and vocabulary shape.

Recommended bucket sizes:

- First measured slice: `slot_count=32`, the frozen rollout's full batch.
- General follow-up: powers or vLLM-like capture sizes such as 1, 2, 4, 8, 16, 24, 32.
- Do not capture every active count until the 32-slot slice proves a real win.

Requests enter a bucket after prefill has completed and each request has:

- a fixed slot id in the bucket,
- a `BlockTable`,
- a prompt length,
- a first sampled token stored in the bucket's input-token buffer,
- `remaining_budget > 0`,
- the common EOS set used by the rollout.

Requests leave a bucket only at graph boundaries. For the first slice, keep all 32 slots in
the 32-slot bucket until the rollout batch is done; mark completed slots inactive on device
and defer block freeing until the host drains the graph run. This may run a small amount of
extra work after a request finishes, but it avoids recapture churn and per-token host sync.
If wasted inactive-slot work dominates, that is evidence for adding smaller buckets later.

For a general scheduler:

- Admit and prefill on the host path.
- Move ready requests into the largest bucket that fits the active set.
- Rebucket only between decode steps when the active count crosses a configured threshold,
  or when cancellation/admission makes the current bucket too wasteful.
- Never move a request between buckets mid-replay.

## Graph-Safe Tensor Plan

Create a typed structure, for example `DecodeGraphPlan`, that owns preallocated tensors.
The first version should live close to the engine/model boundary, not buried in the
benchmark script.

Suggested fields:

- `input_tokens`: `torch.long`, shape `(slot_count,)`. Last generated token for each slot.
- `positions`: `torch.int64` or `torch.float32`, shape `(slot_count,)`. Absolute decode
  position before writing the new K/V row.
- `new_lengths`: `torch.int32`, shape `(slot_count,)`. `positions + 1`.
- `active_mask`: `torch.bool`, shape `(slot_count,)`. True for slots that still count.
- `finished_mask`: `torch.bool`, shape `(slot_count,)`. Written by the graph or post-graph
  device work, read by the host only at polling boundaries.
- `slot_request_ids`: host-only list mapping slot index to request id.
- `write_slots`: `torch.long`, shape `(slot_count,)`. Physical cache slot for the new K/V
  row in each request.
- `read_idx`: `torch.long`, shape `(max_total_kv_tokens,)`. Packed physical slots for
  history gather. This is the graph-safe form of `KVReadPlan.idx`.
- `cu_seqlens`: `torch.int32`, shape `(slot_count + 1,)`. Graph-safe form of
  `KVReadPlan.cu_seqlens`.
- `lengths`: `torch.int32`, shape `(slot_count,)`. Per-slot sequence lengths. Keep the
  Python list only as debug metadata outside capture.
- `max_len_scalar`: host integer baked into a graph key, equivalent to `KVReadPlan.max_len`.
- `packed_token_count`: host integer baked into a graph key, or a device scalar if later
  kernels support masking padded packed histories.
- `output_logits`: dtype model logits, shape `(slot_count, vocab_size)`, optional if sampling
  is captured immediately.
- `output_tokens`: `torch.long`, shape `(slot_count,)`. Sampled next tokens.
- `eos_token_ids`: `torch.long`, shape `(num_eos,)`. For the rollout this is `[151643,
  151645]`.
- `eos_flags`: `torch.bool`, shape `(slot_count,)`. Device-side EOS result for each slot.
- `generated_counts`: `torch.int32`, shape `(slot_count,)`. Number of generated tokens so far,
  for length-cap completion without host inspection.

The current `KVReadPlan` should evolve from a dataclass of freshly allocated tensors into
either:

- a view over these preallocated buffers, or
- a method that fills these buffers in place from `BlockTable` state before replay.

The important invariant from `d853b9c` stays the same: build or fill the read plan once per
decode step, then reuse it across every layer.

## Capture Boundary

The first captured function should be narrower than `InferenceEngine.step()`:

```text
input_tokens, positions, read/write plan tensors, active mask
  -> Qwen decode_many_graph(...)
  -> sample_many(...)
  -> output_tokens, eos_flags
```

Capture inside a new model/engine entrypoint, not around the whole scheduler. Keep prefill,
admission, request allocation, and final result materialization outside the graph.

Capture candidates:

- embedding lookup for `input_tokens`,
- RoPE table calculation for `positions`,
- all decoder layers,
- packed KV write/read using preallocated indices,
- GQA expansion,
- packed FlashAttention,
- logits projection,
- sampling for `temperature=1.0`, `top_p=1.0`,
- EOS flag calculation.

Leave dynamic for the first slice:

- prompt prefill,
- block allocation and physical slot list construction,
- filling `read_idx`, `cu_seqlens`, `lengths`, and `write_slots`,
- request admission/release,
- cancellation,
- Python result lists,
- final token materialization.

This split still removes the repeated Python/layer launch path if the model decode and
sampling graph replay as one captured unit. If PyTorch cannot capture `torch.multinomial`
with the required generator behavior, capture decode through logits only and keep sampling
outside the graph as an explicit measured fallback. That fallback must still preserve the
3026-token path before any speed claim counts.

## Recapture And Cache Invalidation

Graph cache key:

```text
(slot_count, max_total_kv_tokens, max_len_scalar, block_size, dtype, backend class,
 model device, sampling mode, temperature, top_p, eos_token_count)
```

Recapture when:

- active bucket size changes to an uncached `slot_count`,
- `max_total_kv_tokens` exceeds the captured buffer capacity,
- `max_len_scalar` exceeds the captured attention maximum,
- model dtype/backend/device changes,
- sampling mode or EOS token count changes,
- CUDA allocation addresses for plan-owned tensors change.

Do not recapture when:

- token ids change,
- positions change within capacity,
- physical KV slots change within the preallocated `read_idx`/`write_slots` buffers,
- active rows finish but stay represented by `active_mask`.

For the first frozen-rollout slice, capture once after warmup for the 32-slot bucket and
reuse it through measured iterations. A failed recapture/cache miss during timed iterations
invalidates the speed claim unless it is explicitly reported as part of headline timing.

## Preserving Read-Plan Reuse

The current `cache.plan_read_many(tables, new_lengths)` does useful work once per decode
step:

- linearizes each request's block table into physical slots,
- builds one packed `idx`,
- builds cumulative lengths,
- records `max_len`.

The graph-safe equivalent should keep the exact same logical plan but write into stable
buffers:

```text
DecodeGraphPlan.fill_from_requests(tables, positions)
  - reserve one slot per active request outside capture
  - write physical_slot(position) into write_slots[slot]
  - write physical_slots(0, new_length) into read_idx[offset:offset + new_length]
  - write cumulative lengths into cu_seqlens
  - write positions/new_lengths/lengths
```

Inside the captured model, every layer uses the same `read_idx` and `cu_seqlens` buffers.
That preserves the `d853b9c` win. The implementation should add tests that assert one
plan fill serves all layers and that the packed histories match `read_many_plan()` for the
same requests.

One subtlety: `flash_attn_varlen_func` takes `max_seqlen_k` as a Python integer today. For
graph replay, either capture per `max_len_scalar` bucket, or round `max_len_scalar` up to a
small fixed ladder. Do not pass a changing Python `max(read_plan.lengths)` into a graph and
pretend it is static.

## EOS, Admission, Cancellation, Completion

Current EOS handling syncs every decode step for the common EOS set:

```python
mask = (flat.unsqueeze(-1) == eos).any(dim=-1)
return [bool(flag) for flag in mask.cpu().tolist()]
```

For the graph path:

- Compute `eos_flags` on device: `(output_tokens[:, None] == eos_token_ids[None, :]).any(-1)`.
- Compute length-cap completion on device: `generated_counts + 1 >= max_new_tokens`.
- Update `finished_mask |= active_mask & (eos_flags | length_done)`.
- Feed a safe token for inactive rows on the next replay. For example, keep their last token
  but mask their contribution to output accounting; do not append inactive-row tokens.
- Host polls `finished_mask` only at configured graph boundaries.

First slice policy:

- No mid-run admission after the initial 32 requests. This matches the frozen rollout because
  the cache pool admits the whole batch at once.
- No cancellation.
- No block freeing until graph run completion.
- Host may poll completion every replay only if the poll is outside the captured section and
  explicitly measured. The stronger target is polling every `N` steps or using a device-side
  `all_done` flag, then syncing only when the graph loop might terminate.

General scheduler policy after the first slice:

- Admit only between graph replays.
- Cancel by marking `active_mask[slot] = False` at the next boundary, then rebucket or leave
  the row inactive until bucket exit.
- Release blocks only after the host has observed completion/cancellation and the graph is no
  longer reading that slot's request state.
- If a waiting request can reuse a completed slot, do it only after resetting that slot's
  plan buffers and generated counters outside capture.

Correctness rule: EOS tokens remain included in generated output, as `Request.record()` does
today. Device-side EOS only changes when the host learns the fact, not what is generated.

## Sampling And Token Validation

The frozen rollout is sampled, so exact cross-system token equality is not expected. But
within llm-infer, this tranche must preserve the accepted local path:

- same workload fixture,
- same merged grpo-s0 model,
- same seed,
- same request order,
- same admission schedule,
- same `temperature=1.0`, `top_p=1.0`,
- same EOS set,
- same final total output tokens: 3026.

Acceptance gate for a graph implementation:

1. Existing greedy correctness tests pass.
2. Modal smoke passes fp32 truth agreement for the greedy benchmark.
3. Frozen rollout produces `total_output_tokens == 3026` for `llm_infer`.
4. If token count differs, the experiment is rejected unless a deliberate sampling-order
   change is documented and Adam accepts a new baseline. Default is reject.
5. If token count matches but text differs, record a request-level diff for the first two
   completions; do not block only on sampled text unless output length or correctness gates fail.

If sampling moves inside a CUDA graph, be especially strict. Captured RNG behavior must be
proved by repeated measured iterations producing the same output lengths and by comparing the
full `outputs` dict against the current accepted path. If PyTorch graph RNG cannot preserve
that path cleanly, leave sampling outside the graph for the first slice and measure decode-only
capture.

## Benchmark And Profile Evidence

No future speed claim counts without this evidence table:

| Field | Required value |
| --- | --- |
| Branch and commit | local commit SHA under test |
| Command | exact Modal command |
| Artifact | `bench-results/...json` path |
| Hardware | GPU name, clocks, power limit, memory |
| Versions | torch, CUDA, transformers, flash-attn, vLLM if present, numpy |
| Workload | prompt count, G, max tokens, seed, sampling config, EOS ids |
| Wall-clock | median seconds |
| Output tokens | llm-infer total, must be 3026 for accepted rollout path |
| tok/s | derived from artifact |
| Cost | `$ / 1k rollouts` from artifact |
| Correctness gates | local pytest, smoke truth gate, token-count gate |
| Profile | separate `--profile` artifact; diagnostic only |

Minimum commands for a candidate implementation:

```bash
uv run ruff check
uv run python -m pytest tests/correctness/test_batched_decode.py tests/correctness/test_paged_decode.py tests/correctness/test_sampler.py -q
uv run python -m pytest tests/kv_cache/test_paged_kv_cache.py tests/benchmarks/test_runner_profile.py -q
modal run scripts/modal_benchmark.py --command smoke
modal run scripts/modal_rollout.py --command rollout
modal run scripts/modal_rollout.py --command rollout --profile
```

Run `modal run scripts/modal_benchmark.py --command bench` if the graph path also claims a
greedy benchmark improvement. Keep profile runs separate from headline timing, as protected
by `tests/benchmarks/test_runner_profile.py`.

## Rollback Criteria

Revert the implementation slice if any of these happen:

- Modal smoke fails the greedy truth gate.
- Frozen rollout `llm_infer` token count is not 3026.
- Frozen rollout speed is below the accepted 365.8 tok/s on comparable A100 hardware.
- Speed improves by less than 10 percent and the code adds graph/scheduler complexity.
- The graph path requires disabling existing correctness tests.
- Timed iterations include graph capture/recapture without reporting it.
- Memory use forces lower batch admission for the frozen rollout.
- The graph path silently changes EOS inclusion, max-token stopping, or request order.

Keep a marginal implementation only if it is a clean measurement scaffold that proves a
ceiling and is explicitly labeled diagnostic, not a headline speed path.

## Risks

- CUDA graph capture may reject dynamic allocations inside PyTorch ops, FlashAttention, or
  `torch.multinomial`.
- `flash_attn_varlen_func` static arguments may force too many graph keys as sequence
  lengths grow.
- Keeping inactive rows in a fixed 32-slot graph may waste enough compute to erase the
  launch savings.
- Captured sampling may advance RNG differently from eager sampling.
- Device-side completion without per-token host sync may make debugging harder.
- Preallocating maximum packed histories may increase memory enough to disturb cache sizing.
- A graph path can accidentally optimize only the frozen benchmark shape; keep the fallback
  eager path clear and tested.

## Smallest Implementation Slice

First slice: an opt-in graph-safe decode plan for the frozen rollout's 32-slot bucket, with
sampling outside the graph if needed.

Expected files touched:

- `llm_infer/serving/engine.py`: route eligible full-batch sampled rollout runs to a graph
  decode loop; keep eager fallback.
- `llm_infer/model/qwen.py`: add `decode_many_graph_inputs(...)` or a similarly explicit
  entrypoint that consumes preallocated plan tensors instead of Python `BlockTable` lists.
- `llm_infer/kv_cache/paged_kv_cache.py`: add graph-plan buffer filling beside
  `plan_read_many()`, preserving the current eager method.
- `llm_infer/serving/request.py`: add host boundary helpers for bulk materialization if the
  graph loop stores output tokens in a tensor slab.
- `llm_infer/serving/sampler.py`: only if sampling is captured or a deterministic graph-safe
  sampling boundary is needed.
- `llm_infer/benchmarks/runners.py`: expose an opt-in flag/config record for graph decode;
  default should remain eager until evidence earns it.
- `scripts/modal_rollout.py`: add a pinned flag only after local tests pass, for example
  `--graph-decode`; do not make it the default until it beats the accepted path.
- New tests under `tests/kv_cache/`, `tests/correctness/`, and `tests/benchmarks/`.

Suggested test list:

- Unit: graph-plan buffer fill equals `plan_read_many()` for varied block tables and lengths.
- Unit: write slots match `BlockTable.physical_slot(position)` for each active row.
- Unit: inactive slots do not append generated tokens.
- Unit: EOS flags include EOS token and stop further output accounting.
- Correctness: graph-disabled path remains byte-for-byte current behavior.
- Correctness: graph-enabled greedy small batch matches eager greedy on CPU-skipped/CUDA-gated
  test, or skips cleanly without CUDA.
- Benchmark contract: profile remains outside headline timing for graph mode.

Implementation sequence:

1. Add graph-plan dataclasses and buffer-fill tests, no CUDA graphs yet.
2. Add a graph-compatible eager replay path that reads from the preallocated buffers and
   proves identical greedy/sample token counts to the current path.
3. Add CUDA graph capture around decode-through-logits for the 32-slot bucket.
4. Measure smoke and frozen rollout.
5. Only then consider moving sampling/EOS into the graph.

This sequence makes the first failure cheap and interpretable. If buffer replay alone is
wrong, the graph is not the problem. If buffer replay is correct but graph capture does not
move speed, the measured ceiling points to larger kernel fusion or true paged attention work.
