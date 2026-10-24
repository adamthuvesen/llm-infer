# Esme Serving Evaluation

`scripts/esme_serving_eval.py` runs six workloads against the Esme serving path. Each workload
stresses scheduler features together: prefix caching, chunked prefill, preemption, speculative
decode, streaming, and sampling. The harness records TTFT, queue time, preemption count, and KV
utilization per workload. It reports `tok/s` only when the workload passes the reference gate.

The harness runs on CPU by default (`--device cpu`). The A100 benchmark is in
[benchmark.md](benchmark.md).

## Run Locally

Install the local serving and test dependencies:

```bash
uv sync --extra dev --extra serving
```

Run the reference-gated harness against an Esme export bundle:

```bash
export ESME_BUNDLE_PATH=/path/to/Esme-214M-Chat
uv run python scripts/esme_serving_eval.py \
  --bundle "$ESME_BUNDLE_PATH" \
  --output bench-results/esme-serving-eval-local.json \
  --jsonl-output bench-results/esme-serving-eval-local.jsonl
```

Run selected workloads:

```bash
uv run python scripts/esme_serving_eval.py \
  --bundle "$ESME_BUNDLE_PATH" \
  --workloads shared-prefix-greedy,tight-kv-preemption,api-streaming-mixed
```

Drive an already running HTTP server:

```bash
uv run python -m llm_infer.serve \
  --backend esme \
  --bundle "$ESME_BUNDLE_PATH" \
  --preemption-policy recompute \
  --prefill-chunk-size 4 \
  --prompt-lookup-speculative \
  --prompt-lookup-max-draft-tokens 3 \
  --prompt-lookup-max-ngram-size 3

uv run python scripts/esme_serving_eval.py \
  --target external-http \
  --base-url http://127.0.0.1:8000 \
  --model esme-214m-chat \
  --workloads api-streaming-mixed,api-blocking-sampled \
  --output bench-results/esme-serving-eval-http.json
```

External HTTP mode records transport metrics but cannot observe exact engine token ids, so it
does not report reference-gated throughput.

## Serving Controls

Server-wide controls exposed by `python -m llm_infer.serve`:

- `--preemption-policy off|recompute`: strict reservation or KV-pressure eviction with
  recompute resume.
- `--prefill-chunk-size N`: cache at most `N` prompt tokens per prefill step.
- `--decode-window-size N`: decode steps per EOS/stop host sync for all-greedy batches;
  `1` restores the classic per-step decode path.
- `--prompt-lookup-speculative` / `--no-prompt-lookup-speculative`: tri-state; auto is on
  for non-CUDA serving where the backend supports it, off on CUDA.
- `--prefix-cache` / `--no-prefix-cache`: cross-turn prefix reuse of finished requests'
  prompt KV; auto is on for non-CUDA bundle backends, off on CUDA.
- `--device` / `--dtype` / `--attention-backend`: `auto` resolves locally to cpu + fp16 +
  `torch_sdpa` (the startup line labels any non-fp32 config experimental against the cpu
  fp32 reference); explicit values always win.

Per-request HTTP controls:

- `llm_infer_prefix_group_id`: extension field on `/v1/chat/completions`,
  `/v1/completions`, and `/v1/responses`; requests with the same id and exact prompt can
  share cached prefix blocks.
- `stream_options: {"include_usage": true}` on streaming chat/completions returns one final
  usage chunk before `[DONE]`. `/v1/responses` streaming reports usage on
  `response.completed`.

## Workloads

| workload | surface | shape | techniques stressed | throughput policy |
| --- | --- | --- | --- | --- |
| `shared-prefix-greedy` | `engine` | 4 greedy siblings, identical prompt | prefix caching, continuous batching | reported only if all 4 match greedy reference |
| `chunked-long-short` | `engine` | 1 long prompt + 3 short prompts | chunked prefill interleave | reported only if all 4 match greedy reference |
| `tight-kv-preemption` | `engine` | 3 greedy requests in a 3-block KV pool | preemption, recompute resume, KV utilization | reported only if all 3 match greedy reference |
| `speculative-greedy` | `engine` | repeated prompt suffixes | prompt-lookup speculative decode | reported only if all 2 match greedy reference |
| `api-streaming-mixed` | `asgi-http` / `external-http` | 3 streaming chat requests | streaming usage, prefix grouping, TTFT | local ASGI can report if reference-gated; external never reports tok/s |
| `api-blocking-sampled` | `asgi-http` / `external-http` | 2 greedy + 2 sampled completions | sampling, blocking usage, status accounting | never reports tok/s because sampled rows are not reference-gated |

## Metric Definitions

- `throughput_tokens_per_s`: `output_tokens / wall_s`, populated only when every request
  completed, every request was observed by the local engine observer, and the workload has
  `reference.status == "pass"`.
- `observed_output_tokens_per_s`: telemetry over observed output tokens. This can exist when
  `throughput_tokens_per_s` is withheld; do not use it as a speed claim.
- `TTFT`: client arrival to first emitted token. `ttft_*` is client-observed;
  `engine_ttft_*` is engine-side emission time.
- `queue_time`: client arrival to engine admission.
- `preemption_count`: sampled from the local engine or `/metrics` in external mode.
- `kv_utilization_peak` / `kv_utilization_final`: used KV blocks divided by total blocks.
- `reference.status`: `pass`, `fail`, `review` (a sampled replay group disagreed with
  itself, see below), `partial` for mixed greedy/sampled rows, `skipped` for all sampled
  rows, or `unavailable` for external HTTP.

## Sampled reference: same-shape seeded replay

Greedy rows gate against the fp32 full-recompute reference with traced bf16 ties.
Sampled rows gate on **same-shape seeded replay**. Every observed record of one request
signature must carry the identical token stream. A signature is the same prompt, sampling
params, and seed, across all measured runs and all batchmates.

Exact sampled tokens across batch shapes are not a valid target under bf16. bf16 decode
kernels produce slightly different logits per batch shape. Near a tie, that difference flips
one multinomial draw, and the flipped draw forks the whole continuation. RNG state stays
identical per request, so the divergence is shape numerics and not an RNG bug. See
[findings entry 9](findings.md) for the A100 probes behind this contract. CPU tests
(`test_sampling_batch_equivalence`, `test_decode_window`) pin the per-request seeded
generator design.

Status mapping under reference policy v2:

- replay-consistent and equal to the seeded single-request anchor: record `pass`, row
  `exact`.
- replay-consistent but anchor-divergent: record `pass_replay`, row `accepted_numerical`.
  The record stores the anchor's first mismatch as evidence (`replay_anchor_note`).
- any disagreement inside a replay group: `replay_divergent`, row `review_required`, and
  throughput is withheld (`not_reported_reference_review_required`). Admission-composition
  drift lands here, for example a straggler admitted a step late that decodes under different
  shapes. Real nondeterminism lands here too. Neither can pass silently.

The sampler unit tests pin sampling semantics (seeds, penalties, top-k, top-p). Greedy rows in
the same record keep the fp32 gate, so model math regressions still fail loudly.

## What the harness demonstrates

The full workload set shows that the public serving path can force and count real
preemptions and keep reference agreement. It also exposes prefix caching, separates
reference-gated rows from sampled rows, and reports queue time, stream token counts, and KV
utilization. This harness observes the techniques interacting on a local CPU. It does not
publish latency numbers. The per-technique GPU numbers are in
[findings entry 10](findings.md), and current throughput is in [findings entry 13](findings.md).

## Machine-Readable Output

The JSON summary contains one object per workload under `workloads[]`, with `metrics`,
`reference`, `trace`, and per-request records. The optional JSONL file emits one request
record per line with its workload and surface.

Raw outputs should stay under `bench-results/`, which is git-ignored. Curated findings belong
in this document or [benchmark.md](benchmark.md).
