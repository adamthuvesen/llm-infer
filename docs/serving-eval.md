# Esme Serving Evaluation

This harness shows how the Esme serving path behaves when scheduler features interact under
load. It is not a replacement for the A100 benchmark in [../benchmark.md](benchmark.md). Any
`tok/s` field is reported only when the workload passes the local reference gate.

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
  itself — see below), `partial` for mixed greedy/sampled rows, `skipped` for all sampled
  rows, or `unavailable` for external HTTP.

## Sampled reference: same-shape seeded replay (2026-07-11)

Greedy rows gate against the fp32 full-recompute reference with traced bf16 ties, unchanged.
Sampled rows gate on **same-shape seeded replay**: every observed record of one request
signature (same prompt, sampling params, and seed — all measured runs and all batchmates)
must carry the identical token stream.

Why the old gate was wrong: until 2026-07-10 a sampled record had to reproduce the seeded
*single-request* stream exactly. That is a cross-shape exactness requirement, and bf16
decode kernels legitimately produce slightly different logits per batch shape; one flipped
multinomial draw then forks the whole continuation. The phase0 `sampled-b8`/`sampled-b64`
rows failed this gate from their first run (2026-07-09, both grouped A/B arms) while every
batchmate agreed token-for-token with every other batchmate and every measured run — the
signature of shape numerics, not an RNG bug. The per-request seeded generator design is
separately pinned by CPU tests (`test_sampling_batch_equivalence`, `test_decode_window`).

Status mapping under reference policy v2:

- replay-consistent and equal to the seeded single-request anchor → record `pass`, row
  `exact`;
- replay-consistent but anchor-divergent → record `pass_replay`, row `accepted_numerical`,
  with the anchor's first mismatch stored as evidence (`replay_anchor_note`);
- any disagreement inside a replay group → `replay_divergent`, row `review_required`,
  throughput withheld (`not_reported_reference_review_required`). Admission-composition
  drift (a straggler admitted a step late decodes under different shapes) and real
  nondeterminism both land here; neither may pass silently.

Cross-shape exact sampled tokens are explicitly **not** required (performance roadmap,
Phase 5 correctness checks). Sampling semantics (seeds, penalties, top-k, top-p) are pinned
by the sampler unit tests, and greedy rows in the same record keep the fp32 gate, so model
math regressions still fail loudly.

## What the harness demonstrates

Local CPU runs of the full workload set (last checked `2026-06-30`, before the decode
window became the engine default, so absolute latencies from that run are not current) show
the public serving path can force and count real preemptions, preserve reference
agreement, expose prefix caching, distinguish reference-gated rows from sampled rows, and
report queue time, stream token counts, and KV utilization. The published per-technique
GPU numbers live in [../benchmark.md](benchmark.md)'s technique gallery; this harness is
the local, no-spend way to observe the same techniques interacting.

## Machine-Readable Output

The JSON summary contains one object per workload under `workloads[]`, with `metrics`,
`reference`, `trace`, and per-request records. The optional JSONL file emits one request
record per line with its workload and surface.

Raw outputs should stay under `bench-results/`, which is git-ignored. Curated findings belong
in this document or [../benchmark.md](benchmark.md).
