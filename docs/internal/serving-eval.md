# Esme Serving Evaluation

This harness shows how the Esme serving path behaves when scheduler features interact under
load. It is not a replacement for the A100 benchmark in [../benchmark.md](../benchmark.md). Any
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
- `--prompt-lookup-speculative`: enable prompt-lookup speculative decode when supported.

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
- `reference.status`: `pass`, `fail`, `partial` for mixed greedy/sampled rows, `skipped` for
  all sampled rows, or `unavailable` for external HTTP.

## What the harness demonstrates

Local CPU runs of the full workload set (last verified `2026-06-30`, before the decode
window became the engine default — absolute latencies from that run are not current) show
the public serving path can force and count real preemptions, preserve reference
agreement, expose prefix caching, distinguish reference-gated rows from sampled rows, and
report queue time, stream token counts, and KV utilization. The published per-technique
GPU numbers live in [../benchmark.md](../benchmark.md)'s technique gallery; this harness is
the local, no-spend way to observe the same techniques interacting.

## Machine-Readable Output

The JSON summary contains one object per workload under `workloads[]`, with `metrics`,
`reference`, `trace`, and per-request records. The optional JSONL file emits one request
record per line with its workload and surface.

Raw outputs should stay under `bench-results/`, which is git-ignored. Curated findings belong
in this document or [../benchmark.md](../benchmark.md).
