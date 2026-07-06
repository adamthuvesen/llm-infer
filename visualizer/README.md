# KV Cache Observatory

Static local visualizer for schema-v3 `InferenceEngine(trace=...)` JSONL traces. It has no
runtime dependencies and fetches no network assets: plain HTML, CSS, and ES modules.

The interface replays a trace frame by frame: a hero player with a scrubber and speed
control, a live telemetry strip (engine step, occupancy, throughput, generated tokens), a
per-request decode timeline, the paged KV cache wall, a current-event inspector, and the raw
event stream. Color encodes engine semantics: blue admit/schedule, teal prefill, amber
decode, violet speculative, rose preempt/resume, green finish.

## Open Locally

```bash
python -m http.server 8765
```

Then open `http://localhost:8765/visualizer/`.

The viewer loads the committed fixture at `docs/assets/kv_trace_schema_v3.jsonl` when served
over HTTP. Use `Load JSONL` to inspect another schema-v3 trace.

## Regenerate The Fixture

```bash
uv run python scripts/generate_kv_trace_fixture.py
```

The generator is a standalone synthetic simulation (no engine/model/torch import) that emits the
same schema-v3 event shapes `InferenceEngine(trace=...)` produces: admission, chunked prefill,
prefix sharing, batched decode, clear block lifecycle, recompute preemption, and finish. The
viewer ships on its own. The injected clock only makes throughput sample fields stable in git.
The same viewer equally replays a real `InferenceEngine(trace=..., preemption=True)` trace.

## Replay A Real Esme Run

```bash
uv run python scripts/generate_esme_kv_trace.py --write
```

This drives the **actual** `InferenceEngine` over the tiny `Esme-214M-Chat`-format bundle with
preemption on and writes `docs/assets/esme_kv_trace_schema_v3.jsonl`. Every event is emitted by
the real engine/scheduler/allocator on the Esme backend, including a forced recompute preemption.
Load it with `Load JSONL` to watch a real Esme run. A fixed step clock keeps the artifact
byte-stable so it can be regression-checked.

## What The Viewer Shows

- request lanes from `request_admitted`, `prefill_chunk_progress`, `decode_step`, and
  `request_finished`
- chunk ranges from `start_pos`, `end_pos`, `cached_tokens`, and `total_prompt_tokens`
- batch and waiting signals from `batch_size_changed`
- throughput samples from `tokens_per_second_sampled`
- the paged KV wall from real `block_allocated` / `block_freed` events: filled blocks are
  physically held right now, with the scheduler's `reserved_blocks` shown as fainter headroom
- logical cache footprint from observed cached/generated token counts

Block allocation/free is emitted clearly from the allocator boundary: a `block_allocated`
fires only when a block leaves the free pool (copy-on-write included), a `block_freed` fires only
when a block truly returns (refcount-0), and a prefix-shared block retained by a sibling is freed
once, by its last owner.

Speculative traces are rendered clearly when present: if one `decode_step` has one
`request_id` and multiple `token_ids`, the viewer keeps those tokens together as a burst on
that request lane. The committed fixture is non-speculative.

Preemption is rendered when present: a `request_preempted` puts a slashed "evicted" notch on the
request's lane (and the KV wall shows its blocks freed at the same moment, via the matching
`block_freed`), and a later `request_resumed` marks where it re-enters and recomputes its KV.
Its replayed prefill chunks are the recompute. The committed fixture includes one such
preempt + resume cycle (`chat-quick` under a tight pool).
