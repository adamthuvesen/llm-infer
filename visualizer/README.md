# KV Cache Observatory

Static local visualizer for schema-v3 `InferenceEngine(trace=...)` JSONL traces. It has no
runtime dependencies and does not fetch network assets — plain HTML, CSS, and ES modules.

The interface replays a trace frame by frame: a hero player with a scrubber and speed
control, a live telemetry strip (engine step, occupancy, throughput, generated tokens), a
per-request decode timeline, the paged KV cache wall, a current-event inspector, and the raw
event stream. Color encodes engine semantics — blue admit/schedule, teal prefill, amber
decode, violet speculative, green finish.

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

The generator uses a small deterministic model-shaped object, but it drives the real
`InferenceEngine(trace=...)`, scheduler, request state, paged KV cache, chunked prefill, decode,
and finish path. The injected clock only makes throughput sample fields stable in git.

## What The Viewer Shows

- request lanes from `request_admitted`, `prefill_chunk_progress`, `decode_step`, and
  `request_finished`
- chunk ranges from `start_pos`, `end_pos`, `cached_tokens`, and `total_prompt_tokens`
- batch and waiting signals from `batch_size_changed`
- throughput samples from `tokens_per_second_sampled`
- the paged KV wall from real `block_allocated` / `block_freed` events: filled blocks are
  physically held right now, with the scheduler's `reserved_blocks` shown as fainter headroom
- logical cache footprint from observed cached/generated token counts

Block allocation/free is now emitted honestly from the allocator boundary: a `block_allocated`
fires only when a block leaves the free pool (copy-on-write included), a `block_freed` fires only
when a block truly returns (refcount-0), and a prefix-shared block retained by a sibling is freed
once, by its last owner.

Speculative traces are rendered honestly when present: if one `decode_step` has one
`request_id` and multiple `token_ids`, the viewer keeps those tokens together as a burst on
that request lane. The committed fixture is non-speculative.
