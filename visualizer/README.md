# KV Trace Theater

Static local visualizer for schema-v2 `InferenceEngine(trace=...)` JSONL traces. It has no
runtime dependencies and does not fetch network assets.

## Open Locally

```bash
python -m http.server 8765
```

Then open `http://localhost:8765/visualizer/`.

The viewer loads the committed fixture at `docs/assets/kv_trace_schema_v2.jsonl` when served
over HTTP. Use `Load JSONL` to inspect another schema-v2 trace.

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
- KV/cache pressure from scheduler `reserved_blocks` and observed logical cached/generated
  token counts

Block allocation/free lifecycle is intentionally absent because schema v2 does not emit
`block_allocated` or `block_freed`. Add those only after request-aware cache hooks exist.

Speculative traces are rendered honestly when present: if one `decode_step` has one
`request_id` and multiple `token_ids`, the viewer keeps those tokens together as a burst on
that request lane. The committed fixture is non-speculative.
