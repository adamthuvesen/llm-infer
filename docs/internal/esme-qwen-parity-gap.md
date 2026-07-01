# Historical Esme/Qwen serving-technique parity gap map

Archived Phase 0 of the Esme-Qwen-parity mission, kept for context after Esme became the
repo story. At the time, Qwen was the comparator for what the serving-technique
role looked like. For each surface this recorded Esme's status
— **works** / **qwen-only** / **missing** / **decision-needed** — with `file:line`
evidence, so the closing work builds on what already landed at `e135f5d` (real
paged KV, prefix caching/speculative/preemption capability flags + unit tests)
rather than re-doing it.

## Headline finding

The serving *plumbing* is already backend-agnostic. The engine, async engine,
scheduler, sampler, tracer, and HTTP app all run against the `CausalLMBackend`
protocol and never branch on Qwen vs Esme. `PretrainBundleModel` implements the
**same** method surface as `QwenModel` (`prefill`, `prefill_chunk`, `decode_one`,
`decode_many`, `decode_tokens`, `logits`, `release_table`), and
`load_model_runtime("esme"|"dense", bundle_path=...)` returns a fully-populated
`ModelRuntime` with `DENSE_CAPABILITIES == QWEN_CAPABILITIES`.

At this snapshot, the remaining gap was **not** engine code. It was **evidence and ergonomics**:
the serving-path/loadgen/visualizer/benchmark proof for Esme is thinner or
Qwen-shaped, and one doc line is stale. The closing work is tests + a loadgen
affordance + a trace artifact + doc truth, not a new backend.

## Surface-by-surface

### `llm_infer/serve.py` — end-to-end serving — **works**
- `--backend` accepts every registered backend, including `esme`/`dense`
  (`serve.py:67`); `--bundle` defaults to `$LLM_INFER_BUNDLE`
  (`serve.py:68-74`); `load_model_runtime(args.backend, ..., bundle_path=args.bundle)`
  is backend-agnostic (`serve.py:90-97`).
- `build_app_from_runtime` wires any runtime's `model`/`capabilities`/`tokenizer`/
  `eos_token_ids` into the app (`serve.py:27-52`). Esme already serves: `test_runtime.py`
  has `test_dense_runtime_drives_serving_path` (`tests/model/test_runtime.py:103`) and
  `test_dense_runtime_eos_metadata_drives_serving_finish_reason` (`:124`) driving the
  real bundle loader through `TestClient`.
- **Gap (evidence, not code):** the acceptance gate wants a test **under
  `tests/serving/` named with an `esme` token** proving Esme serves end-to-end
  through the async engine. Today the bundle-serving tests live in `tests/model/`
  and are `dense`-tokened. → add `tests/serving/test_esme_serving.py`.
- Closed: `serve.py` accepts `ESME_BUNDLE_PATH` first, then `LLM_INFER_BUNDLE`, so the
  documented Esme bundle env works for the server.

### `llm_infer/serving/server/` — async engine / app / streaming / detok — **works**
- `AsyncInferenceEngine` is constructed around an injected `InferenceEngine`
  (`async_engine.py:52-73`); the docstring's "real Qwen in production" is a comment,
  not a code dependency (`async_engine.py:47-49`). Gauges bind to generic engine/
  scheduler/allocator state (`async_engine.py:108-126`).
- `app.py` is "transport only" over an injected engine + tokenizer (`app.py:1`,
  `:58-72`); chat template via `tokenizer.apply_chat_template` (`app.py:344-353`),
  which the bundle tokenizer implements (`runtime.py:64-72`).
- No Qwen assumption anywhere in the server. **Status: works**; covered by the new
  `tests/serving` esme test rather than new server code.

### `llm_infer/serving/` engine + `llm_infer/scheduler/` — prefix caching + preemption under load — **mixed**
- Engine and scheduler are generic; `_infer_capabilities` returns `DENSE_CAPABILITIES`
  for `PretrainBundleModel` (`engine.py:213-216`). Prefix caching is gated on
  `capabilities.prefix_caching` (`engine.py:119-122`), preemption on `paged_kv`
  (`engine.py:74-78`). Real recompute-preemption + batched decode are wired
  (`engine.py:177-198`, `engine_decode.py:119-144`).
- **Unit capability coverage exists for the bundle:** `test_dense_capabilities.py`
  runs prefix sharing (`:43`), speculative (`:63`), and preemption (`:80`) on the
  tiny bundle, each asserting `== greedy_decode(...)` recompute reference. Real-bundle
  paged greedy/batched parity is in `test_esme_paged_parity.py` (`:196-247`).
- **Gap (evidence):** prefix caching + preemption are exercised on the bundle at the
  **engine** level but not through the **serving/scheduler path under concurrent
  load** the way the mission frames it, and not against the real Esme bundle through
  serving. The capability tests are single-request `engine.run()`. → add a serving-path
  test that drives multiple concurrent Esme requests through the async engine with a
  prefix group and a tight KV pool that forces preemption, each verified against the
  per-sequence recompute reference. (Preemption-under-load on `PretrainBundleModel`
  is otherwise only synthetic/Qwen: `test_preemption.py` uses `_tiny_qwen`.)

### `llm_infer/serving/speculative.py` + `engine_decode.py` — speculative — **works (no draft model needed)**
- Speculation is **prompt-lookup** (`speculative.py:1`, `PromptLookupDraft`,
  `:26-50`): the draft is copied from the request's own prompt/history, so there is
  **no separate draft model** — the "decision-needed draft model" risk does **not**
  apply to Esme.
- Verification runs through `model.decode_tokens(...)` (`engine_decode.py:178`), which
  `PretrainBundleModel` implements identically to Qwen (`pretrain_bundle.py:299-335`).
  The greedy verifier + safe fallback is backend-neutral.
- `test_dense_capabilities.py::test_dense_accepts_speculative_init` (`:63`) already
  proves speculative on the bundle matches recompute. **Status: works.** Add an
  explicit Esme speculative parity assertion if it strengthens the gate, but no
  decision is blocked. → **DONE-class**, optionally reinforced.

### `scripts/loadgen.py` — drive load against Esme — **closed**
- Historical gap: the pure HTTP client used to default to
  `Qwen/Qwen2.5-Coder-3B-Instruct`, which failed against an Esme server unless callers
  passed `--model esme-214m-chat`.
- Current state: `--model` defaults to `esme-214m-chat`, the example starts with Esme, and
  `tests/serving/test_loadgen.py` drives the in-process Esme app by its served model id.

### `llm_infer/tracing.py` + `visualizer/` — Esme trace renders — **works (engine) / no Esme artifact**
- Tracing is engine-level and backend-agnostic (`tracing.py:1-10`, `engine.py:99-105`);
  any backend run with `InferenceEngine(trace=...)` emits schema-v3 events.
- The committed visualizer fixture is a **standalone synthetic generator** that does
  not import the engine or torch (`generate_kv_trace_fixture.py:1-25`), so it is not an
  Esme run. The visualizer loader test (`visualizer/trace_loader.test.mjs`) validates
  schema, not provenance.
- **Gap (evidence):** there is no demonstrated **Esme** trace artifact. → add a test
  that runs the real/tiny bundle through `InferenceEngine(trace=...)` and asserts it
  emits a well-formed schema-v3 stream the visualizer loader accepts, and produce a
  small committed Esme trace artifact for the report. (Loader untouched, so the
  `node --test` gate is informational unless I touch loader files.)

### `llm_infer/benchmarks/` + `scripts/modal_esme_*` — benchmark evidence + vLLM story — **closed**
- Historical state: Qwen had the public three-way table, while Esme had only
  reference-gated paged-vs-recompute evidence.
- Current state: Esme has both the paged-vs-recompute record and the A100 three-way
  naive HF / `llm_infer` / vLLM benchmark through the converted HF checkpoint
  (`scripts/modal_esme_three_way.py`, `llm_infer/benchmarks/esme_three_way.py`,
  `docs/benchmark.md`).

## Stale doc fixed
- The old scoping defect that said Esme did not implement true paged KV is fixed in
  `docs/scoping.md`; Esme is now documented as the primary paged-KV path.

## What closing parity concretely means (acceptance → plan)
| Gate | Status today | Action |
| --- | --- | --- |
| Esme serves end-to-end, `tests/serving` + `esme` token | works in code; test is `dense`-tokened in `tests/model` | add `tests/serving/test_esme_serving.py` |
| loadgen targets Esme | closed: Esme model id is the default and covered by `tests/serving/test_loadgen.py` | keep Esme first in examples |
| prefix caching + preemption through serving path, recompute-verified | engine-level unit tests only; preempt-under-load is qwen/synthetic | add concurrent Esme serving-path test (prefix group + tight pool → preempt), verified vs recompute |
| speculative: participate or decision-needed | works (prompt-lookup, no draft model) | DONE; optionally add explicit Esme speculative parity assertion |
| Esme KV trace renders | engine emits; fixture is synthetic | add Esme `InferenceEngine(trace=...)` test + committed artifact |
| Esme external baseline | closed: Esme three-way A100 benchmark exists through converted HF checkpoint | keep Esme first in benchmark docs |

## Decisions surfaced (not silently resolved)
- **vLLM packaging for Esme** — resolved by converting the bundle to a native
  `Qwen3ForCausalLM` checkpoint and parity-gating it against the source bundle oracle.
- **Speculative draft model** — **not needed.** Esme speculation is prompt-lookup, no
  external draft model, so there is no blocker to surface.
- **GPU Esme benchmark** — landed as the A100 Esme three-way result in `docs/benchmark.md`.
