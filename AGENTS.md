# AGENTS.md: llm-infer

`llm-infer` is a small inference engine that builds real serving techniques and measures them
clearly. `Esme-214M-Chat` is the headline model. Qwen2.5-Coder-3B-Instruct is the independent
HuggingFace reference for correctness regression. Read [`docs/scoping.md`](docs/scoping.md)
before you propose work. [`docs/architecture.md`](docs/architecture.md) maps modules, backends,
and the reference contract. [`docs/benchmark.md`](docs/benchmark.md) holds the benchmark method.

## Judgment Calls

- Speed needs a reference. Keep raw timing in every measured row. A public tok/s claim needs a
  qualified fp32 reference row (`headline_eligible`). Label a path experimental until it passes.
- The single-request unit path needs exact token ids.
- Keep fp32 reference status separate from direct candidate/baseline parity. Exact direct parity
  can support a relative A/B result even when both paths share a numerical review.
- A difference above the 0.1-logit bf16 bound needs a durable review. `docs/findings.md` entry 20
  says when a review may cite the measured bf16 error.
- Do not rerun a settled experiment only to move its record to a newer policy format.
- Modal runs bill until they finish. Never add `--approved` or `LLM_INFER_SERVE_APPROVED=1`
  yourself: when a run refuses, stop and ask Adam, quoting the printed command.
- Run one GPU job at a time unless the experiment needs more.
  Whoever starts a background run watches its log and raises an alarm after 10 minutes of silence.

## Commands

Use Python 3.13, [`uv`](https://docs.astral.sh/uv/), and Node 24. CI runs the same `make check`.

```bash
make install
make check        # import contracts, lint, format, types, CPU tests, evidence, visualizer
make fix         # ruff fixes and format
make help         # every target with its purpose
make slow         # opt-in 3B Qwen reference check and figure regeneration
```

Prove changes with `make verify` (`verify-llm-infer` skill at `.agents/skills/verify-llm-infer/`).

GPU-only paths (FlashInfer, flash-attn) run on Modal A100-80GB through `scripts/modal_*.py`.
An import contract failure names the forbidden import and its source line. Fix the import, then
run `make architecture-check` and `make check`.

## Enforced Rules

The scheduler views restrict typed callers but do not enforce runtime immutability. Change
membership through `add`, `admit`, `release`, `requeue`, or `cancel`.

| Rule | Enforced by |
| --- | --- |
| Dependencies follow module ownership; `llm_infer` never imports `modal` | `pyproject.toml::tool.importlinter.contracts`, `Makefile::architecture-check` |
| Modal images install exact versions through uv: lock versions from `uv.lock`, vLLM from one pin | `tests/test_modal_image_pins.py::test_no_modal_image_installs_a_floating_version_or_uses_pip`, `scripts/modal_image_pins.py::locked` |
| Process, network, and Modal modules stay in their owner files; Modal only in `scripts/modal_*.py`, where the spend checks look | `tests/test_process_network_owners.py::test_process_and_network_modules_stay_in_their_owners` |
| Public tok/s requires qualified fp32 reference evidence | `llm_infer/benchmarks/reference_policy.py::headline_tokens_per_second`, `tests/benchmarks/test_evidence_contract.py::test_public_tables_reject_raw_timing_disguised_as_a_headline`, `Makefile::evidence-check` |
| A/B parity and fp32 reference status stay separate | `llm_infer/benchmarks/paired_ab.py::judge_records`, `tests/benchmarks/test_evidence_contract.py::test_exact_relative_parity_can_share_a_pending_reference_review` |
| Deferred tokens survive abort and batch changes | `llm_infer/serving/decode_window.py::DecodeWindows`, `tests/serving/test_decode_window.py::test_abort_flush_lands_in_next_step_result` |
| Decode-ready requests always have a KV table | `llm_infer/serving/request.py::DecodeReady`, `tests/correctness/test_request_lifecycle.py::test_initial_prefill_requires_complete_cache_before_recording` |
| Scheduler membership changes use scheduler methods | `llm_infer/scheduler/scheduler.py::Scheduler`, `tests/scheduler/test_scheduler.py::test_cancel_releases_only_the_cancelled_requests_reservation` |
| Trace replay rejects malformed events | `visualizer/trace_loader.js::validateEvent`, `Makefile::visualizer-test`, `tests/serving/test_tracing.py::test_trace_recorder_captures_real_engine_events` |
| Modal entrypoints and every Modal function, CPU-only included, refuse to spend without `--approved`; the server needs `LLM_INFER_SERVE_APPROVED=1`; paid work uses `@app.function`, never `@app.cls` | `scripts/modal_approval.py::require_spend_approval`, `tests/test_modal_approval.py::test_local_entrypoints_gate_before_any_modal_call`, `tests/test_modal_approval.py::test_modal_functions_refuse_direct_calls_without_approval`, `tests/test_modal_approval.py::test_paid_work_uses_modal_functions_not_classes`, `tests/test_modal_approval.py::test_only_a_gated_entrypoint_forwards_approval`, `tests/test_modal_approval.py::test_server_apps_refuse_to_start_without_approval` |
| No doc shows a runnable Modal command with `--approved` or `LLM_INFER_SERVE_APPROVED=1` | `tests/test_docs_approval_commands.py::test_docs_show_no_approved_command` |
| Modal GPU functions arm the stall guard; web servers are exempt | `scripts/modal_stall_guard.py::arm`, `tests/test_modal_stall_guard.py::test_gpu_functions_arm_the_stall_guard` |
| `make verify ARGS=control` passes only on its exact expected FAIL; CI runs it after `verify all` | `.agents/skills/verify-llm-infer/drives.py::control_serve_http`, `Makefile::verify` |

## Change Navigation

Paths start inside `llm_infer/`, except `tests/` and `visualizer/`.

| Change | Start here | Check |
| --- | --- | --- |
| Backend support or grouped graph construction | `model/interface.py`, `model/runtime.py` | `tests/model/test_runtime.py`, `tests/serving/test_engine_grouped_decode.py` |
| Request lifecycle or KV release | `serving/request.py`, `serving/engine.py`, `scheduler/scheduler.py` | `tests/correctness/test_request_lifecycle.py`, `tests/correctness/test_preemption.py` |
| Deferred decode windows | `serving/decode_window.py`, `serving/engine.py` | `tests/serving/test_decode_window.py` |
| Benchmark timing or oracle caching | `benchmarks/esme_paged.py`, `benchmarks/esme_measurement.py` | `tests/benchmarks/test_esme_measurement.py` |
| Benchmark qualification or A/B verdicts | `benchmarks/reference_policy.py`, `benchmarks/paired_ab.py` | `tests/benchmarks/`, `make evidence-check` |
| Trace events or replay | `tracing.py`, `visualizer/trace_loader.js` | `tests/serving/test_tracing.py`, `make visualizer-test` |
