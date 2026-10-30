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
- Modal runs bill until they finish. Run one GPU job at a time unless the experiment needs more.
  Whoever starts a background run watches its log and raises an alarm after 10 minutes of silence.

## Commands

Use Python 3.13, [`uv`](https://docs.astral.sh/uv/), and Node 24. CI runs the same `make check`.

```bash
uv sync --extra dev --extra serving
make check        # import contracts, lint, format, types, CPU tests, evidence, visualizer
make help         # every target with its purpose
uv run pytest tests/correctness -q -m slow   # opt-in 3B Qwen reference check
```

GPU-only paths (FlashInfer, flash-attn) run on Modal A100-80GB through `scripts/modal_*.py`.
An import contract failure names the forbidden import and its source line. Fix the import, then
run `make architecture-check` and `make check`.

## Enforced Rules

| Rule | Enforced by | Added |
| --- | --- | --- |
| Dependencies follow module ownership | `pyproject.toml` `[tool.importlinter]`, `make architecture-check` | 2026-11-04 |
| Package and evidence CLI code must typecheck | `pyproject.toml` `[tool.mypy]`, `make typecheck` | 2026-11-04 |
| Public tok/s requires qualified fp32 reference evidence | `llm_infer/benchmarks/reference_policy.py`, `tests/benchmarks/test_evidence_contract.py`, `make evidence-check` | 2026-11-04 |
| A/B parity and fp32 reference status stay separate | `llm_infer/benchmarks/paired_ab.py`, `tests/benchmarks/test_evidence_contract.py` | 2026-11-04 |
| Deferred tokens survive abort and batch changes | `llm_infer/serving/decode_window.py`, `tests/serving/test_decode_window.py` | 2026-11-04 |
| Decode-ready requests always have a KV table | `llm_infer/serving/request.py`, `tests/correctness/test_request_lifecycle.py` | 2026-11-04 |
| Scheduler membership changes use scheduler methods | `llm_infer/scheduler/scheduler.py`, `tests/scheduler/test_scheduler.py` | 2026-11-04 |
| Trace replay rejects malformed events | `llm_infer/tracing.py`, `visualizer/trace_loader.js`, `tests/serving/test_tracing.py` | 2026-11-04 |
| Modal GPU functions arm the stall guard; web servers are exempt | `scripts/modal_stall_guard.py`, `tests/test_modal_stall_guard.py` | 2026-11-06 |
| Paths and targets in this table exist | `tests/test_agents_rules_table.py` | 2026-11-06 |

The scheduler views restrict typed callers but do not enforce runtime immutability. Change
membership through `add`, `admit`, `release`, `requeue`, or `cancel`.

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

## Conventions

- Atomic [conventional commits](https://www.conventionalcommits.org/) (`type(scope): message`).
- No `Any` casts. Fix the type.
- Comments explain why, not what. Names use concrete domain terms.
- Validate untrusted input loudly. Keep `try` bodies small.
