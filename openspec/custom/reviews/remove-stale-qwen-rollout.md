# Review: remove-stale-qwen-rollout

Scope: `git diff main...HEAD` on branch `adam/remove-stale-qwen-rollout`.

## Low

**Leftover Qwen benchmark builder is still exported** — `llm_infer/benchmarks/__init__.py:21` — `build_workload` remains in the public benchmark API even though the Qwen benchmark harnesses were deleted. Its implementation still defaults to `QWEN_COT_GOLDEN` and returns the fixture's Qwen model identity (`llm_infer/benchmarks/workload.py:24`, `llm_infer/benchmarks/workload.py:102`). No callers outside its definition/export were found, so this is stale surface area rather than live behavior. It matters because new benchmark work could accidentally build on the old Qwen fixture path despite the branch's "Qwen is correctness reference only" cleanup.

Resolution: removed `build_workload` and the Qwen fixture default from `llm_infer/benchmarks/workload.py`, and removed the export from `llm_infer/benchmarks/__init__.py`.

## Areas Reviewed & Found Clean

- Deleted rollout files: no tracked live references found to `modal_rollout.py`, `modal_benchmark.py`, `merge_adapter.py`, `build_rollout_fixture.py`, or `rollout_grpo_s0_spider_dev.json`.
- Qwen correctness path: fixture, `QwenModel`, CPU/flash reference docs, and `scripts/modal_reference_check.py` still point at Qwen as an independent correctness reference.
- Benchmark reporting: wrong-reference rows still suppress tok/s via `throughput_rows`; focused tests passed.
- Sampling behavior: rollout-specific benchmark seeding was removed, while serving-level sampling coverage remains in place.

## Summary

| Severity | Count |
| --- | ---: |
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low | 1 |

Overall: no blocking bugs found. One low cleanup finding was fixed.
