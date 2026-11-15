---
name: verify-llm-infer
description: "Drives the real llm-infer server, load generator, and Modal launch paths on CPU with a tiny Esme bundle. Use to run, launch, drive, or verify llm-infer, or to prove a change to it works."
---

# verify-llm-infer

One command proves a change works in the real entry points. `python -m llm_infer.serve` serves a
tiny random-weight Esme bundle on a free loopback port. The drives call its HTTP API and
`scripts/loadgen.py`. Every `scripts/modal_*.py` is imported with the real `modal` package and
must refuse each paid path. Nothing reaches Modal, a GPU, or the Hugging Face Hub.

## Run

```bash
make verify                       # drives for files changed against main; the smoke drive if none match
make verify ARGS=all              # every drive, about 15 s idle (two server starts)
make verify ARGS=list             # each drive, what it proves, the paths it covers
make verify ARGS=control          # one drive with a wrong expectation; exits 0 only on FAIL
make verify ARGS="launch-plan"    # named drives
```

Run `make install` once first.

## Reading results

- Each drive prints `PASS|FAIL <drive> <seconds>s <what it proved or why it failed>`.
- A FAIL names the expected value and the observed value. A drive that crashes is a FAIL.
- The run fails when it changes the checkout: a tracked, untracked, or ignored file, the index,
  HEAD, or `.git/config`. Only `.verify/` and `__pycache__/` may change.
- Evidence is in `.verify/runs/<utc>-<sha7>[-dirty]/`: per drive, the argv, redacted env,
  exit code, stdout, and stderr, plus `server.log`, the chat and stream bodies, `/metrics`, the
  loadgen summary, `launch-plan.json`, and `summary.json` (overall `result` and `checkout_changed`).
- The helper does not retry, so it never reports `FLAKY`. Run a failed drive again by name.

## Limits

- The tiny bundle has random weights. The drives prove transport, batching, and determinism.
  They do not prove model quality or speed. `make slow` checks the 3B Qwen reference on CPU.
- No drive runs FlashInfer, CUDA graphs, or a benchmark. Those are paid Modal runs. When one
  refuses, quote the printed command to Adam and wait. Never add `--approved` or
  `LLM_INFER_SERVE_APPROVED=1` yourself.

## Gotchas

- Some GPU functions are generators. Their body, and so their gate, runs only on the first
  `next()`, so the probe iterates them.
- The probe sets `sys.argv` to the `modal run` or `modal serve` command it emulates, because the
  serve gate echoes it.
- HTTP calls bypass proxy settings. Loadgen gets `NO_PROXY=127.0.0.1,localhost`.
