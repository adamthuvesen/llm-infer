# AGENTS.md — llm-infer

A minimal, honest paged LLM inference engine with pluggable model backends. Qwen2.5-Coder
is the pinned HF-oracle backend; exported llm-pretrain DenseBackbone bundles are a
correctness-only sibling backend. Full scope and build order live in [`docs/scoping.md`](docs/scoping.md).
Read it before proposing work.

## Core doctrine: the correctness oracle comes first

The project is sequenced so the **trusted reference comes before anything fast**.
Every optimized path is validated against a trusted greedy reference before it reports a
single tok/s. For Qwen, that reference is HuggingFace greedy decoding at the pinned revision.
For exported DenseBackbone bundles, the reference is the source bundle logits/generation
contract. There is no "correct-ish": exact token ids on the single-request unit path, or a
divergence traced to a numerical tie and documented.

## Backends

- `qwen`: `Qwen/Qwen2.5-Coder-3B-Instruct` at HF revision
  `488639f1ff808d1d3d0ba301aef8c11461451ec5`. Use the **Instruct** variant and its
  chat template. Plain `-3B` is a different model and would invalidate the Qwen oracle.
  Full paged KV, prefix caching, speculative decode, preemption, and flash-attn.
- `dense`: `llm_pretrain_dense_v1` export bundles from `llm-pretrain`. Correctness bridge
  only — full recompute through the engine API, no real paged KV. Rejects prefix caching,
  speculative decode, and preemption at init/add_request. See `BackendCapabilities` in
  `llm_infer/model/interface.py`.

Qwen's default pin lives in `llm_infer/model/config.py` (`MODEL_ID`, `MODEL_REVISION`).
Backend selection lives in `llm_infer/model/runtime.py`.

## Layout

```
llm_infer/
  model/        # backend interface/registry, Qwen, DenseBackbone loader, rope utils
  kernels/      # AttentionBackend protocol, torch_naive reference, flash_attn_paged
  kv_cache/     # block allocator, block tables, paged page store
  scheduler/    # prefill/decode admission, continuous batching, optional preemption
  serving/      # InferenceEngine step loop, sampler, OpenAI HTTP server, metrics
  benchmarks/   # shared workload + runners (Modal harnesses in scripts/)
tests/correctness/   # HF-exact greedy oracle + batch/prefix/preemption/speculative suites
docs/                # scoping.md, architecture.md, fixture-format.md, benchmark.md
scripts/             # generate_goldens, modal_oracle, modal_benchmark, modal_rollout,
                     # merge_adapter, build_rollout_fixture, loadgen, kv trace fixture
visualizer/          # schema-v3 KV trace replay UI
```

Implemented through Phases A–E plus prefix caching, chunked prefill, speculative decode v1,
request preemption, KV trace visualizer, and OpenAI-compatible serving with metrics/loadgen.

## The honesty bar

Exact token ids on the single-request unit path is the bar. bf16 greedy can diverge
from HF on a genuine tie-break step; that is acceptable **only** when each divergence
is traced to a numerical tie (logits equal within tolerance) and documented — never
waved off as "close enough." The oracle runs in **fp32** by default, where ties
near-vanish. See `docs/fixture-format.md`. A backend that fails the oracle reports no tok/s.

## Development

Managed with [`uv`](https://docs.astral.sh/uv/). Python 3.11+.

```bash
uv sync --extra dev --extra serving   # dev tests + HTTP server deps
uv run ruff check                     # lint (must be clean)
uv run pytest tests/correctness -q    # fast CPU oracle gate (slow 3B oracles deselected)
uv run pytest -q                      # full fast suite before merge
uv run pytest tests/correctness -q -m slow  # opt-in 3B CPU oracle
```

The local gate is **CPU-runnable by design** — no GPU required, and the slow 3B
oracles are opt-in because they load the pinned Qwen model and can take minutes on CPU.
The flash-attn backend is GPU-only; its oracle runs on the target GPU via
`scripts/modal_oracle.py`. Benchmark and rollout harnesses use Modal A100-80GB
(`scripts/modal_benchmark.py`, `scripts/modal_rollout.py`).

## Conventions

- Atomic [conventional commits](https://www.conventionalcommits.org/) (`type(scope): message`),
  one per logical unit.
- Type hints throughout; no `Any` casts — fix the type.
- `pathlib.Path` for filesystem work.
- Comments explain *why*, not *what*. Names use concrete domain terms.
- Validate untrusted input loudly; keep `try` bodies small.
