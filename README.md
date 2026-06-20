# llm-infer

A minimal, honest paged LLM inference engine for **Qwen2.5-Coder-3B-Instruct**,
sequenced so the correctness oracle comes before anything fast or batched. The
eventual goal is to benchmark it as an rlvr-sql GRPO rollout backend.

This is **Phase A**: the trusted reference and the repo skeleton.

- `kernels/` — an `AttentionBackend` protocol and a slow, readable `torch_naive`
  reference backend (the truth every future backend is validated against).
- `model/` — the minimal Qwen2.5-Coder loading + forward path to greedily decode a
  single request end-to-end through the reference backend.
- `tests/correctness/` — the oracle: single-request, token-for-token vs HuggingFace
  greedy, run against small committed golden fixtures.

Paging, continuous batching, fast kernels, and benchmarks are later phases. See
[`docs/scoping.md`](docs/scoping.md) for the full plan and [`AGENTS.md`](AGENTS.md)
for the working agreement.

## Quickstart

```bash
uv sync --extra dev
uv run ruff check
uv run pytest tests/correctness -q
```

Phase A is CPU-runnable: the oracle checks committed golden token ids and needs no
GPU. Regenerating goldens (`scripts/generate_goldens.py`) loads the 3B model in fp32
on CPU.

## Model pin

`Qwen/Qwen2.5-Coder-3B-Instruct` at HF revision
`488639f1ff808d1d3d0ba301aef8c11461451ec5` (the Instruct variant — see
`llm_infer/model/config.py`).
