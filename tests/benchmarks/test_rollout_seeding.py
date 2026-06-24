"""GRPO rollout seeding: a prompt's G completions must be independent, not G copies.

A real GRPO group needs diverse rollouts per prompt — that diversity is what advantage
estimation feeds on. The runner expands ``prompts × G`` and, under sampling, must give each
request its own seed so the G completions diverge while staying reproducible across iterations.
A single shared seed (the bug) seeds every request's generator identically and collapses the
group to one repeated completion.

Runs on the tiny random-weight Qwen (CPU, real logits with genuine entropy) — no 3B load, no GPU.
"""

from __future__ import annotations

from llm_infer.benchmarks.runners import run_llm_infer
from llm_infer.benchmarks.workload import BenchRequest, SamplingConfig, Workload
from tests.correctness.test_chunked_prefill import _tiny_qwen


def _group_workload() -> Workload:
    prompt = (1, 5, 9, 13)
    # One prompt expanded to G=4 completions, exactly as the rollout loader builds prompts × G.
    requests = tuple(
        BenchRequest(request_id=f"p0-g{g}", prompt_ids=prompt, case_id="p0") for g in range(4)
    )
    return Workload(
        requests=requests,
        max_new_tokens=10,
        eos_token_ids=frozenset({36}),  # tiny model never greedily emits this; length caps decode
        model_id="tiny",
        model_revision=None,
        source="test",
        sampling=SamplingConfig(temperature=1.0, top_p=1.0, seed=123),
    )


def test_group_completions_diverge_and_reproduce() -> None:
    model = _tiny_qwen()
    workload = _group_workload()

    run_a = run_llm_infer(model, workload, num_blocks=64, warmup=0, iters=1, device="cpu")
    run_b = run_llm_infer(model, workload, num_blocks=64, warmup=0, iters=1, device="cpu")

    # Independent draws: the G completions of the prompt are not all identical.
    distinct = {tuple(ids) for ids in run_a.outputs.values()}
    assert len(distinct) > 1, "G completions are identical — per-request seeds were not applied"
    # Reproducible: the same derived seeds produce the same tokens on a second run.
    assert run_a.outputs == run_b.outputs
