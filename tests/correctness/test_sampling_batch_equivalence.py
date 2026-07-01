"""Benchmark rules for per-request sampling: a request's draw is independent of its batch.

This is the load-bearing correctness property of per-request sampling. Each request samples
from its OWN seeded generator against its OWN generated history, so the token it draws at a
given decode step depends only on its seed and its own steps — never on which other requests
share the fused decode batch. The consequence, asserted here:

* **batched == serial under sampling** — a ``temperature > 0`` request with a fixed seed
  produces the identical token sequence whether it runs alone or batched with other,
  differently-configured requests (a mix of seeds, temperatures, and greedy rows);
* greedy rows in the same batch stay token-for-token the proven greedy path;
* seed reproducibility and seed divergence hold end-to-end through the engine.

Run on the tiny random-weight Qwen (37-token vocab) on CPU — real logits with genuine
entropy, so sampling actually exercises the RNG, but no 3B load and no GPU.
"""

from __future__ import annotations

from llm_infer.serving import InferenceEngine, Request, SamplingParams
from llm_infer.tracing import TraceRecorder
from tests.correctness.test_chunked_prefill import _tiny_qwen

EOS = frozenset({36})  # the tiny model does not emit this on these prompts, so length caps decode
STEPS = 12


def _engine() -> InferenceEngine:
    return InferenceEngine(_tiny_qwen(), block_size=8, num_blocks=64)


def _run_alone(prompt: list[int], sampling: SamplingParams) -> list[int]:
    engine = _engine()
    engine.add_request(Request("solo", list(prompt), STEPS, EOS, sampling=sampling))
    return engine.run()["solo"]


def test_sampled_request_batched_equals_serial() -> None:
    """A seeded sampled request decodes identically alone or batched with other requests.

    The batch deliberately mixes a high-temperature request, a different-seed request, a
    top-k/penalty request, and a greedy request — none of which may perturb the target's draw.
    """
    target_prompt = [3, 7, 11, 2]
    target_sampling = SamplingParams(temperature=1.0, top_p=0.95, seed=42)

    alone = _run_alone(target_prompt, target_sampling)

    engine = _engine()
    engine.add_request(Request("target", list(target_prompt), STEPS, EOS, sampling=target_sampling))
    engine.add_request(
        Request("hot", [1, 2, 3], STEPS, EOS, sampling=SamplingParams(temperature=2.0, seed=7))
    )
    engine.add_request(
        Request(
            "other_seed",
            [5, 9, 1, 8],
            STEPS,
            EOS,
            sampling=SamplingParams(temperature=1.0, top_p=0.95, seed=999),
        )
    )
    engine.add_request(
        Request(
            "topk",
            [2, 4, 6, 8, 10],
            STEPS,
            EOS,
            sampling=SamplingParams(temperature=1.0, top_k=5, frequency_penalty=1.5, seed=3),
        )
    )
    engine.add_request(Request("greedy", [4, 4, 4], STEPS, EOS))
    batched = engine.run()

    diff = next(
        (i for i, (a, b) in enumerate(zip(batched["target"], alone, strict=False)) if a != b), None
    )
    assert batched["target"] == alone, f"sampled target diverged batched-vs-serial at step {diff}"


def test_sampled_request_survives_preemption_batched_equals_serial() -> None:
    """A sampled request preempted and resumed under KV pressure still equals its solo run.

    The two hardest properties compose here: recompute rebuilds the *exact* KV of the evicted
    request, AND its per-request RNG advances only on real emitted tokens (the recompute resume
    samples nothing), so the seeded draw stream is identical no matter when eviction lands. We
    force the target to be the LIFO victim, confirm it is actually preempted, then assert its
    tokens match the same request decoded alone in a roomy, never-preempting pool — proof that
    eviction perturbs neither the cache nor the sampler.
    """
    model = _tiny_qwen()
    target_prompt = [3, 7, 11]
    target_sampling = SamplingParams(temperature=1.0, top_p=0.95, seed=42)

    # Reference: the target alone in a roomy pool, never preempted.
    solo = InferenceEngine(model, block_size=4, num_blocks=8)
    solo.add_request(Request("target", list(target_prompt), 6, EOS, sampling=target_sampling))
    alone = solo.run()["target"]

    # Tight preempting pool: fillers admitted first, the target last so the LIFO victim policy
    # evicts it under pressure. A mix of sampled and greedy batchmates must not perturb its draw.
    recorder = TraceRecorder()
    engine = InferenceEngine(model, block_size=4, num_blocks=3, preemption=True, trace=recorder)
    engine.add_request(
        Request("filler", [2, 6, 10], 6, EOS, sampling=SamplingParams(temperature=1.5, seed=7))
    )
    engine.add_request(Request("filler2", [4, 8, 12], 6, EOS))  # greedy batchmate
    engine.add_request(Request("target", list(target_prompt), 6, EOS, sampling=target_sampling))
    batched = engine.run()

    assert any(
        e.event == "request_preempted" and e.request_id == "target" for e in recorder.events
    ), "the target must actually be preempted or this test does not exercise the combined path"
    assert batched["target"] == alone, (
        f"sampled target diverged after preemption (batched={batched['target']}, alone={alone})"
    )


def test_greedy_row_in_a_sampled_batch_matches_greedy_alone() -> None:
    """A greedy request batched with sampled requests stays token-for-token the greedy path."""
    prompt = [4, 4, 4]
    greedy_alone = _run_alone(prompt, SamplingParams())  # default greedy

    engine = _engine()
    engine.add_request(Request("greedy", list(prompt), STEPS, EOS))
    engine.add_request(
        Request("s1", [1, 2, 3], STEPS, EOS, sampling=SamplingParams(temperature=1.5, seed=11))
    )
    engine.add_request(
        Request("s2", [7, 8, 9], STEPS, EOS, sampling=SamplingParams(temperature=1.0, seed=22))
    )
    batched = engine.run()

    assert batched["greedy"] == greedy_alone


def test_same_seed_reproduces_different_seed_diverges() -> None:
    """End to end through the engine: same seed → same tokens; different seed → different."""
    prompt = [3, 7, 11, 2]
    same_a = _run_alone(prompt, SamplingParams(temperature=1.0, seed=5))
    same_b = _run_alone(prompt, SamplingParams(temperature=1.0, seed=5))
    other = _run_alone(prompt, SamplingParams(temperature=1.0, seed=6))

    assert same_a == same_b
    assert same_a != other


def test_high_frequency_penalty_reduces_repetition() -> None:
    """A high frequency penalty measurably lowers repeated tokens vs no penalty (same seed)."""
    prompt = [3, 7, 11, 2]
    no_penalty = _run_alone(prompt, SamplingParams(temperature=1.0, seed=5))
    penalized = _run_alone(prompt, SamplingParams(temperature=1.0, frequency_penalty=2.0, seed=5))

    assert _max_repeat(penalized) < _max_repeat(no_penalty)


def _max_repeat(tokens: list[int]) -> int:
    """The count of the single most-repeated token id in ``tokens``."""
    return max((tokens.count(t) for t in set(tokens)), default=0)
