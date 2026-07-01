"""The shared three-way benchmark workload — one identical input for every system.

Cleary rule #1 of the benchmark: naive HF, llm-infer, and vLLM must decode the *same*
prompts under the *same* stop config, or the tokens/s numbers compare different work.
This module is the single source of that workload, so no runner can quietly use an
easier input.

The prompts are the committed golden ``prompt_ids`` (``tests/correctness/goldens/``) —
already tokenized through the pinned Instruct chat template and frozen, byte-identical to
the reference check. The benchmark replays those prompt token ids directly (no
tokenizer, no model here), cycling the small case pool up to ``num_requests`` so the batch
is large enough to exercise continuous batching. Identical-prompt replication is fair only
because vLLM prefix caching is pinned **off** in the vLLM runner — every system recomputes
every prefill, so replication hands no one a free cache hit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "correctness"
    / "goldens"
    / "qwen2_5_coder_3b_instruct_cot.json"
)
ROLLOUT_FIXTURE = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "rollout_grpo_s0_spider_dev.json"
)


@dataclass(frozen=True)
class SamplingConfig:
    """Decoding params for the rollout workload (greedy when ``temperature == 0``)."""

    temperature: float
    top_p: float
    seed: int


@dataclass(frozen=True)
class BenchRequest:
    """One benchmark request: a stable id and its frozen prompt token ids."""

    request_id: str
    prompt_ids: tuple[int, ...]
    case_id: str


@dataclass(frozen=True)
class Workload:
    """The full benchmark input every system runs, plus its provenance for the result.

    ``model_id``/``model_revision`` name the *served* weights — the pinned base for the
    benchmark greedy benchmark, or a local merged-adapter path (revision ``None``) for the
    rollout rollout. ``sampling`` is ``None`` for greedy decoding and set for the rollout.
    """

    requests: tuple[BenchRequest, ...]
    max_new_tokens: int
    eos_token_ids: frozenset[int]
    model_id: str
    model_revision: str | None
    source: str
    sampling: SamplingConfig | None = None

    @property
    def num_requests(self) -> int:
        return len(self.requests)

    @property
    def prompt_lengths(self) -> tuple[int, ...]:
        return tuple(len(r.prompt_ids) for r in self.requests)


def build_workload(
    num_requests: int,
    max_new_tokens: int,
    *,
    fixture_path: Path = DEFAULT_FIXTURE,
) -> Workload:
    """Build ``num_requests`` requests by cycling the committed golden prompts.

    ``max_new_tokens`` overrides the fixture's short reference check length (40) with a
    decode-heavy length so the decode loop — where paging and the fused kernel matter —
    dominates the measured time rather than prefill. EOS, model id, and revision come
    straight from the pinned fixture so the benchmark and the reference check agree on stop rules.
    """
    if num_requests < 1:
        raise ValueError(f"num_requests must be >= 1; got {num_requests}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1; got {max_new_tokens}")

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    if not cases:
        raise ValueError(f"fixture {fixture_path} has no cases")
    eos = frozenset(fixture["decoding"]["eos_token_ids"])

    requests = tuple(
        BenchRequest(
            request_id=f"req-{i:03d}",
            prompt_ids=tuple(cases[i % len(cases)]["prompt_ids"]),
            case_id=cases[i % len(cases)]["case_id"],
        )
        for i in range(num_requests)
    )
    source = (
        f"{fixture_path.name}: {len(cases)} llm-rlvr cot prompts cycled to "
        f"{num_requests} requests (prefix caching off → replication is fair)"
    )
    return Workload(
        requests=requests,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos,
        model_id=fixture["model"]["id"],
        model_revision=fixture["model"]["revision"],
        source=source,
    )


def build_rollout_workload(
    served_model_id: str,
    served_model_revision: str | None = None,
    *,
    fixture_path: Path = ROLLOUT_FIXTURE,
    num_prompts: int | None = None,
    num_generations: int | None = None,
    max_completion_length: int | None = None,
) -> Workload:
    """Replay one frozen llm-rlvr GRPO rollout batch from the committed fixture.

    Expands the ``num_prompts`` frozen prompts into ``num_prompts × num_generations``
    completions (one ``BenchRequest`` each, ``p{i}-g{j}``) — the actual GRPO rollout shape.
    ``served_model_id`` is the merged grpo-s0 weights (revision ``None`` for a local path),
    *not* the base in the fixture. The ``num_*`` overrides exist only for the cheap smoke;
    the full run takes them from the fixture's pinned ``rollout`` block.
    """
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    rollout = fixture["rollout"]
    prompts = fixture["prompts"]
    n_prompts = num_prompts if num_prompts is not None else rollout["num_prompts"]
    g = num_generations if num_generations is not None else rollout["num_generations"]
    max_new = (
        max_completion_length
        if max_completion_length is not None
        else rollout["max_completion_length"]
    )
    if n_prompts < 1 or n_prompts > len(prompts):
        raise ValueError(f"num_prompts must be in [1, {len(prompts)}]; got {n_prompts}")
    if g < 1:
        raise ValueError(f"num_generations must be >= 1; got {g}")

    requests = tuple(
        BenchRequest(
            request_id=f"{prompts[p]['prompt_id']}-g{gen}",
            prompt_ids=tuple(prompts[p]["prompt_ids"]),
            case_id=prompts[p]["prompt_id"],
        )
        for p in range(n_prompts)
        for gen in range(g)
    )
    sampling = SamplingConfig(
        temperature=rollout["temperature"],
        top_p=rollout["top_p"],
        seed=rollout["sampling_seed"],
    )
    source = (
        f"{fixture_path.name}: llm-rlvr GRPO rollout (anchor {rollout['anchor']}), "
        f"{n_prompts} Spider-dev prompts × G={g} = {len(requests)} completions, "
        f"temp={sampling.temperature} top_p={sampling.top_p} seed={sampling.seed}, "
        f"base {fixture['model']['id']}@{fixture['model']['revision'][:8]} + merged grpo-s0"
    )
    return Workload(
        requests=requests,
        max_new_tokens=max_new,
        eos_token_ids=frozenset(rollout["eos_token_ids"]),
        model_id=served_model_id,
        model_revision=served_model_revision,
        source=source,
        sampling=sampling,
    )
