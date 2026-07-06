"""The shared benchmark workload: one identical greedy input for every system.

Benchmark rule #1: naive HF, llm-infer, and vLLM must decode the *same*
prompts under the *same* stop config, or the tokens/s numbers compare different work.
This module is the single source of that workload, so no runner can quietly use an
easier input.

The prompts are the committed golden ``prompt_ids`` (``llm_infer/fixtures/``) —
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
from importlib.resources.abc import Traversable
from pathlib import Path

from llm_infer.fixtures import QWEN_COT_GOLDEN

DEFAULT_FIXTURE = QWEN_COT_GOLDEN


@dataclass(frozen=True)
class BenchRequest:
    """One benchmark request: a stable id and its frozen prompt token ids."""

    request_id: str
    prompt_ids: tuple[int, ...]
    case_id: str


@dataclass(frozen=True)
class Workload:
    """The full benchmark input every system runs, plus its provenance for the result.

    ``model_id``/``model_revision`` name the served weights. The benchmark workload is greedy;
    sampling behavior is covered by serving-level tests instead of the speed harness.
    """

    requests: tuple[BenchRequest, ...]
    max_new_tokens: int
    eos_token_ids: frozenset[int]
    model_id: str
    model_revision: str | None
    source: str

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
    fixture_path: Traversable | Path = DEFAULT_FIXTURE,
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
