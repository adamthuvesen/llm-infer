"""Tests for shared Esme benchmark workload helpers."""

from __future__ import annotations

import pytest
import torch

from llm_infer.benchmarks.esme_paged import (
    EsmeBenchRequest,
    requests_at_context_length,
    single_request_prompt_coverage,
)
from llm_infer.benchmarks.esme_three_way import tie_tolerant_agreement


def _requests() -> list[EsmeBenchRequest]:
    return [
        EsmeBenchRequest("r0", "p0", (1, 4, 7)),
        EsmeBenchRequest("r1", "p1", (2, 5)),
    ]


def test_requests_at_context_length_repeats_valid_prompts_to_exact_size() -> None:
    resized = requests_at_context_length(_requests(), 8)

    assert [len(request.prompt_ids) for request in resized] == [8, 8]
    assert resized[0].prompt_ids == (1, 4, 7, 1, 4, 7, 1, 4)
    assert resized[1].prompt_ids == (2, 5, 2, 5, 2, 5, 2, 5)
    assert [request.request_id for request in resized] == ["r0", "r1"]


def test_requests_at_context_length_rejects_zero() -> None:
    with pytest.raises(ValueError, match="context_length must be >= 1"):
        requests_at_context_length(_requests(), 0)


def test_single_request_coverage_includes_every_prompt_at_every_context() -> None:
    class Tokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            add_generation_prompt: bool,
            tokenize: bool,
        ) -> list[int]:
            assert add_generation_prompt and tokenize
            return [len(messages[0]["content"]), 7]

    coverage = single_request_prompt_coverage(Tokenizer(), (32, 768), ("one", "two", "three"))

    assert len(coverage) == 6
    assert {(context, request.request_id) for context, request in coverage} == {
        (32, "esme-000"),
        (32, "esme-001"),
        (32, "esme-002"),
        (768, "esme-000"),
        (768, "esme-001"),
        (768, "esme-002"),
    }
    assert all(len(request.prompt_ids) == context for context, request in coverage)


class _OracleMustNotBeUsed:
    def logits(self, _token_ids: list[int]) -> torch.Tensor:
        raise AssertionError("exact/missing-output comparison should not consult logits")


def test_tie_tolerant_agreement_fails_missing_or_extra_request_ids() -> None:
    requests = _requests()
    reference = {"r0": [1, 2], "r1": [3, 4]}

    missing = tie_tolerant_agreement(
        _OracleMustNotBeUsed(),
        requests,
        {"r0": [1, 2]},
        reference,
        frozenset({99}),
    )
    assert missing.total == 2
    assert missing.nontie == 1
    assert not missing.all_ties_or_exact
    assert "missing outputs" in str(missing.divergences_sample)

    extra = tie_tolerant_agreement(
        _OracleMustNotBeUsed(),
        requests,
        {"r0": [1, 2], "r1": [3, 4], "r2": [5]},
        reference,
        frozenset({99}),
    )
    assert extra.total == 2
    assert extra.nontie == 1
    assert not extra.all_ties_or_exact
    assert "unexpected outputs" in str(extra.divergences_sample)
