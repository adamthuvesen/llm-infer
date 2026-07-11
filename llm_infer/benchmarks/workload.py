"""Shared benchmark workload dataclasses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchRequest:
    """One benchmark request: a stable id and its frozen prompt token ids."""

    request_id: str
    prompt_ids: tuple[int, ...]
    case_id: str


@dataclass(frozen=True)
class Workload:
    """The full benchmark input every system runs, plus its provenance for the result."""

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
        return tuple(len(request.prompt_ids) for request in self.requests)
