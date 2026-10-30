"""The fields tests read from the committed Qwen chain-of-thought golden fixture."""

from __future__ import annotations

from typing import TypedDict


class GoldenCase(TypedDict):
    case_id: str
    prompt_ids: list[int]
    continuation_ids: list[int]


class GoldenModel(TypedDict):
    id: str
    revision: str


class GoldenDecoding(TypedDict):
    do_sample: bool
    max_new_tokens: int
    eos_token_ids: list[int]


class GoldenEnvironment(TypedDict):
    dtype: str


class GoldenFixture(TypedDict):
    model: GoldenModel
    environment: GoldenEnvironment
    decoding: GoldenDecoding
    cases: list[GoldenCase]
