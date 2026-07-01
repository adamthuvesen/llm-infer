from __future__ import annotations

import json
from pathlib import PurePath

from llm_infer.benchmarks.workload import ROLLOUT_FIXTURE, build_rollout_workload


def _strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for child in value.values():
            strings.extend(_strings(child))
        return strings
    if isinstance(value, list):
        strings = []
        for child in value:
            strings.extend(_strings(child))
        return strings
    return []


def test_rollout_fixture_provenance_is_repo_agnostic() -> None:
    fixture = json.loads(ROLLOUT_FIXTURE.read_text(encoding="utf-8"))
    dataset = fixture["dataset"]

    assert "tables_json" not in dataset
    assert dataset["tables_json_source"] == "spider_data/tables.json"
    assert dataset["tables_json_sha256"] == (
        "61bb20aa401f03164e2d7f3b16509b7b5f79cc9c943ca7bd159046df1159e2ed"
    )

    for text in _strings(dataset):
        path = PurePath(text)
        assert not path.is_absolute()
        assert "/Users/" not in text


def test_rollout_fixture_workload_shape_is_unchanged() -> None:
    workload = build_rollout_workload("/merged/grpo-s0")

    assert workload.num_requests == 32
    assert workload.max_new_tokens == 1024
    assert workload.prompt_lengths == tuple(
        length for length in (251, 868, 334, 336, 359, 357, 672, 677) for _ in range(4)
    )
    assert workload.sampling is not None
    assert workload.sampling.temperature == 1.0
    assert workload.sampling.top_p == 1.0
    assert workload.sampling.seed == 0
