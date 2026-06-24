"""Prefix caching correctness: shared prompt prefill must not change generated tokens."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from llm_infer.kv_cache import BlockTable, PagedKVCache
from llm_infer.serving import InferenceEngine, Request, SamplingParams

GOLDEN_PATH = Path(__file__).parent / "goldens" / "qwen2_5_coder_3b_instruct_cot.json"
FIXTURE = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
EOS = frozenset(FIXTURE["decoding"]["eos_token_ids"])
CASE = FIXTURE["cases"][0]


class CountingToyModel:
    """Small model-shaped object for proving engine scheduling without loading 3B weights."""

    def __init__(self) -> None:
        self.num_layers = 1
        self.num_kv_heads = 1
        self.head_dim = 2
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.profiler = None
        self.prefill_calls = 0

    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        self.prefill_calls += 1
        table.reserve(len(prompt_ids))
        rows = torch.arange(len(prompt_ids), dtype=torch.float32).reshape(-1, 1, 1)
        key = torch.cat([rows, rows + 0.5], dim=-1)
        value = key + 100.0
        cache.write(table, layer=0, start_pos=0, key=key, value=value)
        table.length = len(prompt_ids)
        return self._logits(offset=sum(prompt_ids) % 17)

    def decode_many(
        self,
        cache: PagedKVCache,
        tables: list[BlockTable],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.as_tensor(token_ids, dtype=torch.long)
        positions = [table.length for table in tables]
        for table in tables:
            table.reserve(1)
        for table, pos in zip(tables, positions, strict=True):
            cache.prepare_write(table, pos, 1)
        key = torch.stack(
            [
                torch.tensor([[float(pos), float(token)]], dtype=torch.float32)
                for pos, token in zip(positions, tokens.tolist(), strict=True)
            ]
        )
        cache.write_many(tables, layer=0, positions=positions, key=key, value=key + 200.0)
        for table, pos in zip(tables, positions, strict=True):
            table.length = pos + 1
        return torch.stack([self._logits(offset=int(token)) for token in tokens.tolist()])

    def _logits(self, *, offset: int) -> torch.Tensor:
        base = torch.linspace(-1.0, 1.0, steps=32)
        return torch.roll(base, shifts=offset)


def _toy_run(*, prefix_group_id: str | None) -> tuple[dict[str, list[int]], int]:
    model = CountingToyModel()
    engine = InferenceEngine(
        model,
        block_size=4,
        num_blocks=32,
        default_sampling=SamplingParams(temperature=1.0, top_p=1.0, seed=123),
    )
    for idx in range(4):
        engine.add_request(
            Request(
                request_id=f"p0-g{idx}",
                prompt_ids=[11, 12, 13, 14, 15, 16],
                max_new_tokens=5,
                eos_token_ids=frozenset({31}),
                prefix_group_id=prefix_group_id,
            )
        )
    return engine.run(), model.prefill_calls


def test_shared_prefix_sampling_matches_prefill_per_sibling_baseline() -> None:
    baseline, baseline_prefills = _toy_run(prefix_group_id=None)
    shared, shared_prefills = _toy_run(prefix_group_id="p0")

    assert shared == baseline
    assert baseline_prefills == 4
    assert shared_prefills == 1


@pytest.mark.slow
def test_shared_prefix_real_model_matches_prefill_per_sibling_baseline() -> None:
    from llm_infer.model.qwen import QwenModel

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[FIXTURE["environment"]["dtype"]]
    model = QwenModel.load(dtype=dtype)
    prompt = list(CASE["prompt_ids"])
    steps = 4

    baseline = InferenceEngine(model, block_size=128, num_blocks=16)
    shared = InferenceEngine(model, block_size=128, num_blocks=16)
    for idx in range(4):
        baseline.add_request(Request(f"base-{idx}", prompt, steps, EOS))
        shared.add_request(Request(f"shared-{idx}", prompt, steps, EOS, prefix_group_id="case"))

    baseline_outputs = list(baseline.run().values())
    shared_outputs = list(shared.run().values())

    assert shared_outputs == baseline_outputs
