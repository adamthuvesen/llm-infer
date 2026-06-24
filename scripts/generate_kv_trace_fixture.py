"""Generate the committed schema-v2 trace used by the KV trace visualizer.

The fixture deliberately runs through ``InferenceEngine(trace=...)`` with a tiny model-shaped
object that writes real KV rows. It avoids loading Qwen weights, but it still exercises the
same scheduler, paged cache, request lifecycle, chunked prefill, decode, and trace emission
path as engine traces from a full model.
"""

from __future__ import annotations

from pathlib import Path

import torch

from llm_infer.kv_cache import BlockTable, PagedKVCache
from llm_infer.serving import InferenceEngine, Request
from llm_infer.tracing import TraceRecorder

FIXTURE_PATH = Path("docs/assets/kv_trace_schema_v2.jsonl")


class FixtureClock:
    """Small deterministic clock so throughput samples are stable in git."""

    def __init__(self) -> None:
        self._ticks = -1

    def __call__(self) -> float:
        self._ticks += 1
        return self._ticks * 0.25


class TraceFixtureModel:
    """Model-shaped test double that exercises real KV writes and deterministic logits."""

    def __init__(self) -> None:
        self.num_layers = 1
        self.num_kv_heads = 1
        self.head_dim = 4
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.profiler = None

    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        return self.prefill_chunk(prompt_ids, cache, table, start_pos=0, chunk_size=len(prompt_ids))

    def prefill_chunk(
        self,
        prompt_ids: list[int],
        cache: PagedKVCache,
        table: BlockTable,
        *,
        start_pos: int,
        chunk_size: int,
    ) -> torch.Tensor:
        end_pos = min(len(prompt_ids), start_pos + chunk_size)
        table.reserve(end_pos - start_pos)
        positions = torch.arange(start_pos, end_pos, dtype=torch.float32).reshape(-1, 1, 1)
        prompt_marker = torch.full_like(positions, float(prompt_ids[0]))
        key = torch.cat(
            [
                positions,
                prompt_marker,
                positions + prompt_marker / 100.0,
                positions + 0.5,
            ],
            dim=-1,
        )
        cache.write(table, layer=0, start_pos=start_pos, key=key, value=key + 1000.0)
        table.length = end_pos
        return self._logits(20 + prompt_ids[0] + end_pos)

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
        key = torch.stack(
            [
                torch.tensor(
                    [[float(pos), float(token), float(pos + token), 1.0]],
                    dtype=torch.float32,
                )
                for pos, token in zip(positions, tokens.tolist(), strict=True)
            ]
        )
        cache.write_many(tables, layer=0, positions=positions, key=key, value=key + 2000.0)
        for table, pos in zip(tables, positions, strict=True):
            table.length = pos + 1
        return torch.stack([self._logits(int(token) + 1) for token in tokens.tolist()])

    def _logits(self, token_id: int) -> torch.Tensor:
        logits = torch.full((128,), -100.0)
        logits[token_id] = 100.0
        return logits


def build_trace_jsonl() -> str:
    recorder = TraceRecorder()
    engine = InferenceEngine(
        TraceFixtureModel(),
        block_size=4,
        num_blocks=6,
        prefill_chunk_size=2,
        trace=recorder,
        trace_clock=FixtureClock(),
    )
    engine.add_request(Request("chat-short", [3, 9], 5, frozenset({127})))
    engine.add_request(Request("schema-long", [8, 4, 6, 2, 9, 5, 1], 3, frozenset({127})))
    engine.add_request(
        Request("rollout-a", [12, 7, 7, 3], 4, frozenset({127}), prefix_group_id="rollout")
    )
    engine.add_request(
        Request("rollout-b", [12, 7, 7, 3], 4, frozenset({127}), prefix_group_id="rollout")
    )

    outputs = engine.run()
    expected_lengths = {
        "chat-short": 5,
        "schema-long": 3,
        "rollout-a": 4,
        "rollout-b": 4,
    }
    if {request_id: len(tokens) for request_id, tokens in outputs.items()} != expected_lengths:
        raise RuntimeError(f"unexpected fixture output lengths: {outputs}")
    return recorder.to_jsonl() + "\n"


def main() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(build_trace_jsonl(), encoding="utf-8")
    print(f"wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
