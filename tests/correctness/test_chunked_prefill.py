"""Chunked prefill correctness and scheduling tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.kv_cache import BlockTable, PagedKVCache
from llm_infer.model.qwen import QwenModel
from llm_infer.serving import InferenceEngine, Request
from tests.support.fake_causal_lm import FakeCausalLMBase


class ChunkedModel(FakeCausalLMBase):
    """Small model-shaped object for proving engine scheduling without loading 3B weights."""

    def __init__(self) -> None:
        self.num_layers = 1
        self.num_kv_heads = 1
        self.head_dim = 2
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.profiler = None
        self.prefill_chunks: list[tuple[int, int]] = []

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
        rows = torch.arange(start_pos, end_pos, dtype=torch.float32).reshape(-1, 1, 1)
        key = torch.cat([rows, rows + 0.5], dim=-1)
        cache.write(table, layer=0, start_pos=start_pos, key=key, value=key + 100.0)
        table.length = end_pos
        self.prefill_chunks.append((start_pos, end_pos))
        return self._logits(10 + end_pos)

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
                torch.tensor([[float(pos), float(token)]], dtype=torch.float32)
                for pos, token in zip(positions, tokens.tolist(), strict=True)
            ]
        )
        cache.write_many(tables, layer=0, positions=positions, key=key, value=key + 200.0)
        for table, pos in zip(tables, positions, strict=True):
            table.length = pos + 1
        return torch.stack([self._logits(int(token) + 1) for token in tokens.tolist()])

    def _logits(self, token_id: int) -> torch.Tensor:
        logits = torch.full((64,), -100.0)
        logits[token_id % logits.numel()] = 100.0
        return logits


def _tiny_qwen() -> QwenModel:
    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=16,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        tie_word_embeddings=False,
    )
    vocab_size = 37
    intermediate_size = 32
    generator = torch.Generator().manual_seed(1234)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.08

    weights: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": randn(vocab_size, config.hidden_size),
        "model.norm.weight": torch.ones(config.hidden_size),
        "lm_head.weight": randn(vocab_size, config.hidden_size),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = torch.ones(config.hidden_size)
        weights[prefix + "post_attention_layernorm.weight"] = torch.ones(config.hidden_size)
        weights[prefix + "self_attn.q_proj.weight"] = randn(
            config.num_attention_heads * 4, config.hidden_size
        )
        weights[prefix + "self_attn.k_proj.weight"] = randn(
            config.num_key_value_heads * 4, config.hidden_size
        )
        weights[prefix + "self_attn.v_proj.weight"] = randn(
            config.num_key_value_heads * 4, config.hidden_size
        )
        weights[prefix + "self_attn.o_proj.weight"] = randn(config.hidden_size, config.hidden_size)
        weights[prefix + "mlp.gate_proj.weight"] = randn(intermediate_size, config.hidden_size)
        weights[prefix + "mlp.up_proj.weight"] = randn(intermediate_size, config.hidden_size)
        weights[prefix + "mlp.down_proj.weight"] = randn(config.hidden_size, intermediate_size)

    return QwenModel(weights, config, TorchNaiveAttention(), torch.float32)


def test_active_decode_advances_between_prefill_chunks() -> None:
    model = ChunkedModel()
    engine = InferenceEngine(model, block_size=4, num_blocks=8, prefill_chunk_size=2)
    active = Request("active", [1], 4, frozenset({63}))
    long = Request("long", [2, 3, 4, 5, 6], 2, frozenset({63}))
    engine.add_request(active)
    engine.add_request(long)

    first = engine.step()
    second = engine.step()
    third = engine.step()

    assert first.prefill_chunks == {"active": (0, 1), "long": (0, 2)}
    assert first.tokens.keys() == {"active"}
    assert second.prefill_chunks == {"long": (2, 4)}
    assert second.tokens.keys() == {"active"}
    assert third.prefill_chunks == {"long": (4, 5)}
    assert third.tokens.keys() == {"active", "long"}
    assert active.generated[:3] == [11, 12, 13]
    assert long.generated == [15]


def test_chunked_prefill_logits_match_full_prefill() -> None:
    model = _tiny_qwen()
    prompt_ids = [1, 5, 9, 13, 17, 21, 25]

    full_cache = PagedKVCache(
        num_layers=model.num_layers,
        num_blocks=8,
        block_size=3,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype=model.dtype,
    )
    full_table = full_cache.new_request()
    full_logits = model.prefill(prompt_ids, full_cache, full_table)

    chunk_cache = PagedKVCache(
        num_layers=model.num_layers,
        num_blocks=8,
        block_size=3,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype=model.dtype,
    )
    chunk_table = chunk_cache.new_request()
    chunk_logits = None
    while chunk_table.length < len(prompt_ids):
        chunk_logits = model.prefill_chunk(
            prompt_ids,
            chunk_cache,
            chunk_table,
            start_pos=chunk_table.length,
            chunk_size=2,
        )

    assert chunk_logits is not None
    torch.testing.assert_close(chunk_logits, full_logits, rtol=1e-5, atol=1e-6)


def test_chunked_engine_tokens_match_full_prefill_engine() -> None:
    model = _tiny_qwen()
    prompt_ids = [1, 5, 9, 13, 17, 21, 25]
    eos = frozenset({36})

    full = InferenceEngine(model, block_size=3, num_blocks=8)
    chunked = InferenceEngine(model, block_size=3, num_blocks=8, prefill_chunk_size=2)
    full.add_request(Request("r", prompt_ids, 5, eos))
    chunked.add_request(Request("r", prompt_ids, 5, eos))

    assert chunked.run()["r"] == full.run()["r"]


def test_chunked_prefix_group_shares_full_prompt_blocks_then_cow_partial_block() -> None:
    model = ChunkedModel()
    engine = InferenceEngine(model, block_size=4, num_blocks=24, prefill_chunk_size=2)
    requests = [
        Request(f"p0-g{idx}", [11, 12, 13, 14, 15, 16], 3, frozenset({63}), "p0")
        for idx in range(4)
    ]
    for request in requests:
        engine.add_request(request)

    engine.step()
    engine.step()
    complete = engine.step()
    assert complete.prefill_chunks == {"p0-g0": (4, 6)}
    assert complete.tokens.keys() == {request.request_id for request in requests}
    assert model.prefill_chunks == [(0, 2), (2, 4), (4, 6)]

    prompt_blocks = [request.block_table.blocks for request in requests]
    assert len({blocks[0] for blocks in prompt_blocks}) == 1
    assert len({blocks[1] for blocks in prompt_blocks}) == 1

    engine.step()

    decode_blocks = [request.block_table.blocks for request in requests]
    shared_full_block = decode_blocks[0][0]
    assert {blocks[0] for blocks in decode_blocks} == {shared_full_block}
    assert engine.cache.allocator.refcount(shared_full_block) == 4
    assert len({blocks[1] for blocks in decode_blocks}) == 4
    assert all(engine.cache.allocator.refcount(blocks[1]) == 1 for blocks in decode_blocks)
