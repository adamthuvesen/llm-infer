"""Tiny Qwen-shaped model factory for CPU tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.model.qwen import QwenModel


def tiny_qwen() -> QwenModel:
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
