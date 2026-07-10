"""Tiny DenseBackbone export bundle writer for tests and trace fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from llm_infer.model.pretrain_bundle import BUNDLE_FORMAT

VOCAB_SIZE = 11
HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 8
NUM_LAYERS = 2
NUM_HEADS = 2
NUM_KV_HEADS = 1


def write_tiny_pretrain_bundle(
    root: Path,
    *,
    qk_norm: bool = False,
    logit_soft_cap: float | None = 7.5,
) -> Path:
    bundle = root / "tiny_pretrain_bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.mkdir(exist_ok=True)
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "format": BUNDLE_FORMAT,
                "schema_version": 1,
                "model": {"name": "tiny-dense"},
                "tokenizer": {"path": "tokenizer.json", "format": "tokenizers-json"},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "config.json").write_text(
        json.dumps(
            {
                "vocab_size": VOCAB_SIZE,
                "embedding_dim": HIDDEN_SIZE,
                "feedforward_dim": INTERMEDIATE_SIZE,
                "layers": NUM_LAYERS,
                "heads": NUM_HEADS,
                "kv_heads": NUM_KV_HEADS,
                "norm_eps": 1e-5,
                "rope_theta": 10_000.0,
                "tie_embeddings": True,
                "qk_norm": qk_norm,
                "z_loss_weight": 1e-4,
            }
        ),
        encoding="utf-8",
    )
    if logit_soft_cap is not None:
        config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
        config["logit_soft_cap"] = logit_soft_cap
        (bundle / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (bundle / "tokenizer.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "vocab": {f"tok_{idx}": idx for idx in range(VOCAB_SIZE)},
                    "unk_token": "tok_0",
                },
                "pre_tokenizer": {"type": "Whitespace"},
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "format_version": 1,
            "metadata": {"key_format": BUNDLE_FORMAT},
            "state_dict": tiny_pretrain_state_dict(qk_norm=qk_norm),
        },
        bundle / "weights.pt",
    )
    return bundle


def tiny_pretrain_state_dict(*, qk_norm: bool = False) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260626)

    def randn(*shape: int, scale: float = 0.12) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float32) * scale

    weights: dict[str, torch.Tensor] = {
        "token_embedding.weight": randn(VOCAB_SIZE, HIDDEN_SIZE),
        "final_norm.weight": torch.tensor([0.91, 1.03, 0.97, 1.11], dtype=torch.float32),
    }
    head_dim = HIDDEN_SIZE // NUM_HEADS
    for layer in range(NUM_LAYERS):
        prefix = f"blocks.{layer}."
        weights[prefix + "attention_norm.weight"] = torch.ones(HIDDEN_SIZE) + randn(
            HIDDEN_SIZE, scale=0.03
        )
        weights[prefix + "feedforward_norm.weight"] = torch.ones(HIDDEN_SIZE) + randn(
            HIDDEN_SIZE, scale=0.03
        )
        weights[prefix + "attention.wq.weight"] = randn(NUM_HEADS * head_dim, HIDDEN_SIZE)
        weights[prefix + "attention.wk.weight"] = randn(NUM_KV_HEADS * head_dim, HIDDEN_SIZE)
        weights[prefix + "attention.wv.weight"] = randn(NUM_KV_HEADS * head_dim, HIDDEN_SIZE)
        weights[prefix + "attention.wo.weight"] = randn(HIDDEN_SIZE, HIDDEN_SIZE)
        if qk_norm:
            weights[prefix + "attention.q_norm.weight"] = torch.tensor([0.77, 1.23])
            weights[prefix + "attention.k_norm.weight"] = torch.tensor([1.17, 0.83])
        weights[prefix + "feedforward.w_gate.weight"] = randn(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        weights[prefix + "feedforward.w_up.weight"] = randn(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        weights[prefix + "feedforward.w_down.weight"] = randn(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    return weights
