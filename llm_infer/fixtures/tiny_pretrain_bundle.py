"""Tiny DenseBackbone export bundle writer for tests and the real-engine trace fixture."""

from __future__ import annotations

import hashlib
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tiny_pretrain_bundle(
    root: Path,
    *,
    qk_norm: bool = False,
    logit_soft_cap: float | None = None,
    kv_heads: int | None = NUM_KV_HEADS,
) -> Path:
    bundle = root / "tiny_pretrain_bundle"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.mkdir(exist_ok=True)
    config = {
        "name": "tiny-dense",
        "vocab_size": VOCAB_SIZE,
        "context_length": 128,
        "embedding_dim": HIDDEN_SIZE,
        "feedforward_dim": INTERMEDIATE_SIZE,
        "layers": NUM_LAYERS,
        "heads": NUM_HEADS,
        "kv_heads": kv_heads,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10_000.0,
        "tie_embeddings": True,
        "qk_norm": qk_norm,
        "z_loss_weight": 1e-4,
        "attention_kind": "gqa",
    }
    if logit_soft_cap is not None:
        config["logit_soft_cap"] = logit_soft_cap
    config_path = bundle / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    tokenizer_path = bundle / "tokenizer.json"
    tokenizer_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "model": {
                    "type": "WordLevel",
                    "vocab": {
                        "<pad>": 0,
                        "<bos>": 1,
                        "<eos>": 2,
                        "<unk>": 3,
                        **{f"tok_{idx}": idx for idx in range(4, VOCAB_SIZE)},
                    },
                    "unk_token": "<unk>",
                },
                "pre_tokenizer": {"type": "Whitespace"},
            }
        ),
        encoding="utf-8",
    )
    llm_infer_config = {
        "format": BUNDLE_FORMAT,
        "name": config["name"],
        "vocab_size": config["vocab_size"],
        "context_length": config["context_length"],
        "hidden_size": config["embedding_dim"],
        "intermediate_size": config["feedforward_dim"],
        "num_hidden_layers": config["layers"],
        "num_attention_heads": config["heads"],
        "num_key_value_heads": config["heads"] if kv_heads is None else kv_heads,
        "rms_norm_eps": config["rms_norm_eps"],
        "rope_theta": config["rope_theta"],
        "tie_word_embeddings": config["tie_embeddings"],
        "attention_kind": config["attention_kind"],
        "qk_norm": config["qk_norm"],
        "z_loss_weight": config["z_loss_weight"],
    }
    source_checkpoint_sha256 = "0" * 64
    state_dict = tiny_pretrain_state_dict(
        qk_norm=qk_norm,
        num_kv_heads=NUM_HEADS if kv_heads is None else kv_heads,
    )
    state_dict["lm_head.weight"] = state_dict["token_embedding.weight"]
    weights_path = bundle / "weights.pt"
    torch.save(
        {
            "format_version": 1,
            "format": BUNDLE_FORMAT,
            "key_format": BUNDLE_FORMAT,
            "metadata": {
                "key_format": BUNDLE_FORMAT,
                "target": "llm-infer",
                "state_dict_key": "dense_backbone",
            },
            "state_dict_key": "dense_backbone",
            "state_dict": state_dict,
            "model_config": config,
            "llm_infer_config": llm_infer_config,
            "checkpoint_step": 1,
            "source_checkpoint": "tiny-checkpoint.pt",
            "source_checkpoint_sha256": source_checkpoint_sha256,
        },
        weights_path,
    )
    readme_path = bundle / "README.md"
    readme_path.write_text(
        "# llm-infer Export Bundle\n\nTiny canonical test fixture.\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "format": BUNDLE_FORMAT,
        "target": "llm-infer",
        "weights_format": BUNDLE_FORMAT,
        "model_family": "DenseBackbone",
        "model": {
            "id": "tiny-dense",
            "name": "tiny-dense",
            "format": BUNDLE_FORMAT,
            "family": "DenseBackbone",
        },
        "tokenizer": {"path": "tokenizer.json", "format": "tokenizers-json"},
        "checkpoint_step": 1,
        "source_checkpoint": "tiny-checkpoint.pt",
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "files": {
            "config": {"path": "config.json", "sha256": _sha256(config_path)},
            "tokenizer": {"path": "tokenizer.json", "sha256": _sha256(tokenizer_path)},
            "weights": {"path": "weights.pt", "sha256": _sha256(weights_path)},
            "readme": {"path": "README.md", "sha256": _sha256(readme_path)},
        },
        "model_config": config,
        "llm_infer_config": llm_infer_config,
        "run_metadata": {},
    }
    (bundle / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return bundle


def tiny_pretrain_state_dict(
    *, qk_norm: bool = False, num_kv_heads: int = NUM_KV_HEADS
) -> dict[str, torch.Tensor]:
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
        weights[prefix + "attention.wk.weight"] = randn(num_kv_heads * head_dim, HIDDEN_SIZE)
        weights[prefix + "attention.wv.weight"] = randn(num_kv_heads * head_dim, HIDDEN_SIZE)
        weights[prefix + "attention.wo.weight"] = randn(HIDDEN_SIZE, HIDDEN_SIZE)
        if qk_norm:
            weights[prefix + "attention.q_norm.weight"] = torch.tensor([0.77, 1.23])
            weights[prefix + "attention.k_norm.weight"] = torch.tensor([1.17, 0.83])
        weights[prefix + "feedforward.w_gate.weight"] = randn(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        weights[prefix + "feedforward.w_up.weight"] = randn(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        weights[prefix + "feedforward.w_down.weight"] = randn(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    return weights
