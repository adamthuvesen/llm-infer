"""Convert an Esme ``llm_pretrain_dense_v1`` bundle to an HF Qwen3 checkpoint.

``Esme-214M-Chat`` is architecturally Qwen3 (GQA + per-head QK-norm applied before RoPE,
SwiGLU MLP, RMSNorm, tied embeddings, no biases, logit soft-cap disabled), so its weights map
cleanly onto ``Qwen3ForCausalLM`` — natively supported by HF ``transformers`` (>=4.51) and vLLM.
No custom modeling file or vLLM plugin is needed.

This writes a standard HF checkpoint directory: ``config.json`` with
``architectures: ["Qwen3ForCausalLM"]``, ``model.safetensors`` with Qwen3 parameter names, and
the bundle's own ``tokenizer.json`` copied verbatim. The conversion is a pure key remap plus a
``head_dim`` reshape for the layer-norm-free projections — every value is copied bit-for-bit, so
the converted model must reproduce ``PretrainBundleModel.logits()`` exactly (the parity test is
the make-or-break gate).

    uv run scripts/convert_esme_to_hf.py \
        --bundle /path/to/esme-214m-chat \
        --out /path/to/esme-214m-chat-hf
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from llm_infer.model.pretrain_bundle_loader import (
    PretrainBundleError,
    PretrainDenseConfig,
    read_json_object,
    read_weights,
    required_file,
)

# The bundle export uses these canonical per-block names (see pretrain_bundle_loader aliases and
# the real Esme-214M-Chat state dict). Listed here as the source of truth for the remap so a
# changed export key fails loudly rather than silently dropping a tensor.
BUNDLE_TOKEN_EMBEDDING = "token_embedding.weight"
BUNDLE_FINAL_NORM = "final_norm.weight"
BUNDLE_LM_HEAD = "lm_head.weight"


def _bundle_block_key(layer: int, suffix: str) -> str:
    return f"blocks.{layer}.{suffix}"


def build_hf_config(
    config: PretrainDenseConfig, *, max_position_embeddings: int
) -> dict[str, object]:
    """The Qwen3 ``config.json`` dict the bundle architecture maps onto.

    ``head_dim`` is pinned explicitly (it equals ``hidden_size // num_attention_heads`` for Esme
    but Qwen3 stores it as its own field, and vLLM reads it). Biases off, embeddings tied, sliding
    window disabled — all matching the bundle.
    """
    cap = config.logit_soft_cap
    if cap not in (None, 0.0):
        raise PretrainBundleError(
            "logit_soft_cap is enabled on this bundle; Qwen3 has no final logit soft cap, so the "
            f"mapping does not hold (got {cap}). This needs escalation, not a silent conversion."
        )
    return {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_word_embeddings,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "sliding_window": None,
        "use_sliding_window": False,
        "use_cache": True,
        "torch_dtype": "float32",
    }


def remap_state_dict(
    bundle_state: dict[str, torch.Tensor], config: PretrainDenseConfig
) -> dict[str, torch.Tensor]:
    """Remap bundle weight names onto Qwen3 parameter names, copying values bit-for-bit.

    Per layer: ``blocks.{i}.attention.{wq,wk,wv,wo}`` map to
    ``model.layers.{i}.self_attn.{q,k,v,o}_proj``, ``attention.{q,k}_norm`` to
    ``self_attn.{q,k}_norm``, ``attention_norm`` to ``input_layernorm``, ``feedforward_norm`` to
    ``post_attention_layernorm``, and ``feedforward.{w_gate,w_up,w_down}`` to
    ``mlp.{gate,up,down}_proj``. Roots: ``token_embedding`` to ``model.embed_tokens``,
    ``final_norm`` to ``model.norm``. ``lm_head`` is tied to the embedding and omitted.
    """
    out: dict[str, torch.Tensor] = {}

    def take(name: str) -> torch.Tensor:
        tensor = bundle_state.get(name)
        if tensor is None:
            raise PretrainBundleError(f"bundle weights.pt is missing required tensor {name!r}")
        return tensor.detach().clone().to(torch.float32)

    out["model.embed_tokens.weight"] = take(BUNDLE_TOKEN_EMBEDDING)
    out["model.norm.weight"] = take(BUNDLE_FINAL_NORM)

    # Tied embeddings: the bundle ships lm_head.weight equal to the embedding; assert that and let
    # HF tie at load. A divergent lm_head would mean the export is not actually tied and the
    # tie_word_embeddings config flag would be wrong — fail loudly.
    if config.tie_word_embeddings and BUNDLE_LM_HEAD in bundle_state:
        lm_head = bundle_state[BUNDLE_LM_HEAD].detach().to(torch.float32)
        if not torch.equal(lm_head, out["model.embed_tokens.weight"]):
            raise PretrainBundleError(
                "bundle declares tie_embeddings but lm_head.weight != token_embedding.weight; "
                "the mapping to a tied Qwen3 head does not hold — escalate."
            )

    projection_map = {
        "attention.wq.weight": "self_attn.q_proj.weight",
        "attention.wk.weight": "self_attn.k_proj.weight",
        "attention.wv.weight": "self_attn.v_proj.weight",
        "attention.wo.weight": "self_attn.o_proj.weight",
        "attention.q_norm.weight": "self_attn.q_norm.weight",
        "attention.k_norm.weight": "self_attn.k_norm.weight",
        "attention_norm.weight": "input_layernorm.weight",
        "feedforward_norm.weight": "post_attention_layernorm.weight",
        "feedforward.w_gate.weight": "mlp.gate_proj.weight",
        "feedforward.w_up.weight": "mlp.up_proj.weight",
        "feedforward.w_down.weight": "mlp.down_proj.weight",
    }
    for layer in range(config.num_hidden_layers):
        for bundle_suffix, hf_suffix in projection_map.items():
            source = _bundle_block_key(layer, bundle_suffix)
            out[f"model.layers.{layer}.{hf_suffix}"] = take(source)

    return out


def _validate_bundle_manifest(bundle: Path) -> None:
    manifest = read_json_object(required_file(bundle, "manifest.json"))
    model = manifest.get("model")
    if not isinstance(model, dict) or model.get("name") != "Esme-214M-Chat":
        raise PretrainBundleError(
            f"expected an Esme-214M-Chat bundle, found model={model!r} in {bundle}/manifest.json"
        )


def convert(bundle: Path, out: Path, *, max_position_embeddings: int) -> dict[str, object]:
    """Write an HF Qwen3 checkpoint from the bundle. Returns the written config dict."""
    from safetensors.torch import save_file

    _validate_bundle_manifest(bundle)
    config = PretrainDenseConfig.from_json(read_json_object(required_file(bundle, "config.json")))
    bundle_state, _metadata = read_weights(required_file(bundle, "weights.pt"), "cpu")

    hf_config = build_hf_config(config, max_position_embeddings=max_position_embeddings)
    hf_state = {
        name: tensor.contiguous()
        for name, tensor in remap_state_dict(dict(bundle_state), config).items()
    }

    expected = 2 + config.num_hidden_layers * 11  # 2 roots + 11 per block (lm_head tied)
    if len(hf_state) != expected:
        raise PretrainBundleError(
            f"remapped {len(hf_state)} tensors, expected {expected}; remap is incomplete"
        )

    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(hf_config, indent=2) + "\n", encoding="utf-8")
    save_file(hf_state, str(out / "model.safetensors"), metadata={"format": "pt"})
    shutil.copyfile(required_file(bundle, "tokenizer.json"), out / "tokenizer.json")
    return hf_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an Esme bundle to an HF Qwen3 checkpoint."
    )
    parser.add_argument("--bundle", type=Path, required=True, help="path to the Esme bundle dir")
    parser.add_argument("--out", type=Path, required=True, help="output HF checkpoint dir")
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=1024,
        help="max_position_embeddings for the HF config (>= bundle context length)",
    )
    args = parser.parse_args()
    config = convert(
        args.bundle.expanduser(),
        args.out.expanduser(),
        max_position_embeddings=args.max_position_embeddings,
    )
    print(f"wrote HF Qwen3 checkpoint to {args.out}")
    print(
        f"  layers={config['num_hidden_layers']} hidden={config['hidden_size']} "
        f"heads={config['num_attention_heads']}/{config['num_key_value_heads']} "
        f"head_dim={config['head_dim']} tied={config['tie_word_embeddings']}"
    )


if __name__ == "__main__":
    main()
