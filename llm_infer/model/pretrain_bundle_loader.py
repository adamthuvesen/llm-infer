"""Load and validate esme-pretrain dense export bundles."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

BUNDLE_FORMAT = "llm_pretrain_dense_v1"
# The one bundle schema version this loader supports. esme-pretrain writes it to
# manifest.json (schema_version) and weights.pt (format_version); the contract and its
# compatibility policy live in esme-pretrain's docs/bundle-format.md.
SUPPORTED_BUNDLE_SCHEMA_VERSION = 1


class PretrainBundleError(ValueError):
    """Raised when an esme-pretrain export bundle is malformed."""


@dataclass(frozen=True)
class PretrainDenseConfig:
    """The architecture fields needed for inference."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    qk_norm: bool
    logit_soft_cap: float | None = None

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, raw: Mapping[str, object]) -> PretrainDenseConfig:
        vocab_size = _positive_int(raw, "vocab_size")
        hidden_size = _positive_int(raw, "embedding_dim")
        intermediate_size = _positive_int(raw, "feedforward_dim")
        num_hidden_layers = _positive_int(raw, "layers")
        num_attention_heads = _positive_int(raw, "heads")
        num_key_value_heads = _positive_int(raw, "kv_heads")
        rms_norm_eps = _positive_float(raw, "rms_norm_eps")
        rope_theta = _positive_float(raw, "rope_theta")
        tie_word_embeddings = _bool_field(raw, "tie_embeddings")
        qk_norm = _bool_field(raw, "qk_norm")
        logit_soft_cap = _optional_non_negative_float(raw, "logit_soft_cap")

        if hidden_size % num_attention_heads != 0:
            raise PretrainBundleError(
                "config.json hidden_size must be divisible by num_attention_heads; "
                f"got {hidden_size} and {num_attention_heads}"
            )
        if num_attention_heads % num_key_value_heads != 0:
            raise PretrainBundleError(
                "config.json num_attention_heads must be divisible by num_key_value_heads; "
                f"got {num_attention_heads} and {num_key_value_heads}"
            )

        return cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            tie_word_embeddings=tie_word_embeddings,
            qk_norm=qk_norm,
            logit_soft_cap=logit_soft_cap,
        )


@dataclass(frozen=True)
class _WeightSpec:
    target: str
    source: str
    shape: tuple[int, ...]


def required_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise PretrainBundleError(f"bundle is missing required file {name}")
    return path


def read_json_object(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PretrainBundleError(f"{path.name} is not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise PretrainBundleError(f"{path.name} must contain a JSON object")
    return raw


def require_manifest_format(manifest: Mapping[str, object], path: Path) -> None:
    declared = manifest.get("format")
    if declared != BUNDLE_FORMAT:
        raise PretrainBundleError(
            f"{path.name} must identify format {BUNDLE_FORMAT!r}; found {declared!r}"
        )


def require_supported_schema_version(manifest: Mapping[str, object], path: Path) -> None:
    """Reject a manifest that declares a schema version this loader does not support.

    Bundles that predate the version field load as v1.
    """
    declared = manifest.get("schema_version")
    if declared is None:
        return
    if isinstance(declared, bool) or declared != SUPPORTED_BUNDLE_SCHEMA_VERSION:
        raise PretrainBundleError(
            f"{path.name} declares schema_version {declared!r}; "
            f"this loader supports {SUPPORTED_BUNDLE_SCHEMA_VERSION}"
        )


def require_supported_weights_version(metadata: Mapping[str, object], path: Path) -> None:
    """Reject weights metadata that declares a format version this loader does not support."""
    declared = metadata.get("format_version")
    if declared is None:
        return
    if isinstance(declared, bool) or declared != SUPPORTED_BUNDLE_SCHEMA_VERSION:
        raise PretrainBundleError(
            f"{path.name} declares format_version {declared!r}; "
            f"this loader supports {SUPPORTED_BUNDLE_SCHEMA_VERSION}"
        )


def resolve_tokenizer_path(root: Path, manifest: Mapping[str, object]) -> Path:
    tokenizer_entry = manifest.get("tokenizer")
    if not isinstance(tokenizer_entry, dict):
        raise PretrainBundleError("manifest.json tokenizer must be an object")
    tokenizer_path = tokenizer_entry.get("path")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise PretrainBundleError("manifest.json tokenizer.path must be a non-empty string")

    rel = Path(tokenizer_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise PretrainBundleError(f"tokenizer path must stay inside the bundle: {rel}")
    path = root / rel
    if not path.is_file():
        raise PretrainBundleError(f"tokenizer file is missing: {rel}")
    return path


def read_weights(
    path: Path, device: torch.device | str
) -> tuple[Mapping[str, torch.Tensor], Mapping[str, object]]:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except Exception as exc:
        raise PretrainBundleError(f"failed to load {path.name}: {exc}") from exc

    if not isinstance(payload, dict):
        raise PretrainBundleError("weights.pt must contain a dictionary payload")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not _is_tensor_state_dict(state_dict):
        raise PretrainBundleError("weights.pt must contain a tensor state dict under 'state_dict'")

    metadata: dict[str, object] = {}
    nested = payload.get("metadata")
    if nested is not None and not isinstance(nested, dict):
        raise PretrainBundleError("weights.pt metadata must be an object")
    if isinstance(nested, dict):
        metadata.update({str(key): value for key, value in nested.items()})
    for key in ("format_version", "key_format"):
        if key in payload:
            metadata[key] = payload[key]
    return {str(name): tensor for name, tensor in state_dict.items()}, metadata


def _is_tensor_state_dict(value: Mapping[object, object]) -> bool:
    return bool(value) and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor) for key, tensor in value.items()
    )


def require_weight_key_format(metadata: Mapping[str, object], path: Path) -> None:
    declared = metadata.get("key_format")
    if declared != BUNDLE_FORMAT:
        raise PretrainBundleError(
            f"{path.name} metadata must identify dense key format {BUNDLE_FORMAT!r}; "
            f"found {declared!r}"
        )


def normalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    config: PretrainDenseConfig,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    layer_norm_specs = _layer_norm_specs(config)
    qk_norm_specs = _qk_norm_specs(config)
    attention_projection_specs, mlp_projection_specs = _projection_specs(config)

    for spec in _root_tensor_specs(config):
        _copy_weight(
            weights,
            spec.target,
            state_dict,
            spec.source,
            spec.shape,
            dtype=dtype,
            device=device,
        )

    for layer in range(config.num_hidden_layers):
        for spec in layer_norm_specs:
            _copy_layer_weight(
                weights,
                layer,
                spec.target,
                state_dict,
                spec.source,
                spec.shape,
                dtype=dtype,
                device=device,
            )
        for spec in attention_projection_specs:
            _copy_projection(
                weights,
                layer,
                spec.target,
                state_dict,
                spec.source,
                spec.shape,
                dtype=dtype,
                device=device,
            )
        for spec in qk_norm_specs:
            _copy_layer_weight(
                weights,
                layer,
                spec.target,
                state_dict,
                spec.source,
                spec.shape,
                dtype=dtype,
                device=device,
            )
        for spec in mlp_projection_specs:
            _copy_projection(
                weights,
                layer,
                spec.target,
                state_dict,
                spec.source,
                spec.shape,
                dtype=dtype,
                device=device,
            )

    return weights


def _root_tensor_specs(config: PretrainDenseConfig) -> tuple[_WeightSpec, ...]:
    specs = [
        _WeightSpec(
            "embed_tokens.weight",
            "token_embedding.weight",
            (config.vocab_size, config.hidden_size),
        ),
        _WeightSpec(
            "norm.weight",
            "final_norm.weight",
            (config.hidden_size,),
        ),
    ]
    if not config.tie_word_embeddings:
        specs.append(
            _WeightSpec(
                "lm_head.weight",
                "lm_head.weight",
                (config.vocab_size, config.hidden_size),
            )
        )
    return tuple(specs)


def _layer_norm_specs(config: PretrainDenseConfig) -> tuple[_WeightSpec, ...]:
    return (
        _WeightSpec(
            "input_norm.weight",
            "attention_norm.weight",
            (config.hidden_size,),
        ),
        _WeightSpec(
            "post_attention_norm.weight",
            "feedforward_norm.weight",
            (config.hidden_size,),
        ),
    )


def _qk_norm_specs(config: PretrainDenseConfig) -> tuple[_WeightSpec, ...]:
    if not config.qk_norm:
        return ()
    return (
        _WeightSpec(
            "attn.q_norm.weight",
            "attention.q_norm.weight",
            (config.head_dim,),
        ),
        _WeightSpec(
            "attn.k_norm.weight",
            "attention.k_norm.weight",
            (config.head_dim,),
        ),
    )


def _projection_specs(
    config: PretrainDenseConfig,
) -> tuple[tuple[_WeightSpec, ...], tuple[_WeightSpec, ...]]:
    head_dim = config.head_dim
    attention = (
        _WeightSpec(
            "attn.q_proj",
            "attention.wq",
            (config.num_attention_heads * head_dim, config.hidden_size),
        ),
        _WeightSpec(
            "attn.k_proj",
            "attention.wk",
            (config.num_key_value_heads * head_dim, config.hidden_size),
        ),
        _WeightSpec(
            "attn.v_proj",
            "attention.wv",
            (config.num_key_value_heads * head_dim, config.hidden_size),
        ),
        _WeightSpec(
            "attn.o_proj",
            "attention.wo",
            (config.hidden_size, config.hidden_size),
        ),
    )
    mlp = (
        _WeightSpec(
            "mlp.gate_proj",
            "feedforward.w_gate",
            (config.intermediate_size, config.hidden_size),
        ),
        _WeightSpec(
            "mlp.up_proj",
            "feedforward.w_up",
            (config.intermediate_size, config.hidden_size),
        ),
        _WeightSpec(
            "mlp.down_proj",
            "feedforward.w_down",
            (config.hidden_size, config.intermediate_size),
        ),
    )
    return attention, mlp


def _copy_layer_weight(
    weights: dict[str, torch.Tensor],
    layer: int,
    target: str,
    state_dict: Mapping[str, torch.Tensor],
    source: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    _copy_weight(
        weights,
        f"layers.{layer}.{target}",
        state_dict,
        f"blocks.{layer}.{source}",
        shape,
        dtype=dtype,
        device=device,
    )


def _copy_projection(
    weights: dict[str, torch.Tensor],
    layer: int,
    target: str,
    state_dict: Mapping[str, torch.Tensor],
    source: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    _copy_weight(
        weights,
        f"layers.{layer}.{target}.weight",
        state_dict,
        f"blocks.{layer}.{source}.weight",
        shape,
        dtype=dtype,
        device=device,
    )


def _copy_weight(
    weights: dict[str, torch.Tensor],
    target: str,
    state_dict: Mapping[str, torch.Tensor],
    source: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    tensor = state_dict.get(source)
    if tensor is None:
        raise PretrainBundleError(f"weights.pt is missing required tensor for {target}")
    weights[target] = _validated_tensor(source, tensor, shape, dtype=dtype, device=device)


def _validated_tensor(
    alias: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    if tuple(tensor.shape) != shape:
        raise PretrainBundleError(
            f"weights.pt tensor {alias!r} has shape {tuple(tensor.shape)}, expected {shape}"
        )
    if not tensor.is_floating_point():
        raise PretrainBundleError(f"weights.pt tensor {alias!r} must be floating point")
    return tensor.detach().to(device=device, dtype=dtype)


def _positive_int(
    raw: Mapping[str, object],
    name: str,
) -> int:
    value = _required_field(raw, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PretrainBundleError(f"config.json {name} must be a positive integer")
    return value


def _positive_float(
    raw: Mapping[str, object],
    name: str,
) -> float:
    value = _required_field(raw, name)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise PretrainBundleError(f"config.json {name} must be a finite positive number")
    if value <= 0:
        raise PretrainBundleError(f"config.json {name} must be a finite positive number")
    return float(value)


def _optional_non_negative_float(
    raw: Mapping[str, object],
    name: str,
) -> float | None:
    value = raw.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise PretrainBundleError(f"config.json {name} must be a finite non-negative number")
    if value < 0:
        raise PretrainBundleError(f"config.json {name} must be a finite non-negative number")
    return float(value)


def _bool_field(
    raw: Mapping[str, object],
    name: str,
) -> bool:
    value = _required_field(raw, name)
    if not isinstance(value, bool):
        raise PretrainBundleError(f"config.json {name} must be a boolean")
    return value


def _required_field(raw: Mapping[str, object], name: str) -> object:
    if name not in raw:
        raise PretrainBundleError(f"config.json is missing required field {name}")
    return raw[name]
