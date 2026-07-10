"""Load and validate esme-pretrain dense export bundles."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

BUNDLE_FORMAT = "llm_pretrain_dense_v1"
# The one bundle schema version this loader supports. esme-pretrain writes it to
# manifest.json (schema_version) and weights.pt (format_version); the contract and its
# compatibility policy live in esme-pretrain's docs/bundle-format.md.
SUPPORTED_BUNDLE_SCHEMA_VERSION = 1
_MISSING = object()


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
        hidden_size = _positive_int(
            raw, "embedding_dim", aliases=("hidden_size", "d_model", "n_embd")
        )
        intermediate_size = _positive_int(
            raw,
            "feedforward_dim",
            aliases=(
                "intermediate_size",
                "mlp_hidden_size",
                "ffn_hidden_size",
                "feed_forward_size",
            ),
        )
        num_hidden_layers = _positive_int(
            raw, "layers", aliases=("num_hidden_layers", "n_layers", "num_layers", "n_layer")
        )
        num_attention_heads = _positive_int(
            raw, "heads", aliases=("num_attention_heads", "n_heads", "num_heads", "n_head")
        )
        num_key_value_heads = _positive_int(
            raw,
            "kv_heads",
            aliases=("num_key_value_heads", "n_kv_heads", "num_kv_heads", "n_key_value_heads"),
            default=num_attention_heads,
        )
        rms_norm_eps = _positive_float(raw, "rms_norm_eps", aliases=("norm_eps",), default=1e-6)
        rope_theta = _positive_float(raw, "rope_theta", aliases=("rope_base",), default=10_000.0)
        tie_word_embeddings = _bool_field(raw, "tie_embeddings", aliases=("tie_word_embeddings",))
        qk_norm = _bool_field(raw, "qk_norm", default=False)
        logit_soft_cap = _optional_non_negative_float(
            raw, "logit_soft_cap", aliases=("final_logit_softcapping",)
        )

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
class _TensorSpec:
    target: str
    aliases: tuple[str, ...]
    shape: tuple[int, ...]


@dataclass(frozen=True)
class _ProjectionSpec:
    target: str
    bases: tuple[str, ...]
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
    candidates = [
        _string_at(manifest, "format"),
        _string_at(manifest, "bundle_format"),
        _string_at(manifest, "model_format"),
        _string_at(manifest, "architecture"),
        _nested_string_at(manifest, "model", "format"),
        _nested_string_at(manifest, "model", "architecture"),
    ]
    if BUNDLE_FORMAT not in candidates:
        found = ", ".join(sorted({item for item in candidates if item})) or "none"
        raise PretrainBundleError(
            f"{path.name} must identify format {BUNDLE_FORMAT!r}; found {found}"
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
    tokenizer_path = _string_at(manifest, "tokenizer_path")
    tokenizer_entry = manifest.get("tokenizer")
    if isinstance(tokenizer_entry, str):
        tokenizer_path = tokenizer_entry
    elif isinstance(tokenizer_entry, dict):
        nested = tokenizer_entry.get("path")
        if not isinstance(nested, str) or not nested:
            raise PretrainBundleError("manifest.json tokenizer.path must be a non-empty string")
        tokenizer_path = nested
    elif tokenizer_entry is not None:
        raise PretrainBundleError("manifest.json tokenizer must be a path string or object")

    rel = Path(tokenizer_path or "tokenizer.json")
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

    state_candidate = _state_dict_candidate(payload)
    if state_candidate is None:
        raise PretrainBundleError(
            "weights.pt must contain a tensor state dict under 'state_dict' or 'model_state_dict'"
        )

    metadata = _metadata_candidate(payload)
    return state_candidate, metadata


def _state_dict_candidate(payload: Mapping[object, object]) -> Mapping[str, torch.Tensor] | None:
    for key in ("state_dict", "model_state_dict", "weights"):
        value = payload.get(key)
        if isinstance(value, dict) and _is_tensor_state_dict(value):
            return {str(name): tensor for name, tensor in value.items()}
    if _is_tensor_state_dict(payload):
        return {str(name): tensor for name, tensor in payload.items()}
    return None


def _metadata_candidate(payload: Mapping[object, object]) -> Mapping[str, object]:
    metadata: dict[str, object] = {}
    nested = payload.get("metadata")
    if isinstance(nested, dict):
        metadata.update({str(key): value for key, value in nested.items()})
    for key, value in payload.items():
        if key not in {"state_dict", "model_state_dict", "weights", "metadata"}:
            metadata[str(key)] = value
    return metadata


def _is_tensor_state_dict(value: Mapping[object, object]) -> bool:
    return bool(value) and all(
        isinstance(key, str) and isinstance(tensor, torch.Tensor) for key, tensor in value.items()
    )


def require_weight_key_format(metadata: Mapping[str, object], path: Path) -> None:
    fields = ("key_format", "dense_key_format", "state_dict_format", "weights_format", "format")
    candidates = [metadata.get(field) for field in fields]
    if BUNDLE_FORMAT not in {value for value in candidates if isinstance(value, str)}:
        found = (
            ", ".join(sorted({value for value in candidates if isinstance(value, str)})) or "none"
        )
        raise PretrainBundleError(
            f"{path.name} metadata must identify dense key format {BUNDLE_FORMAT!r}; found {found}"
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
            _root_aliases(*spec.aliases),
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
                spec.aliases,
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
                spec.bases,
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
                spec.aliases,
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
                spec.bases,
                spec.shape,
                dtype=dtype,
                device=device,
            )

    return weights


def _root_tensor_specs(config: PretrainDenseConfig) -> tuple[_TensorSpec, ...]:
    specs = [
        _TensorSpec(
            "embed_tokens.weight",
            ("token_embedding.weight", "embed_tokens.weight", "tok_embeddings.weight"),
            (config.vocab_size, config.hidden_size),
        ),
        _TensorSpec(
            "norm.weight",
            ("final_norm.weight", "norm.weight", "ln_f.weight"),
            (config.hidden_size,),
        ),
    ]
    if not config.tie_word_embeddings:
        specs.append(
            _TensorSpec(
                "lm_head.weight",
                ("lm_head.weight", "output.weight", "output_projection.weight"),
                (config.vocab_size, config.hidden_size),
            )
        )
    return tuple(specs)


def _layer_norm_specs(config: PretrainDenseConfig) -> tuple[_TensorSpec, ...]:
    return (
        _TensorSpec(
            "input_norm.weight",
            ("attention_norm.weight", "attn_norm.weight", "input_layernorm.weight", "norm1.weight"),
            (config.hidden_size,),
        ),
        _TensorSpec(
            "post_attention_norm.weight",
            (
                "feedforward_norm.weight",
                "ffn_norm.weight",
                "mlp_norm.weight",
                "post_attention_layernorm.weight",
                "norm2.weight",
            ),
            (config.hidden_size,),
        ),
    )


def _qk_norm_specs(config: PretrainDenseConfig) -> tuple[_TensorSpec, ...]:
    if not config.qk_norm:
        return ()
    return (
        _TensorSpec(
            "attn.q_norm.weight",
            (
                "attention.q_norm.weight",
                "attn.q_norm.weight",
                "self_attn.q_norm.weight",
            ),
            (config.head_dim,),
        ),
        _TensorSpec(
            "attn.k_norm.weight",
            (
                "attention.k_norm.weight",
                "attn.k_norm.weight",
                "self_attn.k_norm.weight",
            ),
            (config.head_dim,),
        ),
    )


def _projection_specs(
    config: PretrainDenseConfig,
) -> tuple[tuple[_ProjectionSpec, ...], tuple[_ProjectionSpec, ...]]:
    head_dim = config.head_dim
    attention = (
        _ProjectionSpec(
            "attn.q_proj",
            ("attention.wq", "attention.q_proj", "attn.q_proj", "self_attn.q_proj", "attn.wq"),
            (config.num_attention_heads * head_dim, config.hidden_size),
        ),
        _ProjectionSpec(
            "attn.k_proj",
            ("attention.wk", "attention.k_proj", "attn.k_proj", "self_attn.k_proj", "attn.wk"),
            (config.num_key_value_heads * head_dim, config.hidden_size),
        ),
        _ProjectionSpec(
            "attn.v_proj",
            ("attention.wv", "attention.v_proj", "attn.v_proj", "self_attn.v_proj", "attn.wv"),
            (config.num_key_value_heads * head_dim, config.hidden_size),
        ),
        _ProjectionSpec(
            "attn.o_proj",
            ("attention.wo", "attention.o_proj", "attn.o_proj", "self_attn.o_proj", "attn.wo"),
            (config.hidden_size, config.hidden_size),
        ),
    )
    mlp = (
        _ProjectionSpec(
            "mlp.gate_proj",
            ("feedforward.w_gate", "feed_forward.w1", "ffn.w1", "mlp.gate_proj", "mlp.w1"),
            (config.intermediate_size, config.hidden_size),
        ),
        _ProjectionSpec(
            "mlp.up_proj",
            ("feedforward.w_up", "feed_forward.w3", "ffn.w3", "mlp.up_proj", "mlp.w3"),
            (config.intermediate_size, config.hidden_size),
        ),
        _ProjectionSpec(
            "mlp.down_proj",
            ("feedforward.w_down", "feed_forward.w2", "ffn.w2", "mlp.down_proj", "mlp.w2"),
            (config.hidden_size, config.intermediate_size),
        ),
    )
    return attention, mlp


def _copy_layer_weight(
    weights: dict[str, torch.Tensor],
    layer: int,
    target: str,
    state_dict: Mapping[str, torch.Tensor],
    suffixes: Sequence[str],
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    _copy_weight(
        weights,
        f"layers.{layer}.{target}",
        state_dict,
        _layer_aliases(layer, suffixes),
        shape,
        dtype=dtype,
        device=device,
    )


def _copy_projection(
    weights: dict[str, torch.Tensor],
    layer: int,
    target: str,
    state_dict: Mapping[str, torch.Tensor],
    bases: Sequence[str],
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    _copy_weight(
        weights,
        f"layers.{layer}.{target}.weight",
        state_dict,
        _layer_aliases(layer, tuple(base + ".weight" for base in bases)),
        shape,
        dtype=dtype,
        device=device,
    )
    _copy_weight(
        weights,
        f"layers.{layer}.{target}.bias",
        state_dict,
        _layer_aliases(layer, tuple(base + ".bias" for base in bases)),
        (shape[0],),
        dtype=dtype,
        device=device,
        required=False,
    )


def _copy_weight(
    weights: dict[str, torch.Tensor],
    target: str,
    state_dict: Mapping[str, torch.Tensor],
    aliases: Iterable[str],
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    required: bool = True,
) -> None:
    for alias in aliases:
        tensor = state_dict.get(alias)
        if tensor is not None:
            weights[target] = _validated_tensor(alias, tensor, shape, dtype=dtype, device=device)
            return
    if required:
        raise PretrainBundleError(f"weights.pt is missing required tensor for {target}")


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


def _root_aliases(*suffixes: str) -> tuple[str, ...]:
    prefixes = ("", "model.", "backbone.", "transformer.")
    return tuple(prefix + suffix for prefix in prefixes for suffix in suffixes)


def _layer_aliases(layer: int, suffixes: Sequence[str]) -> tuple[str, ...]:
    prefixes = (
        f"blocks.{layer}.",
        f"layers.{layer}.",
        f"backbone.blocks.{layer}.",
        f"model.layers.{layer}.",
        f"backbone.layers.{layer}.",
        f"transformer.h.{layer}.",
    )
    return tuple(prefix + suffix for prefix in prefixes for suffix in suffixes)


def _positive_int(
    raw: Mapping[str, object],
    name: str,
    *,
    aliases: Sequence[str] = (),
    default: int | None = None,
) -> int:
    value = _first_present(raw, (name, *aliases), default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PretrainBundleError(f"config.json {name} must be a positive integer")
    return value


def _positive_float(
    raw: Mapping[str, object],
    name: str,
    *,
    aliases: Sequence[str] = (),
    default: float | None = None,
) -> float:
    value = _first_present(raw, (name, *aliases), default)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise PretrainBundleError(f"config.json {name} must be a finite positive number")
    if value <= 0:
        raise PretrainBundleError(f"config.json {name} must be a finite positive number")
    return float(value)


def _optional_non_negative_float(
    raw: Mapping[str, object],
    name: str,
    *,
    aliases: Sequence[str] = (),
) -> float | None:
    value = _first_present(raw, (name, *aliases), None)
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
    *,
    aliases: Sequence[str] = (),
    default: bool | object = _MISSING,
) -> bool:
    value = _first_present(raw, (name, *aliases), default)
    if not isinstance(value, bool):
        raise PretrainBundleError(f"config.json {name} must be a boolean")
    return value


def _first_present(
    raw: Mapping[str, object],
    names: Sequence[str],
    default: int | float | bool | None | object = _MISSING,
) -> object:
    for name in names:
        if name in raw:
            return raw[name]
    if default is not _MISSING:
        return default
    raise PretrainBundleError(f"config.json is missing required field {names[0]}")


def _string_at(raw: Mapping[str, object], key: str) -> str | None:
    value = raw.get(key)
    return value if isinstance(value, str) else None


def _nested_string_at(raw: Mapping[str, object], outer: str, inner: str) -> str | None:
    nested = raw.get(outer)
    if not isinstance(nested, dict):
        return None
    value = nested.get(inner)
    return value if isinstance(value, str) else None
