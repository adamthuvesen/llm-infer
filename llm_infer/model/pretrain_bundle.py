"""Loader and correctness-only forward pass for llm-pretrain dense export bundles."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from llm_infer.kernels.base import AttentionBackend
from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.profiling import TimingProfiler

BUNDLE_FORMAT = "llm_pretrain_dense_v1"
_MISSING = object()


class PretrainBundleError(ValueError):
    """Raised when an llm-pretrain export bundle is malformed."""


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


class PretrainBundleModel:
    """Full-recompute inference for ``llm_pretrain_dense_v1`` bundles.

    This is a correctness bridge for exported llm-pretrain DenseBackbone weights. The
    cached methods participate in the serving engine's block accounting, but they keep
    request token history and recompute logits rather than writing real K/V pages. That
    makes DenseBackbone usable through the same runtime path as optimized backends
    without pretending this v1 path is fast.
    """

    def __init__(
        self,
        *,
        weights: Mapping[str, torch.Tensor],
        config: PretrainDenseConfig,
        tokenizer_path: Path,
        backend: AttentionBackend,
        dtype: torch.dtype,
    ) -> None:
        self.w = dict(weights)
        self.config = config
        self.tokenizer_path = tokenizer_path
        self.backend = backend
        self.dtype = dtype
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.rms_eps = config.rms_norm_eps
        self.rope_theta = config.rope_theta
        self.tie_word_embeddings = config.tie_word_embeddings
        self.qk_norm = config.qk_norm
        self.device = self.w["embed_tokens.weight"].device
        self.profiler: TimingProfiler | None = None
        self._history_by_table: dict[int, list[int]] = {}

    @classmethod
    def load(
        cls,
        bundle_path: Path | str,
        *,
        dtype: torch.dtype = torch.float32,
        backend: AttentionBackend | None = None,
        device: torch.device | str = "cpu",
    ) -> PretrainBundleModel:
        """Load and validate an exported llm-pretrain dense bundle."""
        root = Path(bundle_path)
        if not root.is_dir():
            raise PretrainBundleError(f"bundle path must be a directory: {root}")

        manifest_path = _required_file(root, "manifest.json")
        config_path = _required_file(root, "config.json")
        weights_path = _required_file(root, "weights.pt")

        manifest = _read_json_object(manifest_path)
        _require_manifest_format(manifest, manifest_path)
        tokenizer_path = _resolve_tokenizer_path(root, manifest)
        _read_json_object(tokenizer_path)

        config = PretrainDenseConfig.from_json(_read_json_object(config_path))
        state_dict, metadata = _read_weights(weights_path, device)
        _require_weight_key_format(metadata, weights_path)
        weights = _normalize_state_dict(state_dict, config, dtype=dtype, device=device)

        return cls(
            weights=weights,
            config=config,
            tokenizer_path=tokenizer_path,
            backend=backend or TorchNaiveAttention(),
            dtype=dtype,
        )

    @torch.no_grad()
    def logits(self, token_ids: list[int]) -> torch.Tensor:
        """Next-token logits for every input position. Shape ``(seq_len, vocab_size)``."""
        self._validate_token_ids(token_ids)
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        hidden = self.w["embed_tokens.weight"][ids].to(self.dtype)

        cos, sin = self._rope_tables(len(token_ids))
        for layer in range(self.num_layers):
            hidden = self._decoder_layer(hidden, layer, cos, sin)

        hidden = _rms_norm(hidden, self.w["norm.weight"], self.rms_eps)
        return hidden @ self._lm_head().T

    @torch.no_grad()
    def generate_ids(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int,
        eos_token_ids: set[int] | frozenset[int] = frozenset(),
    ) -> list[int]:
        """Greedy token-id generation. Returns generated ids only."""
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be >= 1; got {max_new_tokens}")

        tokens = list(prompt_ids)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            next_id = int(torch.argmax(self.logits(tokens)[-1]).item())
            tokens.append(next_id)
            generated.append(next_id)
            if next_id in eos_token_ids:
                break
        return generated

    @torch.no_grad()
    def prefill(
        self, prompt_ids: list[int], cache: PagedKVCache, table: BlockTable
    ) -> torch.Tensor:
        """Correctness-first prefill: reserve blocks, remember prompt ids, recompute logits."""
        del cache
        self._validate_token_ids(prompt_ids)
        if table.length != 0:
            raise PretrainBundleError(f"prefill expected an empty table; got length {table.length}")
        table.reserve(len(prompt_ids))
        table.length = len(prompt_ids)
        self._history_by_table[id(table)] = list(prompt_ids)
        return self.logits(prompt_ids)[-1]

    @torch.no_grad()
    def prefill_chunk(
        self,
        prompt_ids: list[int],
        cache: PagedKVCache,
        table: BlockTable,
        *,
        start_pos: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """Correctness-first chunked prefill over remembered token history."""
        del cache
        self._validate_token_ids(prompt_ids)
        if not 0 <= start_pos < len(prompt_ids):
            raise ValueError(f"start_pos must be in [0, {len(prompt_ids)}); got {start_pos}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1; got {chunk_size}")
        if table.length != start_pos:
            raise ValueError(
                f"chunk start {start_pos} must equal cached prompt length {table.length}"
            )
        history = self._history_for(table)
        if len(history) != start_pos:
            raise PretrainBundleError(
                f"cached token history length {len(history)} does not match "
                f"table length {start_pos}"
            )
        end_pos = min(len(prompt_ids), start_pos + chunk_size)
        chunk = list(prompt_ids[start_pos:end_pos])
        table.reserve(len(chunk))
        history.extend(chunk)
        table.length = end_pos
        return self.logits(history)[-1]

    @torch.no_grad()
    def decode_one(
        self, cache: PagedKVCache, table: BlockTable, token_id: int | torch.Tensor
    ) -> torch.Tensor:
        """Correctness-first one-token decode by full recompute over remembered ids."""
        del cache
        token = self._scalar_token(token_id)
        self._validate_token_ids([token])
        history = self._history_for(table)
        table.reserve(1)
        history.append(token)
        table.length = len(history)
        return self.logits(history)[-1]

    @torch.no_grad()
    def decode_many(
        self,
        cache: PagedKVCache,
        tables: list[BlockTable],
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Correctness-first batched decode; loops per request and stacks logits rows."""
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).reshape(-1)
        if len(tables) != int(ids.numel()):
            raise ValueError(f"tables/token_ids length mismatch: {len(tables)} vs {ids.numel()}")
        rows = [
            self.decode_one(cache, table, token) for table, token in zip(tables, ids, strict=True)
        ]
        return torch.stack(rows)

    @torch.no_grad()
    def decode_tokens(
        self,
        cache: PagedKVCache,
        table: BlockTable,
        token_ids: list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Correctness-first speculative verification path."""
        ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device).reshape(-1)
        if ids.numel() < 1:
            raise ValueError("decode_tokens needs at least one token")
        rows = [self.decode_one(cache, table, token) for token in ids]
        return torch.stack(rows)

    def _decoder_layer(
        self, hidden: torch.Tensor, layer: int, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        prefix = f"layers.{layer}."
        residual = hidden
        x = _rms_norm(hidden, self.w[prefix + "input_norm.weight"], self.rms_eps)
        hidden = residual + self._attention(x, prefix, cos, sin)

        residual = hidden
        x = _rms_norm(hidden, self.w[prefix + "post_attention_norm.weight"], self.rms_eps)
        return residual + self._mlp(x, prefix)

    def _attention(
        self, x: torch.Tensor, prefix: str, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q, k, v = self._project_heads(x, prefix)
        if self.qk_norm:
            q = _rms_norm(q, self.w[prefix + "attn.q_norm.weight"], self.rms_eps)
            k = _rms_norm(k, self.w[prefix + "attn.k_norm.weight"], self.rms_eps)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        k, v = self._expand_kv(k, v)
        attn = self.backend.forward(q, k, v)
        return self._output_proj(attn, prefix)

    def _project_heads(
        self, x: torch.Tensor, prefix: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = x.shape[0]
        q = self._linear(x, prefix + "attn.q_proj")
        k = self._linear(x, prefix + "attn.k_proj")
        v = self._linear(x, prefix + "attn.v_proj")
        q = q.view(seq_len, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, self.num_kv_heads, self.head_dim).transpose(0, 1)
        return q, k, v

    def _expand_kv(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        repeat = self.num_heads // self.num_kv_heads
        return k.repeat_interleave(repeat, dim=0), v.repeat_interleave(repeat, dim=0)

    def _output_proj(self, attn: torch.Tensor, prefix: str) -> torch.Tensor:
        seq_len = attn.shape[1]
        merged = attn.transpose(0, 1).reshape(seq_len, self.num_heads * self.head_dim)
        return self._linear(merged, prefix + "attn.o_proj")

    def _mlp(self, x: torch.Tensor, prefix: str) -> torch.Tensor:
        gate = self._linear(x, prefix + "mlp.gate_proj")
        up = self._linear(x, prefix + "mlp.up_proj")
        return self._linear(torch.nn.functional.silu(gate) * up, prefix + "mlp.down_proj")

    def _linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        out = x @ self.w[name + ".weight"].to(self.dtype).T
        bias = self.w.get(name + ".bias")
        if bias is not None:
            out = out + bias.to(self.dtype)
        return out

    def _lm_head(self) -> torch.Tensor:
        if self.tie_word_embeddings:
            return self.w["embed_tokens.weight"].to(self.dtype)
        return self.w["lm_head.weight"].to(self.dtype)

    def _rope_tables(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, dtype=torch.float32, device=self.device)
        half = self.head_dim // 2
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, half, dtype=torch.float32, device=self.device) / half)
        )
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def _validate_token_ids(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("token_ids must be non-empty")
        bad = [
            token_id
            for token_id in token_ids
            if isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < self.config.vocab_size
        ]
        if bad:
            raise ValueError(
                f"token_ids must be ints in [0, {self.config.vocab_size}); got {bad[:3]}"
            )

    def _history_for(self, table: BlockTable) -> list[int]:
        history = self._history_by_table.get(id(table))
        if history is not None:
            return history
        if table.length == 0:
            history = []
            self._history_by_table[id(table)] = history
            return history
        raise PretrainBundleError(
            "dense pretrain backend cannot reconstruct token history for this block table; "
            "disable prefix caching for dense bundles until a real KV-cache path exists"
        )

    def _scalar_token(self, token_id: int | torch.Tensor) -> int:
        if isinstance(token_id, bool):
            raise ValueError("token_id must be an integer token id, not bool")
        if isinstance(token_id, int):
            return token_id
        tensor = torch.as_tensor(token_id, device=self.device)
        if tensor.numel() != 1:
            raise ValueError(f"token_id must be scalar; got shape {tuple(tensor.shape)}")
        return int(tensor.item())


def _required_file(root: Path, name: str) -> Path:
    path = root / name
    if not path.is_file():
        raise PretrainBundleError(f"bundle is missing required file {name}")
    return path


def _read_json_object(path: Path) -> Mapping[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PretrainBundleError(f"{path.name} is not valid JSON: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise PretrainBundleError(f"{path.name} must contain a JSON object")
    return raw


def _require_manifest_format(manifest: Mapping[str, object], path: Path) -> None:
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


def _resolve_tokenizer_path(root: Path, manifest: Mapping[str, object]) -> Path:
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


def _read_weights(
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


def _require_weight_key_format(metadata: Mapping[str, object], path: Path) -> None:
    fields = ("key_format", "dense_key_format", "state_dict_format", "weights_format", "format")
    candidates = [metadata.get(field) for field in fields]
    if BUNDLE_FORMAT not in {value for value in candidates if isinstance(value, str)}:
        found = (
            ", ".join(sorted({value for value in candidates if isinstance(value, str)})) or "none"
        )
        raise PretrainBundleError(
            f"{path.name} metadata must identify dense key format {BUNDLE_FORMAT!r}; found {found}"
        )


def _normalize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    config: PretrainDenseConfig,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    head_dim = config.head_dim

    _copy_weight(
        weights,
        "embed_tokens.weight",
        state_dict,
        _root_aliases("token_embedding.weight", "embed_tokens.weight", "tok_embeddings.weight"),
        (config.vocab_size, config.hidden_size),
        dtype=dtype,
        device=device,
    )
    _copy_weight(
        weights,
        "norm.weight",
        state_dict,
        _root_aliases("final_norm.weight", "norm.weight", "ln_f.weight"),
        (config.hidden_size,),
        dtype=dtype,
        device=device,
    )
    if not config.tie_word_embeddings:
        _copy_weight(
            weights,
            "lm_head.weight",
            state_dict,
            _root_aliases("lm_head.weight", "output.weight", "output_projection.weight"),
            (config.vocab_size, config.hidden_size),
            dtype=dtype,
            device=device,
        )

    for layer in range(config.num_hidden_layers):
        _copy_layer_weight(
            weights,
            layer,
            "input_norm.weight",
            state_dict,
            ("attention_norm.weight", "attn_norm.weight", "input_layernorm.weight", "norm1.weight"),
            (config.hidden_size,),
            dtype=dtype,
            device=device,
        )
        _copy_layer_weight(
            weights,
            layer,
            "post_attention_norm.weight",
            state_dict,
            (
                "feedforward_norm.weight",
                "ffn_norm.weight",
                "mlp_norm.weight",
                "post_attention_layernorm.weight",
                "norm2.weight",
            ),
            (config.hidden_size,),
            dtype=dtype,
            device=device,
        )
        _copy_projection(
            weights,
            layer,
            "attn.q_proj",
            state_dict,
            ("attention.wq", "attention.q_proj", "attn.q_proj", "self_attn.q_proj", "attn.wq"),
            (config.num_attention_heads * head_dim, config.hidden_size),
            dtype=dtype,
            device=device,
        )
        _copy_projection(
            weights,
            layer,
            "attn.k_proj",
            state_dict,
            ("attention.wk", "attention.k_proj", "attn.k_proj", "self_attn.k_proj", "attn.wk"),
            (config.num_key_value_heads * head_dim, config.hidden_size),
            dtype=dtype,
            device=device,
        )
        _copy_projection(
            weights,
            layer,
            "attn.v_proj",
            state_dict,
            ("attention.wv", "attention.v_proj", "attn.v_proj", "self_attn.v_proj", "attn.wv"),
            (config.num_key_value_heads * head_dim, config.hidden_size),
            dtype=dtype,
            device=device,
        )
        _copy_projection(
            weights,
            layer,
            "attn.o_proj",
            state_dict,
            ("attention.wo", "attention.o_proj", "attn.o_proj", "self_attn.o_proj", "attn.wo"),
            (config.hidden_size, config.hidden_size),
            dtype=dtype,
            device=device,
        )
        if config.qk_norm:
            _copy_layer_weight(
                weights,
                layer,
                "attn.q_norm.weight",
                state_dict,
                ("attention.q_norm.weight", "attn.q_norm.weight", "self_attn.q_norm.weight"),
                (head_dim,),
                dtype=dtype,
                device=device,
            )
            _copy_layer_weight(
                weights,
                layer,
                "attn.k_norm.weight",
                state_dict,
                ("attention.k_norm.weight", "attn.k_norm.weight", "self_attn.k_norm.weight"),
                (head_dim,),
                dtype=dtype,
                device=device,
            )
        _copy_projection(
            weights,
            layer,
            "mlp.gate_proj",
            state_dict,
            ("feedforward.w_gate", "feed_forward.w1", "ffn.w1", "mlp.gate_proj", "mlp.w1"),
            (config.intermediate_size, config.hidden_size),
            dtype=dtype,
            device=device,
        )
        _copy_projection(
            weights,
            layer,
            "mlp.up_proj",
            state_dict,
            ("feedforward.w_up", "feed_forward.w3", "ffn.w3", "mlp.up_proj", "mlp.w3"),
            (config.intermediate_size, config.hidden_size),
            dtype=dtype,
            device=device,
        )
        _copy_projection(
            weights,
            layer,
            "mlp.down_proj",
            state_dict,
            ("feedforward.w_down", "feed_forward.w2", "ffn.w2", "mlp.down_proj", "mlp.w2"),
            (config.hidden_size, config.intermediate_size),
            dtype=dtype,
            device=device,
        )

    return weights


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
    _copy_optional_weight(
        weights,
        f"layers.{layer}.{target}.bias",
        state_dict,
        _layer_aliases(layer, tuple(base + ".bias" for base in bases)),
        (shape[0],),
        dtype=dtype,
        device=device,
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
) -> None:
    for alias in aliases:
        tensor = state_dict.get(alias)
        if tensor is not None:
            weights[target] = _validated_tensor(alias, tensor, shape, dtype=dtype, device=device)
            return
    raise PretrainBundleError(f"weights.pt is missing required tensor for {target}")


def _copy_optional_weight(
    weights: dict[str, torch.Tensor],
    target: str,
    state_dict: Mapping[str, torch.Tensor],
    aliases: Iterable[str],
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> None:
    for alias in aliases:
        tensor = state_dict.get(alias)
        if tensor is not None:
            weights[target] = _validated_tensor(alias, tensor, shape, dtype=dtype, device=device)
            return


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


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    return (weight.to(torch.float32) * xf).to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.to(x.dtype).unsqueeze(0)
    sin = sin.to(x.dtype).unsqueeze(0)
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin
