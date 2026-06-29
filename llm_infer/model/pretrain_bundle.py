"""Loader and correctness-only forward pass for llm-pretrain dense export bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

from llm_infer.kernels.base import AttentionBackend
from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.kv_cache.block_table import BlockTable
from llm_infer.kv_cache.paged_kv_cache import PagedKVCache
from llm_infer.model.pretrain_bundle_loader import (
    BUNDLE_FORMAT,
    PretrainBundleError,
    PretrainDenseConfig,
    _normalize_state_dict,
    _read_json_object,
    _read_weights,
    _require_manifest_format,
    _require_weight_key_format,
    _required_file,
    _resolve_tokenizer_path,
)
from llm_infer.model.rope_utils import apply_rope, rms_norm
from llm_infer.profiling import TimingProfiler

__all__ = ["BUNDLE_FORMAT", "PretrainBundleError", "PretrainBundleModel"]


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

        hidden = rms_norm(hidden, self.w["norm.weight"], self.rms_eps)
        logits = hidden @ self._lm_head().T
        return self._apply_logit_soft_cap(logits)

    def release_table(self, table: BlockTable) -> None:
        """Drop remembered token history when a block table is freed."""
        self._history_by_table.pop(id(table), None)

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
        x = rms_norm(hidden, self.w[prefix + "input_norm.weight"], self.rms_eps)
        hidden = residual + self._attention(x, prefix, cos, sin)

        residual = hidden
        x = rms_norm(hidden, self.w[prefix + "post_attention_norm.weight"], self.rms_eps)
        return residual + self._mlp(x, prefix)

    def _attention(
        self, x: torch.Tensor, prefix: str, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        q, k, v = self._project_heads(x, prefix)
        if self.qk_norm:
            q = rms_norm(q, self.w[prefix + "attn.q_norm.weight"], self.rms_eps)
            k = rms_norm(k, self.w[prefix + "attn.k_norm.weight"], self.rms_eps)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
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

    def _apply_logit_soft_cap(self, logits: torch.Tensor) -> torch.Tensor:
        cap = self.config.logit_soft_cap
        if cap is None or cap <= 0.0:
            return logits
        return cap * torch.tanh(logits / cap)
