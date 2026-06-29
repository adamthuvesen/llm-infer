"""Model runtime registry for llm-infer backends."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import torch

from llm_infer.kernels.base import AttentionBackend
from llm_infer.kernels.flash_attn_paged import FlashAttnPagedAttention
from llm_infer.model.config import MODEL_ID, MODEL_REVISION
from llm_infer.model.interface import (
    DENSE_CAPABILITIES,
    BackendCapabilities,
    ModelRuntime,
)
from llm_infer.model.pretrain_bundle import PretrainBundleModel
from llm_infer.model.qwen import QwenModel

RuntimeLoader = Callable[..., ModelRuntime]


class ModelRegistryError(ValueError):
    """Raised when a requested model backend cannot be resolved."""


class TokenizersJsonTokenizer:
    """Small adapter around a standalone ``tokenizer.json`` export."""

    def __init__(self, tokenizer_path: Path) -> None:
        from tokenizers import Tokenizer

        self.path = tokenizer_path
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text).ids)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> object:
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            rendered = f"{rendered}\nassistant:" if rendered else "assistant:"
        return self.encode(rendered) if tokenize else rendered


def available_backends() -> tuple[str, ...]:
    """Return registered backend ids."""
    return tuple(sorted(_LOADERS))


def load_model_runtime(
    backend: str,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
    bundle_path: Path | str | None = None,
    model_id: str | None = None,
    revision: str | None = None,
    attention_backend: AttentionBackend | None = None,
) -> ModelRuntime:
    """Load one registered model backend with its tokenizer and serving metadata."""
    try:
        loader = _LOADERS[backend]
    except KeyError as exc:
        raise ModelRegistryError(
            f"unknown model backend {backend!r}; "
            f"available backends: {', '.join(available_backends())}"
        ) from exc
    return loader(
        dtype=dtype,
        device=device,
        bundle_path=bundle_path,
        model_id=model_id,
        revision=revision,
        attention_backend=attention_backend,
    )


def _load_qwen_runtime(
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    bundle_path: Path | str | None,
    model_id: str | None,
    revision: str | None,
    attention_backend: AttentionBackend | None,
) -> ModelRuntime:
    if bundle_path is not None:
        raise ModelRegistryError("backend 'qwen' does not accept --bundle")
    resolved_model_id = model_id or MODEL_ID
    resolved_revision = MODEL_REVISION if revision is None and model_id is None else revision

    from transformers import AutoTokenizer, GenerationConfig

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, revision=resolved_revision)
    model = QwenModel.load(
        dtype=dtype,
        backend=attention_backend,
        device=device,
        model_id=resolved_model_id,
        revision=resolved_revision,
    )
    resolved_backend = attention_backend or model.backend
    capabilities = BackendCapabilities(
        paged_kv=True,
        prefix_caching=True,
        speculative=True,
        flash_attention=isinstance(resolved_backend, FlashAttnPagedAttention),
    )
    eos_token_ids = _generation_eos_ids(
        model_id=resolved_model_id,
        revision=resolved_revision,
        tokenizer=tokenizer,
        generation_config_loader=GenerationConfig.from_pretrained,
    )
    return ModelRuntime(
        backend_id="qwen",
        model_id=resolved_model_id,
        model=model,
        tokenizer=tokenizer,
        eos_token_ids=eos_token_ids,
        capabilities=capabilities,
        metadata={
            "revision": resolved_revision,
            "source": "huggingface",
            "architecture": "qwen2",
        },
    )


def _load_dense_runtime(
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    bundle_path: Path | str | None,
    model_id: str | None,
    revision: str | None,
    attention_backend: AttentionBackend | None,
) -> ModelRuntime:
    if model_id is not None:
        raise ModelRegistryError("backend 'dense' derives model id from the bundle; omit --model")
    if revision is not None:
        raise ModelRegistryError(
            "backend 'dense' derives revision from the bundle; omit --revision"
        )
    if bundle_path is None:
        raise ModelRegistryError(
            "backend 'dense' requires --bundle pointing at an export directory"
        )

    root = Path(bundle_path)
    model = PretrainBundleModel.load(
        root,
        dtype=dtype,
        backend=attention_backend,
        device=device,
    )
    manifest = _read_json_object(root / "manifest.json")
    tokenizer = TokenizersJsonTokenizer(model.tokenizer_path)
    return ModelRuntime(
        backend_id="dense",
        model_id=_dense_model_id(manifest, root),
        model=model,
        tokenizer=tokenizer,
        eos_token_ids=_dense_eos_ids(manifest),
        capabilities=DENSE_CAPABILITIES,
        metadata={
            "source": "llm-pretrain-export",
            "format": "llm_pretrain_dense_v1",
            "manifest": manifest,
        },
        bundle_path=root,
    )


def _generation_eos_ids(
    *,
    model_id: str,
    revision: str | None,
    tokenizer: object,
    generation_config_loader: Callable[..., object],
) -> frozenset[int]:
    config = generation_config_loader(model_id, revision=revision)
    ids: set[int] = set()
    cfg_eos = getattr(config, "eos_token_id", None)
    if isinstance(cfg_eos, int):
        ids.add(cfg_eos)
    elif cfg_eos is not None:
        ids.update(int(item) for item in cfg_eos)
    tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
    if tokenizer_eos is not None:
        ids.add(int(tokenizer_eos))
    return frozenset(ids)


def _read_json_object(path: Path) -> Mapping[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ModelRegistryError(f"{path.name} must contain a JSON object")
    return raw


def _dense_model_id(manifest: Mapping[str, object], root: Path) -> str:
    model = manifest.get("model")
    if isinstance(model, dict):
        for key in ("id", "name"):
            value = model.get(key)
            if isinstance(value, str) and value:
                return value
    value = manifest.get("model_id")
    if isinstance(value, str) and value:
        return value
    return root.name


def _dense_eos_ids(manifest: Mapping[str, object]) -> frozenset[int]:
    direct = manifest.get("eos_token_ids")
    if isinstance(direct, list):
        return frozenset(int(item) for item in direct)
    decoding = manifest.get("decoding")
    if isinstance(decoding, dict):
        nested = decoding.get("eos_token_ids")
        if isinstance(nested, list):
            return frozenset(int(item) for item in nested)
    return frozenset()


_LOADERS: dict[str, RuntimeLoader] = {
    "dense": _load_dense_runtime,
    "qwen": _load_qwen_runtime,
}
