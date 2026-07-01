"""Model runtime registry for llm-infer backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from llm_infer.model.pretrain_bundle_loader import read_json_object
from llm_infer.model.qwen import QwenModel

RuntimeLoader = Callable[..., ModelRuntime]
BundleBackendId = Literal["dense", "esme"]


class ModelRegistryError(ValueError):
    """Raised when a requested model backend cannot be resolved."""


@dataclass(frozen=True)
class _DenseTokenizerMetadata:
    add_special_tokens: bool
    chat_template: Mapping[str, object] | None


class TokenizersJsonTokenizer:
    """Small adapter around a standalone ``tokenizer.json`` export."""

    def __init__(
        self,
        tokenizer_path: Path,
        *,
        default_add_special_tokens: bool = True,
        chat_template: Mapping[str, object] | None = None,
    ) -> None:
        from tokenizers import Tokenizer

        self.path = tokenizer_path
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._default_add_special_tokens = default_add_special_tokens
        self._chat_template = chat_template

    def encode(self, text: str, *, add_special_tokens: bool | None = None) -> list[int]:
        special_tokens = (
            self._default_add_special_tokens if add_special_tokens is None else add_special_tokens
        )
        return list(self._tokenizer.encode(text, add_special_tokens=special_tokens).ids)

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> object:
        rendered = self._render_chat_template(messages, add_generation_prompt=add_generation_prompt)
        return self.encode(rendered) if tokenize else rendered

    def _render_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool
    ) -> str:
        if self._chat_template is not None:
            return _render_bundle_chat_template(
                self._chat_template, messages, add_generation_prompt=add_generation_prompt
            )
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in messages)
        if add_generation_prompt:
            rendered = f"{rendered}\nassistant:" if rendered else "assistant:"
        return rendered


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


def _load_esme_runtime(
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    bundle_path: Path | str | None,
    model_id: str | None,
    revision: str | None,
    attention_backend: AttentionBackend | None,
) -> ModelRuntime:
    return _load_bundle_runtime(
        "esme",
        dtype=dtype,
        device=device,
        bundle_path=bundle_path,
        model_id=model_id,
        revision=revision,
        attention_backend=attention_backend,
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
    return _load_bundle_runtime(
        "dense",
        dtype=dtype,
        device=device,
        bundle_path=bundle_path,
        model_id=model_id,
        revision=revision,
        attention_backend=attention_backend,
    )


def _load_bundle_runtime(
    backend_id: BundleBackendId,
    *,
    dtype: torch.dtype,
    device: torch.device | str,
    bundle_path: Path | str | None,
    model_id: str | None,
    revision: str | None,
    attention_backend: AttentionBackend | None,
) -> ModelRuntime:
    if model_id is not None:
        raise ModelRegistryError(
            f"backend {backend_id!r} derives model id from the bundle; omit --model"
        )
    if revision is not None:
        raise ModelRegistryError(
            f"backend {backend_id!r} derives revision from the bundle; omit --revision"
        )
    if bundle_path is None:
        raise ModelRegistryError(
            f"backend {backend_id!r} requires --bundle pointing at an export directory"
        )

    root = Path(bundle_path)
    model = PretrainBundleModel.load(
        root,
        dtype=dtype,
        backend=attention_backend,
        device=device,
    )
    manifest = read_json_object(root / "manifest.json")
    tokenizer_metadata = _dense_tokenizer_metadata(manifest)
    tokenizer = TokenizersJsonTokenizer(
        model.tokenizer_path,
        default_add_special_tokens=tokenizer_metadata.add_special_tokens,
        chat_template=tokenizer_metadata.chat_template,
    )
    # Bundle backends expose the full shared-engine capability set; flash_attention reflects
    # the actual backend (torch_naive unless a flash backend is passed in).
    resolved_backend = attention_backend or model.backend
    capabilities = BackendCapabilities(
        paged_kv=DENSE_CAPABILITIES.paged_kv,
        prefix_caching=DENSE_CAPABILITIES.prefix_caching,
        speculative=DENSE_CAPABILITIES.speculative,
        flash_attention=isinstance(resolved_backend, FlashAttnPagedAttention),
    )
    return ModelRuntime(
        backend_id=backend_id,
        model_id=_dense_model_id(manifest, root),
        model=model,
        tokenizer=tokenizer,
        eos_token_ids=_dense_eos_ids(manifest),
        capabilities=capabilities,
        metadata={
            "source": "esme-export",
            "format": "llm_pretrain_dense_v1",
            "manifest": manifest,
            "chat_template": tokenizer_metadata.chat_template,
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


def _dense_tokenizer_metadata(manifest: Mapping[str, object]) -> _DenseTokenizerMetadata:
    tokenizer_entry = manifest.get("tokenizer")
    tokenizer_config = tokenizer_entry if isinstance(tokenizer_entry, dict) else {}
    chat_template = _dense_chat_template(manifest, tokenizer_config)
    add_special_tokens = _dense_add_special_tokens(manifest, tokenizer_config, chat_template)
    return _DenseTokenizerMetadata(
        add_special_tokens=add_special_tokens,
        chat_template=chat_template,
    )


def _dense_chat_template(
    manifest: Mapping[str, object], tokenizer_config: Mapping[str, object]
) -> Mapping[str, object] | None:
    for value in (manifest.get("chat_template"), tokenizer_config.get("chat_template")):
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ModelRegistryError("bundle chat_template must be an object")
        return value
    return None


def _dense_add_special_tokens(
    manifest: Mapping[str, object],
    tokenizer_config: Mapping[str, object],
    chat_template: Mapping[str, object] | None,
) -> bool:
    for value in (
        tokenizer_config.get("add_special_tokens"),
        manifest.get("add_special_tokens"),
        chat_template.get("add_special_tokens") if chat_template is not None else None,
    ):
        if value is None:
            continue
        if not isinstance(value, bool):
            raise ModelRegistryError("bundle add_special_tokens must be a boolean")
        return value
    return True


def _render_bundle_chat_template(
    template: Mapping[str, object],
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    roles = template.get("roles")
    if not isinstance(roles, dict):
        raise ModelRegistryError("bundle chat_template.roles must be an object")
    parts: list[str] = []
    for message in messages:
        role = message["role"]
        pattern = roles.get(role)
        if not isinstance(pattern, str):
            raise ModelRegistryError(f"bundle chat_template is missing role {role!r}")
        parts.append(pattern.format(content=message["content"]))
    if add_generation_prompt:
        generation_prompt = template.get("generation_prompt")
        if not isinstance(generation_prompt, str) or not generation_prompt:
            raise ModelRegistryError("bundle chat_template.generation_prompt must be set")
        parts.append(generation_prompt)
    return "".join(parts)


_LOADERS: dict[str, RuntimeLoader] = {
    "dense": _load_dense_runtime,
    "esme": _load_esme_runtime,
    "qwen": _load_qwen_runtime,
}
