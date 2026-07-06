"""Model runtime registry tests."""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from llm_infer.fixtures.tiny_pretrain_bundle import write_tiny_pretrain_bundle as _write_tiny_bundle
from llm_infer.kernels.torch_naive import TorchNaiveAttention
from llm_infer.model import runtime as runtime_module
from llm_infer.model.decode import greedy_decode
from llm_infer.model.runtime import (
    ATTENTION_BACKEND_CHOICES,
    ModelRegistryError,
    available_backends,
    load_model_runtime,
)
from llm_infer.serve import DEFAULT_BACKEND, _bundle_path_for_backend, build_app_from_runtime


def _client(app) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def _run(coro):
    return asyncio.run(coro)


def test_registry_lists_qwen_esme_and_dense_alias() -> None:
    assert available_backends() == ("dense", "esme", "qwen")


def test_unknown_backend_fails_loudly() -> None:
    with pytest.raises(ModelRegistryError, match="unknown model backend"):
        load_model_runtime("not-a-backend")


def test_attention_backend_choices_are_stable() -> None:
    assert ATTENTION_BACKEND_CHOICES == ("auto", "torch_naive", "flash_attn", "flashinfer")


def test_auto_bundle_cpu_stays_on_reference_attention(tmp_path: Path) -> None:
    runtime = load_model_runtime(
        "esme",
        bundle_path=_write_tiny_bundle(tmp_path),
        dtype=torch.bfloat16,
        device="cpu",
    )

    assert type(runtime.model.backend).__name__ == "TorchNaiveAttention"
    assert runtime.metadata["attention_backend"] == "TorchNaiveAttention"
    assert runtime.metadata["attention_backend_choice"] == "auto"


def test_auto_bundle_fp32_cuda_stays_on_reference_attention() -> None:
    resolved = runtime_module._resolve_attention_backend(
        "esme",
        dtype=torch.float32,
        device="cuda",
        attention_backend=None,
        attention_backend_name="auto",
    )

    assert resolved is None


def test_auto_bundle_cuda_low_precision_selects_flashinfer(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFlashInfer:
        pass

    monkeypatch.setattr(runtime_module, "FlashInferPagedAttention", FakeFlashInfer)

    resolved = runtime_module._resolve_attention_backend(
        "esme",
        dtype=torch.bfloat16,
        device="cuda",
        attention_backend=None,
        attention_backend_name="auto",
    )

    assert isinstance(resolved, FakeFlashInfer)


def test_explicit_torch_naive_backend_loads_bundle(tmp_path: Path) -> None:
    runtime = load_model_runtime(
        "esme",
        bundle_path=_write_tiny_bundle(tmp_path),
        attention_backend_name="torch_naive",
    )

    assert type(runtime.model.backend).__name__ == "TorchNaiveAttention"
    assert runtime.metadata["attention_backend_choice"] == "torch_naive"


def test_named_backend_conflicts_with_direct_backend() -> None:
    with pytest.raises(
        ModelRegistryError,
        match="either attention_backend or attention_backend_name",
    ):
        load_model_runtime(
            "esme",
            attention_backend=TorchNaiveAttention(),
            attention_backend_name="torch_naive",
        )


def test_flash_attn_rejects_cpu_and_fp32() -> None:
    with pytest.raises(ModelRegistryError, match="requires a CUDA device"):
        runtime_module._resolve_attention_backend(
            "esme",
            dtype=torch.bfloat16,
            device="cpu",
            attention_backend=None,
            attention_backend_name="flash_attn",
        )
    with pytest.raises(ModelRegistryError, match="requires float16 or bfloat16"):
        runtime_module._resolve_attention_backend(
            "esme",
            dtype=torch.float32,
            device="cuda",
            attention_backend=None,
            attention_backend_name="flash_attn",
        )


def test_flashinfer_rejects_qwen() -> None:
    with pytest.raises(ModelRegistryError, match="only supported for bundles"):
        runtime_module._resolve_attention_backend(
            "qwen",
            dtype=torch.bfloat16,
            device="cuda",
            attention_backend=None,
            attention_backend_name="flashinfer",
        )


def test_serve_defaults_to_esme_backend() -> None:
    assert DEFAULT_BACKEND == "esme"


def test_serve_bundle_backend_uses_env_bundle() -> None:
    bundle = _bundle_path_for_backend(
        "esme",
        explicit_bundle=None,
        env_bundle="/tmp/esme-214m-chat",
    )

    assert bundle == Path("/tmp/esme-214m-chat")


def test_serve_bundle_backend_requires_bundle() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="requires --bundle"):
        _bundle_path_for_backend("esme", explicit_bundle=None, env_bundle=None)


def test_serve_qwen_reproduction_ignores_env_bundle() -> None:
    bundle = _bundle_path_for_backend(
        "qwen",
        explicit_bundle=None,
        env_bundle="/tmp/esme-214m-chat",
    )

    assert bundle is None


def test_serve_qwen_rejects_explicit_bundle() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="does not accept --bundle"):
        _bundle_path_for_backend(
            "qwen",
            explicit_bundle=Path("/tmp/esme-214m-chat"),
            env_bundle=None,
        )


def test_dense_runtime_loads_bundle_metadata_and_tokenizer(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)

    runtime = load_model_runtime("dense", bundle_path=bundle)

    assert runtime.backend_id == "dense"
    assert runtime.model_id == "tiny-dense"
    assert runtime.bundle_path == bundle
    assert runtime.metadata["format"] == "llm_pretrain_dense_v1"
    assert runtime.tokenizer.encode("tok_1") == [1]
    assert runtime.tokenizer.decode([1]) == "tok_1"


def test_esme_runtime_loads_bundle_metadata_and_tokenizer(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)

    runtime = load_model_runtime("esme", bundle_path=bundle)

    assert runtime.backend_id == "esme"
    assert runtime.model_id == "tiny-dense"
    assert runtime.bundle_path == bundle
    assert runtime.metadata["format"] == "llm_pretrain_dense_v1"
    assert runtime.tokenizer.encode("tok_1") == [1]
    assert runtime.tokenizer.decode([1]) == "tok_1"


def test_esme_runtime_honors_bundle_chat_template_and_special_token_policy(
    tmp_path: Path,
) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    _write_chat_tokenizer(bundle / "tokenizer.json")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["model"] = {"id": "esme-214m-chat"}
    manifest["eos_token_ids"] = [9]
    manifest["tokenizer"] = {
        "path": "tokenizer.json",
        "format": "tokenizers-json",
        "add_special_tokens": False,
        "chat_template": {
            "id": "esme_newline_v1",
            "roles": {
                "system": "system\n{content}\n",
                "user": "user\n{content}\n",
                "assistant": "assistant\n{content}\n",
            },
            "generation_prompt": "assistant\n",
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    runtime = load_model_runtime("esme", bundle_path=bundle)
    rendered = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        tokenize=False,
    )
    tokenized = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        add_generation_prompt=True,
        tokenize=True,
    )

    assert runtime.backend_id == "esme"
    assert runtime.model_id == "esme-214m-chat"
    assert runtime.eos_token_ids == frozenset({9})
    assert rendered == "user\nhello\nassistant\n"
    assert tokenized == [2, 5, 3]
    assert runtime.tokenizer.encode("hello") == [5]
    assert runtime.tokenizer.encode("hello", add_special_tokens=True) == [5, 9]


def test_dense_runtime_drives_serving_path(tmp_path: Path) -> None:
    runtime = load_model_runtime("dense", bundle_path=_write_tiny_bundle(tmp_path))
    app = build_app_from_runtime(runtime, block_size=8, num_blocks=32)

    async def go() -> None:
        async with app.router.lifespan_context(app), _client(app) as client:
            models = await client.get("/v1/models")
            assert models.status_code == 200
            assert models.json()["data"][0]["id"] == "tiny-dense"

            response = await client.post(
                "/v1/completions",
                json={"model": "tiny-dense", "prompt": "tok_1", "max_tokens": 2},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "tiny-dense"
        assert body["usage"]["completion_tokens"] == 2

    _run(go())


def test_dense_runtime_eos_metadata_drives_serving_finish_reason(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)
    probe_runtime = load_model_runtime("dense", bundle_path=bundle)
    prompt_ids = probe_runtime.tokenizer.encode("tok_1")
    first_token = greedy_decode(
        probe_runtime.model, prompt_ids, max_new_tokens=1, eos_token_ids=set()
    )[0]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    manifest["eos_token_ids"] = [first_token]
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    runtime = load_model_runtime("dense", bundle_path=bundle)
    app = build_app_from_runtime(runtime, block_size=8, num_blocks=32)

    async def go() -> None:
        async with app.router.lifespan_context(app), _client(app) as client:
            response = await client.post(
                "/v1/completions",
                json={"model": "tiny-dense", "prompt": "tok_1", "max_tokens": 5},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["completion_tokens"] == 1

    _run(go())


def test_esme_chat_bundle_reference_checked_when_configured() -> None:
    bundle = os.environ.get("ESME_BUNDLE_PATH")
    if bundle is None:
        pytest.skip("set ESME_BUNDLE_PATH to run the real Esme-214M-Chat bundle check")

    from llm_infer.model.decode import greedy_decode
    from llm_infer.serving import InferenceEngine, Request

    runtime = load_model_runtime("esme", bundle_path=Path(bundle), dtype=torch.float32)
    rendered = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hi."}],
        add_generation_prompt=True,
        tokenize=False,
    )
    tokenized = runtime.tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hi."}],
        add_generation_prompt=True,
        tokenize=True,
    )
    assert isinstance(rendered, str)
    assert isinstance(tokenized, list)
    prompt_ids = [int(token_id) for token_id in tokenized]
    reference = greedy_decode(
        runtime.model,
        prompt_ids,
        max_new_tokens=2,
        eos_token_ids=set(runtime.eos_token_ids),
    )

    engine = InferenceEngine(
        runtime.model,
        block_size=64,
        num_blocks=8,
        capabilities=runtime.capabilities,
    )
    engine.add_request(
        Request(
            "esme",
            prompt_ids,
            max_new_tokens=2,
            eos_token_ids=runtime.eos_token_ids,
        )
    )

    assert runtime.backend_id == "esme"
    assert runtime.model_id == "esme-214m-chat"
    assert runtime.eos_token_ids == frozenset({2})
    assert runtime.metadata["format"] == "llm_pretrain_dense_v1"
    assert rendered == "user\nSay hi.\nassistant\n"
    assert engine.run()["esme"] == reference


def _write_chat_tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "tok_0": 0,
                "tok_1": 1,
                "user": 2,
                "assistant": 3,
                "system": 4,
                "hello": 5,
                "tok_6": 6,
                "tok_7": 7,
                "tok_8": 8,
                "<eos>": 9,
                "tok_10": 10,
            },
            unk_token="tok_0",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="$A <eos>",
        special_tokens=[("<eos>", 9)],
    )
    tokenizer.save(str(path))


def test_generic_modules_do_not_import_qwen_directly() -> None:
    root = Path(__file__).resolve().parents[2]
    generic_paths = [
        root / "llm_infer" / "model" / "decode.py",
        root / "llm_infer" / "serving" / "engine.py",
        root / "llm_infer" / "benchmarks" / "runners.py",
        root / "llm_infer" / "serve.py",
    ]
    for path in generic_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ]
        imports.extend(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        assert "llm_infer.model.qwen" not in imports, path
