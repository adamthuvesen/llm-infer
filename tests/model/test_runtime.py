"""Model runtime registry tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from llm_infer.model.decode import greedy_decode
from llm_infer.model.runtime import ModelRegistryError, available_backends, load_model_runtime
from llm_infer.serve import build_app_from_runtime
from tests.correctness.test_pretrain_bundle import _write_tiny_bundle


def test_registry_lists_qwen_and_dense_backends() -> None:
    assert available_backends() == ("dense", "qwen")


def test_unknown_backend_fails_loudly() -> None:
    with pytest.raises(ModelRegistryError, match="unknown model backend"):
        load_model_runtime("not-a-backend")


def test_dense_runtime_loads_bundle_metadata_and_tokenizer(tmp_path: Path) -> None:
    bundle = _write_tiny_bundle(tmp_path)

    runtime = load_model_runtime("dense", bundle_path=bundle)

    assert runtime.backend_id == "dense"
    assert runtime.model_id == "tiny-dense"
    assert runtime.bundle_path == bundle
    assert runtime.metadata["format"] == "llm_pretrain_dense_v1"
    assert runtime.tokenizer.encode("tok_1") == [1]
    assert runtime.tokenizer.decode([1]) == "tok_1"


def test_dense_runtime_honors_bundle_chat_template_and_special_token_policy(
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

    runtime = load_model_runtime("dense", bundle_path=bundle)
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

    assert runtime.model_id == "esme-214m-chat"
    assert runtime.eos_token_ids == frozenset({9})
    assert rendered == "user\nhello\nassistant\n"
    assert tokenized == [2, 5, 3]
    assert runtime.tokenizer.encode("hello") == [5]
    assert runtime.tokenizer.encode("hello", add_special_tokens=True) == [5, 9]


def test_dense_runtime_drives_serving_path(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    runtime = load_model_runtime("dense", bundle_path=_write_tiny_bundle(tmp_path))
    app = build_app_from_runtime(runtime, block_size=8, num_blocks=32)

    with TestClient(app) as client:
        models = client.get("/v1/models")
        assert models.status_code == 200
        assert models.json()["data"][0]["id"] == "tiny-dense"

        response = client.post(
            "/v1/completions",
            json={"model": "tiny-dense", "prompt": "tok_1", "max_tokens": 2},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == "tiny-dense"
        assert body["usage"]["completion_tokens"] == 2


def test_dense_runtime_eos_metadata_drives_serving_finish_reason(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

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

    with TestClient(app) as client:
        response = client.post(
            "/v1/completions",
            json={"model": "tiny-dense", "prompt": "tok_1", "max_tokens": 5},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 1


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
