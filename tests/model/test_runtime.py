"""Model runtime registry tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
