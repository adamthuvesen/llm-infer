from __future__ import annotations

from pathlib import Path

from scripts.modal_esme_bundle import ESME_BUNDLE_ENV, SIBLING_LOCAL_BUNDLE, local_bundle_path


def test_local_bundle_path_explicit_argument_wins(monkeypatch, tmp_path: Path) -> None:
    explicit_path = tmp_path / "explicit-esme-bundle"
    env_path = tmp_path / "env-esme-bundle"

    monkeypatch.setenv(ESME_BUNDLE_ENV, str(env_path))

    assert local_bundle_path(str(explicit_path)) == explicit_path


def test_local_bundle_path_uses_env_without_explicit_arg(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / "env-esme-bundle"

    monkeypatch.setenv(ESME_BUNDLE_ENV, str(env_path))

    assert local_bundle_path("") == env_path


def test_local_bundle_path_uses_sibling_fallback_without_arg_or_env(monkeypatch) -> None:
    monkeypatch.delenv(ESME_BUNDLE_ENV, raising=False)

    assert local_bundle_path("") == SIBLING_LOCAL_BUNDLE
