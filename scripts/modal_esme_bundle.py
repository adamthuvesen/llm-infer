"""Shared Esme Modal bundle path, validation, and staging helpers.

Bundle lookup order is explicit: ``--bundle-path``, then ``ESME_BUNDLE_PATH``,
then the standard sibling checkout export at
``../esme-posttrain/exports/esme-214m-chat``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

VOLUME_NAME = "llm-infer-esme-bundles"
ESME_BUNDLE_DIR = "esme-214m-chat"
ESME_BUNDLE_MOUNT = "/esme-bundles"
REMOTE_BUNDLE_PATH = f"{ESME_BUNDLE_MOUNT}/{ESME_BUNDLE_DIR}"
REQUIRED_BUNDLE_FILES = ("manifest.json", "config.json", "tokenizer.json", "weights.pt")
ESME_BUNDLE_ENV = "ESME_BUNDLE_PATH"

REPO_ROOT = Path(__file__).resolve().parent.parent
SIBLING_LOCAL_BUNDLE = REPO_ROOT.parent / "esme-posttrain" / "exports" / ESME_BUNDLE_DIR


def local_bundle_path(bundle_path: str) -> Path:
    if bundle_path:
        return Path(bundle_path).expanduser()
    env_path = os.environ.get(ESME_BUNDLE_ENV)
    return Path(env_path).expanduser() if env_path else SIBLING_LOCAL_BUNDLE


def validate_local_bundle(bundle_path: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_BUNDLE_FILES if not (bundle_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{bundle_path} is missing required bundle files: {missing}")
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    model = manifest.get("model")
    if not isinstance(model, dict):
        raise ValueError(f"{bundle_path}/manifest.json must contain a model object")
    if model.get("name") != "Esme-214M-Chat":
        raise ValueError(f"expected Esme-214M-Chat bundle, found model.name={model.get('name')!r}")
    if model.get("id") != "esme-214m-chat":
        raise ValueError(f"expected esme-214m-chat bundle id, found model.id={model.get('id')!r}")
    if manifest.get("eos_token_ids") != [2]:
        raise ValueError(f"expected Esme EOS [2], found {manifest.get('eos_token_ids')!r}")
    return manifest


def stage_bundle(volume: modal.Volume, bundle_path: Path, *, label: str) -> dict[str, object]:
    manifest = validate_local_bundle(bundle_path)
    print(f"[{label}] staging {bundle_path} -> {VOLUME_NAME}:/{ESME_BUNDLE_DIR}")
    with volume.batch_upload(force=True) as batch:
        for name in REQUIRED_BUNDLE_FILES:
            batch.put_file(bundle_path / name, f"/{ESME_BUNDLE_DIR}/{name}")
    print(f"[{label}] staged {len(REQUIRED_BUNDLE_FILES)} bundle files")
    return manifest
