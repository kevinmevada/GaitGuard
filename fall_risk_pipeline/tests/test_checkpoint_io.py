"""Tests for signed checkpoint manifest I/O."""

from __future__ import annotations

import os

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from src.utils.checkpoint_io import (
    CheckpointIntegrityError,
    load_checkpoint,
    manifest_path,
    refresh_manifest,
    save_checkpoint,
)


@pytest.fixture
def simple_model():
    return LogisticRegression().fit([[0.0], [1.0], [2.0]], [0, 0, 1])


def test_save_and_load_checkpoint_with_manifest(tmp_path, simple_model, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    path = tmp_path / "model.pkl"
    save_checkpoint(path, simple_model, manifest_dir=tmp_path)
    assert manifest_path(tmp_path).exists()
    loaded = load_checkpoint(path, manifest_dir=tmp_path, require_manifest=True)
    assert hasattr(loaded, "predict")


def test_load_requires_manifest_entry(tmp_path, simple_model, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    path = tmp_path / "rogue.pkl"
    joblib.dump(simple_model, path)
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(path, manifest_dir=tmp_path, require_manifest=True)


def test_tampered_checkpoint_rejected(tmp_path, simple_model, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    path = tmp_path / "model.pkl"
    save_checkpoint(path, simple_model, manifest_dir=tmp_path)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(path, manifest_dir=tmp_path, require_manifest=True)


def test_hmac_verification(tmp_path, simple_model, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_HMAC_KEY", "test-secret-key")
    path = tmp_path / "model.pkl"
    save_checkpoint(path, simple_model, manifest_dir=tmp_path)
    loaded = load_checkpoint(path, manifest_dir=tmp_path, require_manifest=True)
    assert hasattr(loaded, "predict")

    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    with pytest.raises(CheckpointIntegrityError):
        load_checkpoint(path, manifest_dir=tmp_path, require_manifest=True)


def test_refresh_manifest_hashes_existing_pkls(tmp_path, simple_model, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    joblib.dump(simple_model, tmp_path / "legacy.pkl")
    refresh_manifest(tmp_path)
    loaded = load_checkpoint(
        tmp_path / "legacy.pkl",
        manifest_dir=tmp_path,
        require_manifest=True,
    )
    assert hasattr(loaded, "predict")
