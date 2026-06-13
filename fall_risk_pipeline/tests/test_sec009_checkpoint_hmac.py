"""SEC-009: production checkpoint HMAC policy and Hub revision pinning."""

from __future__ import annotations

import pytest

from src.utils.checkpoint_io import (
    CheckpointIntegrityError,
    assert_production_checkpoint_policy,
    assert_production_hub_revision_policy,
    is_floating_hub_revision,
    load_checkpoint,
    save_checkpoint,
)


@pytest.fixture
def simple_model():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression().fit([[0.0], [1.0], [2.0]], [0, 0, 1])


def test_production_requires_hmac_key(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SEC-009"):
        assert_production_checkpoint_policy()


def test_production_load_requires_manifest_hmac(tmp_path, simple_model, monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CHECKPOINT_HMAC_KEY", "deploy-secret")
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    path = tmp_path / "model.pkl"
    # Manifest without HMAC (SHA-256 only)
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    save_checkpoint(path, simple_model, manifest_dir=tmp_path)

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CHECKPOINT_HMAC_KEY", "deploy-secret")
    with pytest.raises(CheckpointIntegrityError, match="no HMAC signature"):
        load_checkpoint(path, manifest_dir=tmp_path, require_manifest=True)


def test_production_load_accepts_hmac_signed_checkpoint(tmp_path, simple_model, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_HMAC_KEY", "deploy-secret")
    path = tmp_path / "model.pkl"
    save_checkpoint(path, simple_model, manifest_dir=tmp_path)

    monkeypatch.setenv("ENVIRONMENT", "production")
    loaded = load_checkpoint(path, manifest_dir=tmp_path, require_manifest=True)
    assert hasattr(loaded, "predict")


def test_floating_hub_revision_detection():
    assert is_floating_hub_revision("main")
    assert is_floating_hub_revision("HEAD")
    assert not is_floating_hub_revision("v1.0.0")


def test_production_rejects_floating_hub_revision(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(RuntimeError, match="SEC-009"):
        assert_production_hub_revision_policy("main")
    assert_production_hub_revision_policy("v1.2.3")
