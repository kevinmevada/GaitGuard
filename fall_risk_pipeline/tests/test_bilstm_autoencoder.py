"""Unit tests for the BiLSTM autoencoder model (Phase 3 primary model).

Previously untested at the model-unit level (ISSUE-39) — only exercised
indirectly via higher-level ablation/LOSO evaluation tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.models.bilstm_autoencoder import (  # noqa: E402
    BiLSTMAutoencoder,
    SensorChannelSlice,
    load_bilstm_autoencoder,
    train_bilstm_autoencoder,
)


def _sensor_slices() -> list[SensorChannelSlice]:
    # 6 channels total: 3 for "lower_back", 3 for "head"
    return [
        SensorChannelSlice("lower_back", 0, 3),
        SensorChannelSlice("head", 3, 6),
    ]


def test_forward_pass_output_shape_matches_input():
    torch.manual_seed(0)
    n_channels, seq_len = 6, 32
    model = BiLSTMAutoencoder(n_channels, seq_len, hidden_size=16, latent_dim=8)
    x = torch.randn(4, n_channels, seq_len)
    recon, latent = model(x)
    assert recon.shape == x.shape
    assert latent.shape == (4, seq_len, 8)


def test_per_sensor_mse_has_one_key_per_slice_plus_total():
    torch.manual_seed(0)
    n_channels, seq_len = 6, 20
    slices = _sensor_slices()
    model = BiLSTMAutoencoder(n_channels, seq_len, hidden_size=16, latent_dim=8, sensor_slices=slices)
    x = torch.randn(3, n_channels, seq_len)
    recon, _ = model(x)
    mse = model.per_sensor_mse(x, recon)

    expected_keys = {s.name for s in slices} | {"total"}
    assert set(mse.keys()) == expected_keys
    for key, val in mse.items():
        assert val.shape == (3,)
        assert torch.all(val >= 0)


def test_train_then_load_checkpoint_roundtrip(tmp_path):
    """train_bilstm_autoencoder → load_bilstm_autoencoder should give an
    equivalent model: identical reconstruction on a fixed input."""
    rng = np.random.default_rng(0)
    n_channels, seq_len, n_windows = 6, 24, 40
    X = rng.normal(size=(n_windows, n_channels, seq_len)).astype(np.float32)
    slices = _sensor_slices()

    config = {
        "reproducibility": {"seed": 42},
        "primary_model": {
            "bilstm_ae_ensemble": {
                "bilstm_autoencoder": {
                    "hidden_size": 8,
                    "latent_dim": 4,
                    "max_epochs": 2,
                    "batch_size": 8,
                    "early_stopping_patience": 1,
                }
            }
        },
    }

    ckpt_path = tmp_path / "checkpoints" / "bilstm_ae_test.pt"
    model = train_bilstm_autoencoder(
        X, sensor_slices=slices, config=config, checkpoint_path=ckpt_path
    )
    assert ckpt_path.is_file()
    # Manifest should have been registered for the .pt checkpoint (ISSUE-10)
    assert (ckpt_path.parent / "checkpoint_manifest.json").is_file()

    loaded = load_bilstm_autoencoder(ckpt_path)

    fixed_input = torch.tensor(X[:2])
    model.eval()
    with torch.no_grad():
        recon_a, _ = model(fixed_input)
        recon_b, _ = loaded(fixed_input)

    assert torch.allclose(recon_a, recon_b, atol=1e-5)


def test_load_bilstm_autoencoder_require_manifest_missing_entry(tmp_path):
    """ISSUE-10 regression: with require_manifest=True and no manifest entry,
    loading must raise rather than silently deserializing an unverified file."""
    from src.utils.checkpoint_io import CheckpointIntegrityError

    rng = np.random.default_rng(1)
    n_channels, seq_len, n_windows = 6, 16, 20
    X = rng.normal(size=(n_windows, n_channels, seq_len)).astype(np.float32)
    slices = _sensor_slices()
    config = {
        "reproducibility": {"seed": 42},
        "primary_model": {
            "bilstm_ae_ensemble": {
                "bilstm_autoencoder": {
                    "hidden_size": 8,
                    "latent_dim": 4,
                    "max_epochs": 1,
                    "batch_size": 8,
                }
            }
        },
    }
    ckpt_path = tmp_path / "checkpoints" / "bilstm_ae_test.pt"
    train_bilstm_autoencoder(X, sensor_slices=slices, config=config, checkpoint_path=ckpt_path)

    # Remove the manifest entry to simulate an unregistered/tampered checkpoint.
    (ckpt_path.parent / "checkpoint_manifest.json").write_text('{"version": 1, "files": {}}')

    with pytest.raises(CheckpointIntegrityError):
        load_bilstm_autoencoder(ckpt_path, require_manifest=True)
