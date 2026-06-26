"""Phase 3 deep representation features."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.features.rocket_features import MiniRocketTransform, RocketTransform
from src.models.bilstm_autoencoder import BiLSTMAutoencoder, SensorChannelSlice
from src.models.deep_models import InceptionMultiscaleExtractor, create_windows


def _synthetic_windows(n: int = 20, channels: int = 8, seq_len: int = 64) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, channels, seq_len)).astype(np.float32)


def test_rocket_transform_max_and_ppv():
    X = _synthetic_windows(12, 4, 48)
    rk = RocketTransform(n_kernels=20, seed=1).fit(X)
    out = rk.transform(X[:3])
    assert out.shape == (3, 40)
    assert np.isfinite(out).all()


def test_minirocket_weights_are_minus_one_or_two():
    X = _synthetic_windows(8, 3, 40)
    mr = MiniRocketTransform(n_kernels=15, seed=2).fit(X)
    for k in mr.kernels:
        assert set(np.unique(k.weights)).issubset({-1.0, 2.0})


def test_bilstm_autoencoder_reconstruction_and_latent():
    X = _synthetic_windows(6, 6, 32)
    slices = [
        SensorChannelSlice("head", 0, 3),
        SensorChannelSlice("lower_back", 3, 6),
    ]
    model = BiLSTMAutoencoder(6, 32, latent_dim=8, hidden_size=16, sensor_slices=slices)
    x = torch.tensor(X[:2])
    recon, h = model(x)
    assert recon.shape == x.shape
    assert h.shape == (2, 32, 8)
    mse = model.per_sensor_mse(x, recon)
    assert "lower_back" in mse
    assert "total" in mse


def test_inception_multiscale_extractor_output_dim():
    X = _synthetic_windows(4, 5, 64)
    model = InceptionMultiscaleExtractor(5, filters=8)
    x = torch.tensor(X)
    out = model(x)
    assert out.shape == (4, 32)
    assert len(InceptionMultiscaleExtractor.feature_names(filters=8)) == 32


def test_create_windows_overlap():
    sig = np.random.default_rng(1).normal(size=(4, 200)).astype(np.float32)
    wins = create_windows(sig, window_len=64, overlap=0.5)
    assert wins.shape[1:] == (4, 64)
    assert len(wins) >= 5


def test_rocket_save_load_roundtrip(tmp_path):
    X = _synthetic_windows(10, 4, 48)
    path = tmp_path / "rocket.npz"
    rk = RocketTransform(n_kernels=12, seed=3).fit(X)
    rk.save(path)
    loaded = RocketTransform.load(path)
    np.testing.assert_allclose(rk.transform(X[:2]), loaded.transform(X[:2]), rtol=1e-5)
