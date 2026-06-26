"""Windowing + fold-safe normalization (v13 leakage guards)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataset.train_fit_mask import healthy_train_fit_mask
from src.models.anomaly_detector import _normalise
from src.models.deep_models import create_windows
from src.preprocessing.fold_normalization import (
    fit_per_channel_znorm,
    fit_standard_scaler_train_only,
    reconstruction_threshold_train_only,
    transform_splits,
)
from src.preprocessing.windowing import WindowSpec, window_many_trials, window_single_trial


def test_windows_reset_at_trial_boundary():
    spec = WindowSpec(window_len=20, overlap=0.5, fs_hz=100)
    a = np.ones((2, 50), dtype=np.float32)
    b = np.zeros((2, 50), dtype=np.float32)
    by_trial, _ = window_many_trials({"t1": a, "t2": b}, spec)
    assert len(by_trial["t1"]) == 4
    assert len(by_trial["t2"]) == 4
    # A cross-trial concat would produce more windows spanning the jump at sample 50.
    concat = np.concatenate([a, b], axis=1)
    cross = create_windows(concat, spec.window_len, spec.overlap)
    per_trial_total = sum(len(v) for v in by_trial.values())
    assert len(cross) > len(by_trial["t1"])
    assert len(cross) >= per_trial_total


def test_standard_scaler_fit_train_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(40, 5))
    X_val = rng.normal(size=(10, 5)) + 5.0
    X_test = rng.normal(size=(10, 5)) - 3.0
    scaler = fit_standard_scaler_train_only(X_train)
    X_train_n, X_val_n, X_test_n = transform_splits(scaler, X_train, X_val, X_test)
    assert np.allclose(X_train_n.mean(axis=0), 0.0, atol=1e-6)
    assert not np.allclose(X_val_n.mean(axis=0), 0.0, atol=0.5)


def test_per_channel_znorm_fit_train_only():
    rng = np.random.default_rng(1)
    X_train = rng.normal(size=(8, 3, 32)).astype(np.float32)
    X_test = rng.normal(size=(4, 3, 32)).astype(np.float32) + 2.0
    norm = fit_per_channel_znorm(X_train)
    X_train_n = norm.transform(X_train)
    X_test_n = norm.transform(X_test)
    assert np.allclose(X_train_n.mean(axis=(0, 2)), 0.0, atol=1e-4)


def test_ensemble_normalisation_uses_train_reference_only():
    scores = np.array([0.0, 1.0, 2.0, 10.0, 100.0], dtype=float)
    fit_mask = np.array([True, True, True, False, False])
    normed_train_ref = _normalise(scores, scores[fit_mask])
    normed_full = _normalise(scores, scores)
    assert not np.allclose(normed_train_ref, normed_full)
    assert normed_train_ref[fit_mask][-1] == pytest.approx(1.0)


def test_reconstruction_threshold_train_only():
    train_err = np.array([1.0, 2.0, 3.0, 4.0, 100.0], dtype=float)
    fit = train_err[:4]
    t = reconstruction_threshold_train_only(fit, percentile=90.0)
    assert t < 100.0
    assert t >= 3.0


def test_healthy_train_fit_mask_excludes_val_test():
    from src.dataset.subject_split import build_holdout_from_participants

    meta = pd.DataFrame(
        {
            "participant_id": ["H1", "H2", "H3", "P1"],
            "cohort": ["Healthy", "Healthy", "Healthy", "PD"],
            "trial_id": ["t1", "t2", "t3", "t4"],
        }
    )
    config = {
        "reproducibility": {"seed": 42},
        "dataset": {
            "subject_split": {
                "healthy_train_fraction": 0.34,
                "healthy_val_fraction": 0.33,
                "healthy_test_fraction": 0.33,
                "random_state": 42,
            }
        },
    }
    split = build_holdout_from_participants(meta, config)
    mask = healthy_train_fit_mask(meta, config)
    assert mask.sum() >= 1
    assert mask.sum() < 3
    assert not mask[3]
    assert set(meta.loc[mask, "participant_id"]) <= set(split.train_ids)
