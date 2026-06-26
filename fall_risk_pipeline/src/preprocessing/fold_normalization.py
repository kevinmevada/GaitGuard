"""
Fold-safe normalization — fit on training data only (v13 GaitGuard fix).

Never call ``fit`` on validation, test, or full-dataset matrices before splitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class PerChannelZNorm:
    """Per-channel mean/std for window tensors (N, C, T)."""

    mean: np.ndarray  # (C, 1)
    std: np.ndarray   # (C, 1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(f"Expected (N, C, T), got {X.shape}")
        n, c, t = X.shape
        flat = X.transpose(1, 0, 2).reshape(c, -1)
        normed = ((flat - self.mean) / self.std).reshape(c, n, t).transpose(1, 0, 2)
        return normed.astype(np.float32)


def fit_per_channel_znorm(X_train: np.ndarray) -> PerChannelZNorm:
    """Fit mean/std on training windows only."""
    if X_train.ndim != 3 or len(X_train) == 0:
        raise ValueError("Need non-empty (N, C, T) training windows")
    _, c, _ = X_train.shape
    flat = X_train.transpose(1, 0, 2).reshape(c, -1)
    mean = flat.mean(axis=1, keepdims=True)
    std = flat.std(axis=1, keepdims=True) + 1e-8
    return PerChannelZNorm(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_per_channel_znorm(X: np.ndarray, norm: PerChannelZNorm) -> np.ndarray:
    return norm.transform(X)


def fit_standard_scaler_train_only(X_train: np.ndarray) -> StandardScaler:
    """sklearn StandardScaler — fit on train rows only."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def transform_splits(
    scaler: StandardScaler,
    X_train: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Transform train + optional val/test without refitting."""
    outs = [scaler.transform(X_train)]
    outs.extend(scaler.transform(x) for x in arrays)
    return tuple(outs)


def reconstruction_threshold_train_only(
    train_errors: np.ndarray,
    percentile: float = 90.0,
) -> float:
    """
    Anomaly / AE deploy threshold from training-fold reconstruction errors only.

    v13 fix: never use held-out or full-dataset error distributions for this.
    """
    err = np.asarray(train_errors, dtype=float)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return float("nan")
    return float(np.percentile(err, percentile))
