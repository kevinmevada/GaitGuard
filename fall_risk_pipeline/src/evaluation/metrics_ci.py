"""Shared subject-level bootstrap confidence intervals for LOSO OOF metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def subject_bootstrap_binary_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 2000,
    z: float = 1.96,
) -> tuple[float, float, str]:
    """Bootstrap 95% CI for binary AUC on LOSO OOF (one row per subject)."""
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    n = len(y)
    if n < 3 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), "insufficient_data"

    try:
        auc_full = float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan"), float("nan"), "undefined_auc"

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, ys = y[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            samples.append(float(roc_auc_score(yt, ys)))
        except ValueError:
            continue

    if len(samples) < 10:
        return float("nan"), float("nan"), "insufficient_bootstrap"

    ci_low = float(np.percentile(samples, 2.5))
    ci_high = float(np.percentile(samples, 97.5))
    return ci_low, ci_high, "subject_bootstrap"
