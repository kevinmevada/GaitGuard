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
) -> tuple[float, float, float, str]:
    """Bootstrap 95% CI for binary AUC on LOSO OOF (one row per subject).

    Returns ``(auc_full, ci_low, ci_high, status)`` where ``auc_full`` is the
    point-estimate AUC computed once on the full sample (avoids callers having
    to redundantly recompute ``roc_auc_score`` themselves).
    """
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    n = len(y)
    if n < 3 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan"), "insufficient_data"

    try:
        auc_full = float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan"), float("nan"), float("nan"), "undefined_auc"

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
        return auc_full, float("nan"), float("nan"), "insufficient_bootstrap"

    ci_low = float(np.percentile(samples, 2.5))
    ci_high = float(np.percentile(samples, 97.5))
    return auc_full, ci_low, ci_high, "subject_bootstrap"


def grouped_bootstrap_binary_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int = 2000,
) -> tuple[float, float, float, str]:
    """Bootstrap 95% CI for binary AUC when rows are clustered by subject.

    Unlike :func:`subject_bootstrap_binary_auc_ci` (one row per subject),
    this resamples whole subjects (with replacement) and pools all of that
    subject's rows (e.g. sliding windows) per bootstrap draw. This avoids
    treating within-subject windows as independent samples, which would
    otherwise understate the true uncertainty (pseudo-replication).

    Returns ``(auc_full, ci_low, ci_high, status)``.
    """
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    grp = np.asarray(groups)
    n = len(y)
    if n < 3 or len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan"), "insufficient_data"

    try:
        auc_full = float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan"), float("nan"), float("nan"), "undefined_auc"

    unique_groups = np.unique(grp)
    if len(unique_groups) < 3:
        return auc_full, float("nan"), float("nan"), "insufficient_groups"

    group_indices = {g: np.where(grp == g)[0] for g in unique_groups}
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_indices[g] for g in chosen])
        yt, ys = y[idx], scores[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            samples.append(float(roc_auc_score(yt, ys)))
        except ValueError:
            continue

    if len(samples) < 10:
        return auc_full, float("nan"), float("nan"), "insufficient_bootstrap"

    ci_low = float(np.percentile(samples, 2.5))
    ci_high = float(np.percentile(samples, 97.5))
    return auc_full, ci_low, ci_high, "subject_cluster_bootstrap"
