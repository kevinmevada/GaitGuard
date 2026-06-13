"""
Shared trial-level anomaly scoring (Healthy-reference one-class models).

Used by deploy ``GaitAnomalyDetector`` and LOSO evaluation (ANOM-001).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

ANOMALY_METHODS = ("isolation_forest", "lof", "one_class_svm")


def normalise_scores(scores: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Min-max normalise to [0, 1] using *reference* range (Healthy train by default)."""
    scores = np.asarray(scores, dtype=float)
    ref = scores if reference is None else np.asarray(reference, dtype=float)
    lo, hi = float(np.nanmin(ref)), float(np.nanmax(ref))
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    out = (scores - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def fit_method_scores(
    X_healthy_train: np.ndarray,
    X_query: np.ndarray,
    method: str,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, Any]:
    """
    Fit scaler + one-class model on Healthy training rows; score ``X_query``.

    Returns (query_scores, healthy_train_scores, binary_flags, model, scaler).
    """
    from sklearn.preprocessing import StandardScaler

    if X_healthy_train.shape[0] < 2:
        raise ValueError("Need at least 2 Healthy training rows for anomaly fitting")

    scaler = StandardScaler()
    scaler.fit(X_healthy_train)
    Xh = scaler.transform(X_healthy_train)
    Xq = scaler.transform(X_query)

    if method == "isolation_forest":
        model = IsolationForest(
            contamination=0.1,
            random_state=random_state,
            n_estimators=100,
        )
        model.fit(Xh)
        labels = model.predict(Xq)
        scores = -model.decision_function(Xq)
    elif method == "lof":
        model = LocalOutlierFactor(
            n_neighbors=min(20, max(2, Xh.shape[0] - 1)),
            contamination=0.1,
            novelty=True,
        )
        model.fit(Xh)
        labels = model.predict(Xq)
        scores = -model.decision_function(Xq)
    elif method == "one_class_svm":
        model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
        model.fit(Xh)
        labels = model.predict(Xq)
        scores = -model.decision_function(Xq)
    else:
        raise ValueError(f"Unknown anomaly method: {method}")

    binary = (labels == -1).astype(int)
    scores_train = -model.decision_function(Xh)
    return (
        np.asarray(scores, dtype=float),
        np.asarray(scores_train, dtype=float),
        binary,
        model,
        scaler,
    )


def ensemble_scores(
    method_score_map: dict[str, np.ndarray],
    *,
    reference_masks: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Average per-method scores after independent min-max normalisation."""
    if not method_score_map:
        raise ValueError("method_score_map is empty")
    layers = []
    for name, scores in method_score_map.items():
        ref = None
        if reference_masks and name in reference_masks:
            ref = scores[reference_masks[name]]
        layers.append(normalise_scores(scores, ref))
    return np.mean(np.stack(layers, axis=0), axis=0)


def eval_binary_labels(cohorts: np.ndarray) -> np.ndarray:
    """Screening pseudo-label: 1 = non-Healthy trial (evaluation only)."""
    return (np.asarray(cohorts).astype(str) != "Healthy").astype(int)
