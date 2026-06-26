"""
Anomaly threshold policy — v13 train-fold calibration only.

Threshold selection is part of model fitting. It must use training-fold scores
(healthy reference distribution only by default), never held-out test scores.

Default: ``np.percentile(healthy_train_scores, 95)`` — descriptive statistic of
the training manifold, not a hyperparameter tuned on validation or test data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.evaluation.clinical_threshold import youden_threshold
from src.preprocessing.fold_normalization import reconstruction_threshold_train_only

THRESHOLD_SOURCE_HEALTHY_PERCENTILE = "loso_healthy_train_percentile"
THRESHOLD_SOURCE_TRAIN_YOUDEN = "loso_train_fold_youden"
DEFAULT_PERCENTILE = 95.0


def _threshold_cfg(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("primary_model", "anomaly", "models"):
        block = config.get(key) or {}
        if isinstance(block, dict):
            if key == "primary_model":
                inner = (block.get("bilstm_ae_ensemble") or {}).get("anomaly_threshold")
                if inner:
                    return dict(inner)
            if key == "anomaly" and block.get("anomaly_threshold"):
                return dict(block["anomaly_threshold"])
    global_cfg = (config.get("anomaly_threshold") or {})
    return dict(global_cfg) if global_cfg else {}


def resolve_threshold_policy(config: dict[str, Any]) -> tuple[str, float]:
    cfg = _threshold_cfg(config)
    policy = str(cfg.get("policy", "healthy_train_percentile"))
    percentile = float(cfg.get("percentile", DEFAULT_PERCENTILE))
    return policy, percentile


def fit_anomaly_threshold(
    train_scores: np.ndarray,
    config: dict[str, Any],
    *,
    healthy_train_mask: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
) -> tuple[float, str]:
    """
    Fit deploy threshold on training-fold scores only.

    ``healthy_train_percentile`` (default): percentile of healthy-train scores.
    ``train_fold_youden`` (legacy): Youden J on all train-fold labels + scores.
    """
    scores = np.asarray(train_scores, dtype=float)
    policy, percentile = resolve_threshold_policy(config)

    if policy == "train_fold_youden":
        if y_train is None:
            raise ValueError("train_fold_youden policy requires y_train")
        return float(youden_threshold(np.asarray(y_train), scores)), THRESHOLD_SOURCE_TRAIN_YOUDEN

    if healthy_train_mask is not None and np.any(healthy_train_mask):
        ref = scores[np.asarray(healthy_train_mask, dtype=bool)]
    else:
        ref = scores
    thr = reconstruction_threshold_train_only(ref, percentile=percentile)
    return thr, THRESHOLD_SOURCE_HEALTHY_PERCENTILE


def apply_threshold(test_scores: np.ndarray, threshold: float) -> np.ndarray:
    """Binary flags for test-fold scores using a train-fitted threshold."""
    return (np.asarray(test_scores, dtype=float) >= float(threshold)).astype(int)
