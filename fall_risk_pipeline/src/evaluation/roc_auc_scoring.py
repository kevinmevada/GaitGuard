"""Shared ROC-AUC scoring for binary and multiclass (OvR) tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.dataset.label_policy import is_binary_task


def roc_auc_scoring_name(y, config: dict | None = None) -> str:
    """Sklearn scorer name for grouped CV / RFECV (uses predict_proba)."""
    if is_binary_task(y, config):
        return "roc_auc"
    return "roc_auc_ovr"


def roc_auc_from_proba(
    y_true: np.ndarray,
    proba: np.ndarray,
    config: dict | None = None,
) -> float:
    """Manual AUC from probability matrix (nested CV, bootstrap, etc.)."""
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    if is_binary_task(y_true, config):
        score = proba[:, 1] if proba.ndim > 1 and proba.shape[1] > 1 else proba.ravel()
        return float(roc_auc_score(y_true, score))
    return float(
        roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
    )
