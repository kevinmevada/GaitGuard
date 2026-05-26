"""Evaluation helpers for 3-class (low / moderate / high) fall-risk labels."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.dataset.label_policy import MULTICLASS_NAMES


def predict_multiclass(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_samples, n_classes) probabilities and argmax predictions."""
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1).astype(int)
    return proba, pred


def build_multiclass_metric_payload(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    cohorts: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    if y_pred is None:
        y_pred = np.argmax(y_proba, axis=1).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    try:
        auc_macro = float(
            roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro", labels=labels
            )
        )
    except ValueError:
        auc_macro = float("nan")

    try:
        auc_weighted = float(
            roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="weighted", labels=labels
            )
        )
    except ValueError:
        auc_weighted = float("nan")

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    per_class: dict[str, dict] = {}
    for lbl in labels:
        key = str(lbl)
        if key in report:
            per_class[MULTICLASS_NAMES.get(lbl, key)] = {
                "precision": float(report[key]["precision"]),
                "recall": float(report[key]["recall"]),
                "f1": float(report[key]["f1-score"]),
                "support": float(report[key]["support"]),
            }

    return {
        "model": name,
        "label_mode": "multiclass",
        "auc": auc_macro,
        "auc_weighted_ovr": auc_weighted,
        "auc_pr": float("nan"),
        "auc_ci_low": float("nan"),
        "auc_ci_high": float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": macro_f1,
        "f1_weighted": weighted_f1,
        "macro_f1": macro_f1,
        "sensitivity": float("nan"),
        "specificity": float("nan"),
        "decision_threshold": float("nan"),
        "threshold_strategy": "argmax",
        "y_true": y_true,
        "y_prob": y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel(),
        "y_proba_full": y_proba,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "report": report,
        "per_class_metrics": per_class,
        "cohorts": cohorts,
    }

