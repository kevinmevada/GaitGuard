"""
Core discriminative metrics for the competitor matrix (PUB-002).

Reported metrics (literature-aligned):
  - F1 (weighted) — Navita, Moon, Trabassi, Sadeghsalehi
  - Balanced accuracy — Sadeghsalehi 2025
  - MCC — Sadeghsalehi 2025 (hardest to game; lead abstract if > 0.7)
  - AUROC — Moon, Trabassi, Sadeghsalehi (headline threshold-independent)
  - Sensitivity, specificity, precision — Navita, Moon, Trabassi, Sadeghsalehi
  - Cohen's κ — Moon, clinical cross-rater audits
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.dataset.label_policy import is_binary_task

DISCRIMINATIVE_COLUMNS = (
    "f1_weighted",
    "f1",  # alias kept for backward-compatible CSV consumers
    "balanced_accuracy",
    "mcc",
    "auroc",
    "sensitivity",
    "specificity",
    "precision",
    "cohen_kappa",
    "accuracy",
)


def nan_discriminative_metrics() -> dict[str, float]:
    return {k: float("nan") for k in DISCRIMINATIVE_COLUMNS}


def _binary_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return {
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
        }
    tn, fp, fn, tp = cm.ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    return {"sensitivity": sens, "specificity": spec, "precision": prec}


def _multiclass_ovr_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> dict[str, float]:
    sens_list: list[float] = []
    spec_list: list[float] = []
    prec_list: list[float] = []
    for lbl in labels:
        y_bin = (y_true == lbl).astype(int)
        y_hat = (y_pred == lbl).astype(int)
        cm = confusion_matrix(y_bin, y_hat, labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        if tp + fn > 0:
            sens_list.append(float(tp / (tp + fn)))
        if tn + fp > 0:
            spec_list.append(float(tn / (tn + fp)))
        if tp + fp > 0:
            prec_list.append(float(tp / (tp + fp)))
    return {
        "sensitivity": float(np.mean(sens_list)) if sens_list else float("nan"),
        "specificity": float(np.mean(spec_list)) if spec_list else float("nan"),
        "precision": float(np.mean(prec_list)) if prec_list else float("nan"),
    }


def _compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray | None,
    y_proba: np.ndarray | None,
    config: dict[str, Any] | None,
) -> float:
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            return float(roc_auc_score(y_true, np.asarray(y_score, dtype=float)))
        except ValueError:
            return float("nan")

    if y_proba is None or len(y_proba) == 0:
        return float("nan")

    y_proba = np.asarray(y_proba, dtype=float)
    if is_binary_task(y_true, config or {}) or y_proba.ndim == 1 or y_proba.shape[1] == 1:
        score = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] > 1 else y_proba.ravel()
        try:
            return float(roc_auc_score(y_true, score))
        except ValueError:
            return float("nan")

    labels = sorted(set(np.unique(y_true)) | set(np.unique(np.argmax(y_proba, axis=1))))
    n_classes = max(labels) + 1 if labels else y_proba.shape[1]
    if y_proba.shape[1] < n_classes:
        padded = np.zeros((len(y_true), n_classes), dtype=float)
        padded[:, : y_proba.shape[1]] = y_proba
        y_proba = padded
    try:
        return float(
            roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
        )
    except ValueError:
        return float("nan")


def compute_discriminative_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    y_score: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, float]:
    """
    Compute full competitor-matrix discriminative metrics from OOF labels/predictions.

    For binary screening (anomaly), pass ``y_score`` for AUROC.
    For supervised multiclass, pass ``y_proba`` (n_samples, n_classes).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return nan_discriminative_metrics()

    binary = is_binary_task(y_true, config or {}) or len(np.unique(y_true)) == 2

    f1_w = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except ValueError:
        mcc = float("nan")
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred))
    except ValueError:
        kappa = float("nan")

    auroc = _compute_auroc(y_true, y_score, y_proba, config)

    if binary:
        rates = _binary_rates(y_true, y_pred)
    else:
        labels = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        rates = _multiclass_ovr_rates(y_true, y_pred, labels)

    acc = float(np.mean(y_true == y_pred))

    out = {
        "f1_weighted": f1_w,
        "f1": f1_w,
        "balanced_accuracy": bal,
        "mcc": mcc,
        "auroc": auroc,
        "sensitivity": rates["sensitivity"],
        "specificity": rates["specificity"],
        "precision": rates["precision"],
        "cohen_kappa": kappa,
        "accuracy": acc,
    }
    return out


def metrics_row_for_csv(metrics: dict[str, float]) -> dict[str, float]:
    """Subset for CSV export with stable column order."""
    return {k: float(metrics.get(k, float("nan"))) for k in DISCRIMINATIVE_COLUMNS if k != "f1" or True}
