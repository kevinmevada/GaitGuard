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
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from loguru import logger

from src.dataset.label_policy import MULTICLASS_NAMES


def is_multiclass_metric_result(result: dict) -> bool:
    """True when result carries multiclass probabilities (not a binary y_prob vector)."""
    if result.get("label_mode") == "multiclass":
        return True
    y_proba_full = result.get("y_proba_full")
    return y_proba_full is not None and np.asarray(y_proba_full).ndim == 2


def predict_multiclass(model: Any, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (n_samples, n_classes) probabilities and argmax predictions."""
    proba = model.predict_proba(X)
    pred = np.argmax(proba, axis=1).astype(int)
    return proba, pred


def _bootstrap_multiclass_auc_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: list[int],
    *,
    seed: int,
    n_bootstrap: int = 2000,
) -> tuple[float, float]:
    """Bootstrap 95 % CI for macro-OVR AUC (multiclass)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_proba[idx]
        if len(np.unique(yt)) < len(labels):
            continue
        try:
            samples.append(
                float(roc_auc_score(yt, yp, multi_class="ovr", average="macro", labels=labels))
            )
        except ValueError:
            continue
    if not samples:
        return float("nan"), float("nan")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def build_multiclass_metric_payload(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray | None = None,
    *,
    seed: int,
    cohorts: np.ndarray | None = None,
    n_bootstrap: int = 2000,
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
    per_class_roc: dict[int, dict] = {}
    macro_sens, macro_spec = [], []

    for lbl in labels:
        key = str(lbl)
        class_name = MULTICLASS_NAMES.get(lbl, key)

        # OvR binary vectors for this class
        y_bin = (y_true == lbl).astype(int)
        p_bin = y_proba[:, lbl] if lbl < y_proba.shape[1] else np.zeros(len(y_true))

        # Per-class AUC
        try:
            cls_auc = float(roc_auc_score(y_bin, p_bin))
        except ValueError:
            cls_auc = float("nan")

        # Per-class ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_bin, p_bin)
            per_class_roc[lbl] = {
                "fpr": fpr, "tpr": tpr, "auc": cls_auc, "name": class_name,
            }
        except ValueError:
            pass

        # Per-class sensitivity / specificity from confusion matrix
        tp = cm[labels.index(lbl), labels.index(lbl)] if lbl in labels else 0
        fn = cm[labels.index(lbl), :].sum() - tp if lbl in labels else 0
        fp = cm[:, labels.index(lbl)].sum() - tp if lbl in labels else 0
        tn = cm.sum() - tp - fn - fp
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        macro_sens.append(sens)
        macro_spec.append(spec)

        # Per-class average precision
        try:
            cls_ap = float(average_precision_score(y_bin, p_bin))
        except ValueError:
            cls_ap = float("nan")

        entry: dict[str, float] = {
            "precision": float(report[key]["precision"]) if key in report else float("nan"),
            "recall": float(report[key]["recall"]) if key in report else float("nan"),
            "f1": float(report[key]["f1-score"]) if key in report else float("nan"),
            "support": float(report[key]["support"]) if key in report else 0,
            "auc_ovr": cls_auc,
            "avg_precision": cls_ap,
            "sensitivity": sens,
            "specificity": spec,
        }
        per_class[class_name] = entry

    ci_low, ci_high = _bootstrap_multiclass_auc_ci(
        y_true, y_proba, labels, seed=seed, n_bootstrap=n_bootstrap
    )

    avg_sens = float(np.nanmean(macro_sens)) if macro_sens else float("nan")
    avg_spec = float(np.nanmean(macro_spec)) if macro_spec else float("nan")

    # Macro average precision (OvR) — binarize y_true to match y_proba columns
    try:
        y_bin = label_binarize(y_true, classes=labels)
        if y_bin.ndim == 1:
            y_bin = y_bin.reshape(-1, 1)
        macro_ap = float(average_precision_score(y_bin, y_proba, average="macro"))
    except (ValueError, IndexError):
        macro_ap = float("nan")

    payload: dict[str, Any] = {
        "model": name,
        "label_mode": "multiclass",
        "auc": auc_macro,
        "auc_weighted_ovr": auc_weighted,
        "auc_pr": macro_ap,
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": macro_f1,
        "f1_weighted": weighted_f1,
        "macro_f1": macro_f1,
        "sensitivity": avg_sens,
        "specificity": avg_spec,
        "decision_threshold": float("nan"),
        "threshold_strategy": "argmax",
        "y_true": y_true,
        "y_proba_full": y_proba,
        "y_pred": y_pred,
        "confusion_matrix": cm,
        "report": report,
        "per_class_metrics": per_class,
        "per_class_roc": per_class_roc,
        "cohorts": cohorts,
    }
    if y_proba.shape[1] > 1:
        logger.warning(
            "Multiclass metric payload stores y_prob_class_1 for CSV export only; "
            "use y_proba_full for ROC/PR/calibration and cohort metrics."
        )
        payload["y_prob_class_1"] = y_proba[:, 1]
    return payload

