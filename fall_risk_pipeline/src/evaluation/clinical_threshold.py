"""
Clinical screening context and data-driven probability cutoffs (Youden J).

Morse Fall Scale (MFS) and STRATIFY are established inpatient fall-risk screens;
this pipeline's IMU classifier uses a **separate**, validation-derived probability
threshold — not a direct conversion to MFS/STRATIFY scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from src.dataset.label_policy import get_dataset_label_config

CLINICAL_SCREENING_TOOLS: dict[str, Any] = {
    "morse_fall_scale": {
        "name": "Morse Fall Scale (MFS)",
        "citation": (
            "Morse JM, Morse RM, Tinzoh J. Development of a scale to identify the "
            "fall-prone patient. Can J Aging. 1989;8(4):373-385."
        ),
        "typical_cutoff": "Score ≥ 45 indicates high fall risk (inpatient nursing).",
        "score_range": "0–125",
        "relation_to_imu_model": (
            "MFS uses clinical observation items; our model uses wearable IMU gait "
            "features. Thresholds are not interchangeable without calibration study."
        ),
    },
    "stratify": {
        "name": "STRATIFY",
        "citation": (
            "Oliver D, Britton M, Seed P, Martin FC, Hopper AH. Development and evaluation "
            "of evidence based risk assessment tool (STRATIFY) to predict which elderly "
            "inpatients will fall. BMJ. 1997;315(7115):1049-1053."
        ),
        "typical_cutoff": "Score ≥ 5 indicates high fall risk.",
        "score_range": "0–5",
        "relation_to_imu_model": (
            "STRATIFY is a brief clinical checklist; IMU-derived risk probability "
            "requires a data-driven cutoff (Youden J) on held-out subjects."
        ),
    },
}

ARTIFACT_FILENAME = "clinical_threshold.json"

DEFAULT_DECISION_THRESHOLD = 0.5
THRESHOLD_STRATEGY_INNER_GROUP_OOF = "inner_group_oof"
THRESHOLD_STRATEGY_INNER_GROUP_SOFT_VOTING_OOF = "inner_group_soft_voting_oof"
THRESHOLD_STRATEGY_FIXED_INSUFFICIENT_GROUPS = "fixed_0.5_insufficient_inner_groups"
THRESHOLD_STRATEGY_FIXED_OOF_UNAVAILABLE = "fixed_0.5_inner_oof_unavailable"
FIXED_THRESHOLD_FALLBACK_STRATEGIES = frozenset(
    {
        THRESHOLD_STRATEGY_FIXED_INSUFFICIENT_GROUPS,
        THRESHOLD_STRATEGY_FIXED_OOF_UNAVAILABLE,
    }
)


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Youden J optimal threshold: argmax (TPR − FPR) on ROC."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    threshold = float(thresholds[idx])
    if not np.isfinite(threshold):
        return 0.5
    return max(0.0, min(1.0, threshold))


def fixed_threshold_when_inner_cv_unavailable() -> tuple[float, str]:
    """
    ML-037: when inner StratifiedGroupKFold cannot run (n_groups < 3 or single class),
    use a fixed 0.5 cutoff instead of tuning Youden on in-sample train predictions.
    """
    return DEFAULT_DECISION_THRESHOLD, THRESHOLD_STRATEGY_FIXED_INSUFFICIENT_GROUPS


def threshold_from_inner_oof(
    y_train: np.ndarray,
    oof_scores: np.ndarray,
    *,
    n_splits: int,
    success_strategy: str = THRESHOLD_STRATEGY_INNER_GROUP_OOF,
) -> tuple[float, str]:
    """
    ML-037: pick Youden from inner grouped OOF, or fixed 0.5 when OOF is too sparse.
    Never tune on in-sample train predictions.
    """
    y_train = np.asarray(y_train).astype(int)
    oof_scores = np.asarray(oof_scores, dtype=float)
    valid = np.isfinite(oof_scores)
    min_valid = max(10, n_splits)
    if int(valid.sum()) < min_valid or len(np.unique(y_train[valid])) < 2:
        return DEFAULT_DECISION_THRESHOLD, THRESHOLD_STRATEGY_FIXED_OOF_UNAVAILABLE
    return youden_threshold(y_train[valid], oof_scores[valid]), success_strategy


def elevated_class_indices(config: dict) -> list[int]:
    """Multiclass indices treated as 'elevated' for binary collapse (API + Youden)."""
    cfg = get_dataset_label_config(config)
    mode = cfg["label_mode"]
    if mode == "binary":
        return [1]
    threshold = int(cfg["high_risk_threshold"])
    return [i for i in (0, 1, 2) if i >= threshold]


def collapse_labels_binary(y: np.ndarray, config: dict) -> np.ndarray:
    cfg = get_dataset_label_config(config)
    if cfg["label_mode"] == "binary":
        return np.asarray(y).astype(int)
    threshold = int(cfg["high_risk_threshold"])
    return (np.asarray(y).astype(int) >= threshold).astype(int)


def elevated_risk_probability(y_proba: np.ndarray, config: dict) -> np.ndarray:
    """
    P(elevated fall-risk) for API / Youden.

    Binary: positive-class probability. Multiclass: sum of class probs ≥ threshold tier.
    """
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        return proba.ravel()
    if proba.ndim != 2:
        raise ValueError(f"Expected 1D or 2D probabilities, got shape {proba.shape}")
    cols = elevated_class_indices(config)
    cols = [c for c in cols if c < proba.shape[1]]
    if not cols:
        return proba[:, -1]
    return proba[:, cols].sum(axis=1)


def metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    sens = float(tp / (tp + fn + 1e-10))
    spec = float(tn / (tn + fp + 1e-10))
    ppv = float(tp / (tp + fp + 1e-10))
    npv = float(tn / (tn + fn + 1e-10))
    acc = float((tp + tn) / (tp + tn + fp + fn + 1e-10))
    return {
        "threshold_probability": float(threshold),
        "threshold_risk_score_0_100": int(round(threshold * 100)),
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "accuracy": acc,
        "n_positive": int(tp + fn),
        "n_negative": int(tn + fp),
    }


def assign_risk_level(
    elevated_prob: float,
    youden_prob: float,
    *,
    include_borderline_moderate: bool = True,
) -> str:
    """
  Assign API risk band relative to validated Youden cutoff.

  - high: prob ≥ Youden (primary clinical cutoff)
  - moderate: [0.5×Youden, Youden) optional borderline screening zone
  - low: prob < 0.5×Youden
    """
    if elevated_prob >= youden_prob:
        return "high"
    if include_borderline_moderate and elevated_prob >= 0.5 * youden_prob:
        return "moderate"
    return "low"


def build_clinical_threshold_artifact(
    config: dict,
    *,
    reference_model: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    train_fold_youden_mean: float | None = None,
    train_fold_metrics: dict[str, float] | None = None,
    validation_strategy: str = "loso_subject_grouped",
) -> dict[str, Any]:
    y_bin = collapse_labels_binary(y_true, config)
    p_elevated = elevated_risk_probability(
        np.asarray(y_proba) if np.asarray(y_proba).ndim > 1 else np.asarray(y_proba).reshape(-1, 1),
        config,
    )

    eval_youden = youden_threshold(y_bin, p_elevated)
    eval_metrics = metrics_at_threshold(y_bin, p_elevated, eval_youden)

    primary_prob = (
        float(train_fold_youden_mean)
        if train_fold_youden_mean is not None and np.isfinite(train_fold_youden_mean)
        else eval_youden
    )
    primary_source = (
        "loso_train_fold_youden_mean"
        if train_fold_youden_mean is not None and np.isfinite(train_fold_youden_mean)
        else "loso_oof_eval_youden"
    )

    primary_metrics = metrics_at_threshold(y_bin, p_elevated, primary_prob)
    if train_fold_metrics:
        sens = train_fold_metrics.get("sensitivity")
        spec = train_fold_metrics.get("specificity")
        if sens is not None and spec is not None and np.isfinite(sens) and np.isfinite(spec):
            primary_metrics["sensitivity"] = float(sens)
            primary_metrics["specificity"] = float(spec)
            if "accuracy" in train_fold_metrics and np.isfinite(train_fold_metrics["accuracy"]):
                primary_metrics["accuracy"] = float(train_fold_metrics["accuracy"])

    cfg = get_dataset_label_config(config)
    return {
        "artifact_version": 1,
        "reference_model": reference_model,
        "validation_strategy": validation_strategy,
        "label_mode": cfg["label_mode"],
        "binary_collapse": {
            "high_risk_threshold": cfg.get("high_risk_threshold"),
            "binary_strategy": cfg.get("binary_strategy"),
            "elevated_definition": (
                f"multiclass tier ≥ {cfg['high_risk_threshold']}"
                if cfg["label_mode"] == "multiclass"
                else "positive class (risk_label=1)"
            ),
        },
        "primary_cutoff": {
            "method": "youden_j",
            "source": primary_source,
            "probability": primary_prob,
            "risk_score_0_100": int(round(primary_prob * 100)),
            **primary_metrics,
        },
        "eval_youden_cutoff": {
            "method": "youden_j",
            "source": "loso_oof_eval_youden",
            "note": "Fit on pooled OOF scores — slightly optimistic; use primary_cutoff for deployment.",
            **eval_metrics,
        },
        "fixed_cutoff_0_5": metrics_at_threshold(y_bin, p_elevated, 0.5),
        "removed_arbitrary_api_bands": {
            "risk_score_high_gte": 70,
            "risk_score_moderate_gte": 40,
            "reason": "Not derived from ROC or clinical scales; replaced by Youden J.",
        },
        "borderline_moderate_band": {
            "definition": "elevated_probability in [0.5 × Youden, Youden)",
            "not_a_clinical_guideline": True,
        },
        "clinical_screening_tools": CLINICAL_SCREENING_TOOLS,
    }


def save_clinical_threshold_artifact(payload: dict[str, Any], metrics_dir: Path) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / ARTIFACT_FILENAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_clinical_threshold_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def default_clinical_threshold() -> dict[str, Any]:
    """Fallback when evaluate has not been run."""
    return {
        "primary_cutoff": {
            "method": "youden_j",
            "source": "default",
            "probability": 0.5,
            "risk_score_0_100": 50,
            "sensitivity": float("nan"),
            "specificity": float("nan"),
        },
        "clinical_screening_tools": CLINICAL_SCREENING_TOOLS,
        "warning": "Run pipeline evaluate stage to export validated Youden threshold.",
    }
