"""
LOSO out-of-fold evaluation for Healthy-reference gait anomaly screening (ANOM-001).

Primary manuscript endpoint when ``evaluation.primary_endpoint: anomaly_ensemble``.
Pseudo ground truth for metrics: non-Healthy trial = positive (screening target).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src.evaluation.clinical_threshold import youden_threshold
from src.evaluation.primary_endpoint import (
    ENDPOINT_ANOMALY_ENSEMBLE,
    PROTOCOL_ANOMALY_LOSO,
    write_anomaly_primary_artifacts,
)
from src.models.anomaly_scoring import (
    ANOMALY_METHODS,
    eval_binary_labels,
    fit_method_scores,
    normalise_scores,
)
from src.utils.reproducibility import get_pipeline_seed

MIN_HEALTHY_TRAIN = 5


def _load_trial_matrix(config: dict) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    feat_dir = Path(config["paths"]["features"])
    df = pd.read_parquet(feat_dir / "trial_features.parquet")
    meta_cols = [
        "trial_id",
        "participant_id",
        "cohort",
        "risk_label",
        "multiclass_label",
        "fall_probability",
        "laterality_biased",
    ]
    feature_cols = [
        c
        for c in df.columns
        if c not in meta_cols and df[c].dtype in (np.float32, np.float64, np.int32, np.int64)
    ]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).values
    metadata = df[[c for c in meta_cols if c in df.columns]].copy()
    return X, metadata, feature_cols


def _metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thresh: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score, dtype=float) >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    return {
        "threshold": float(thresh),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn + 1e-10)),
        "specificity": float(tn / (tn + fp + 1e-10)),
    }


def _score_block(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    method: str,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    valid = np.isfinite(y_score)
    yt, ys = y_true[valid], y_score[valid]
    row: dict[str, Any] = {
        "method": method,
        "evaluation_mode": "loso_trial_oof",
        "feature_selection_protocol": PROTOCOL_ANOMALY_LOSO,
        "n_trials_scored": int(valid.sum()),
    }
    if len(np.unique(yt)) < 2:
        row.update(
            {
                "auc": float("nan"),
                "auc_pr": float("nan"),
                "accuracy": float("nan"),
                "f1": float("nan"),
                "sensitivity": float("nan"),
                "specificity": float("nan"),
                "threshold_youden": float("nan"),
            }
        )
        return row
    auc = float(roc_auc_score(yt, ys))
    auc_pr = float(average_precision_score(yt, ys))
    thresh = youden_threshold(yt, ys)
    m = _metrics_at_threshold(yt, ys, thresh)
    row.update({"auc": auc, "auc_pr": auc_pr, "threshold_youden": thresh, **m})
    return row


def run_anomaly_loso_evaluation(config: dict) -> pd.DataFrame:
    """LOSO OOF anomaly screening; writes ``anomaly_metrics.csv`` + registry."""
    X, metadata, _ = _load_trial_matrix(config)
    groups = metadata["participant_id"].values
    cohorts = metadata["cohort"].values
    y_true = eval_binary_labels(cohorts)
    rs = get_pipeline_seed(config)

    n = len(X)
    oof_method_scores: dict[str, np.ndarray] = {
        m: np.full(n, np.nan, dtype=float) for m in (*ANOMALY_METHODS, "ensemble")
    }

    unique_pids = np.unique(groups)
    logger.info("ANOM-001: LOSO anomaly evaluation over {} participants", len(unique_pids))

    for pid in unique_pids:
        test_mask = groups == pid
        train_mask = ~test_mask
        healthy_train = train_mask & (cohorts == "Healthy")
        if int(healthy_train.sum()) < MIN_HEALTHY_TRAIN:
            logger.warning(
                "Skipping LOSO fold {} — only {} Healthy train trials",
                pid,
                int(healthy_train.sum()),
            )
            continue

        X_h = X[healthy_train]
        X_test = X[test_mask]
        norm_test_layers: list[np.ndarray] = []

        for method in ANOMALY_METHODS:
            sq, sr, _, _, _ = fit_method_scores(X_h, X_test, method, random_state=rs)
            oof_method_scores[method][test_mask] = sq
            norm_test_layers.append(normalise_scores(sq, sr))

        ens_test = np.mean(np.stack(norm_test_layers, axis=0), axis=0)
        oof_method_scores["ensemble"][test_mask] = ens_test

    rows = [_score_block(y_true, oof_method_scores[m], method=m) for m in ANOMALY_METHODS]
    rows.append(_score_block(y_true, oof_method_scores["ensemble"], method="ensemble"))

    metrics_df = pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "anomaly_metrics.csv"
    metrics_df.to_csv(out_path, index=False)
    logger.info("ANOM-001: anomaly_metrics.csv saved → {}", out_path)

    ens_row = metrics_df[metrics_df["method"] == "ensemble"]
    if ens_row.empty:
        ens_row = metrics_df.iloc[[0]] if not metrics_df.empty else pd.DataFrame()
    registry = {
        "primary_endpoint": ENDPOINT_ANOMALY_ENSEMBLE,
        "manuscript_guidance": (
            "Primary endpoint: trial-level LOSO OOF anomaly ensemble detecting "
            "non-Healthy gait vs Healthy-reference training. Supervised pathology-tier "
            "metrics in metrics.csv are secondary."
        ),
        "registered_endpoints": {},
    }
    if not ens_row.empty:
        r = ens_row.iloc[0]
        registry["registered_endpoints"][ENDPOINT_ANOMALY_ENSEMBLE] = {
            "model": "ensemble",
            "auc": float(r["auc"]) if pd.notna(r["auc"]) else None,
            "auc_pr": float(r["auc_pr"]) if pd.notna(r.get("auc_pr")) else None,
            "metric_source": "loso_trial_oof_healthy_reference",
            "feature_selection_protocol": PROTOCOL_ANOMALY_LOSO,
            "threshold_youden": float(r["threshold_youden"])
            if pd.notna(r.get("threshold_youden"))
            else None,
        }
        thresh_path = metrics_dir / "anomaly_threshold.json"
        thresh_payload = {
            "primary_method": "ensemble",
            "probability": float(r["threshold_youden"])
            if pd.notna(r.get("threshold_youden"))
            else 0.5,
            "source": "loso_oof_youden",
            "eval_positive_definition": "non_Healthy_trial",
        }
        thresh_path.write_text(json.dumps(thresh_payload, indent=2), encoding="utf-8")

    write_anomaly_primary_artifacts(metrics_dir, registry)

    oof_path = metrics_dir / "anomaly_loso_oof_scores.csv"
    oof_df = metadata.copy()
    for method, scores in oof_method_scores.items():
        oof_df[f"{method}_score"] = scores
    oof_df["eval_non_healthy"] = y_true
    oof_df.to_csv(oof_path, index=False)

    return metrics_df
