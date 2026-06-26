"""
LOSO out-of-fold evaluation for Healthy-reference gait anomaly screening (ANOM-001).

Primary manuscript endpoint when ``evaluation.primary_endpoint: anomaly_ensemble``.
Pseudo ground truth for metrics: non-Healthy trial = positive (screening target).

Threshold discipline (CRIT-001): Youden J is fit on **train-fold** scores only
(models trained on Healthy train rows), then applied to held-out test scores.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from src.evaluation.competitor_metrics import compute_discriminative_metrics, nan_discriminative_metrics

from src.evaluation.anomaly_threshold_policy import fit_anomaly_threshold
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
    score_fitted_method,
)
from src.dataset.subject_split import assert_loso_fold_disjoint
from src.utils.reproducibility import get_pipeline_seed
from src.utils.progress import progress_bar

MIN_HEALTHY_TRAIN = 5


def _load_trial_matrix(config: dict) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    feat_dir = Path(config["paths"]["features"])
    df = pd.read_parquet(feat_dir / "trial_features.parquet")
    if "trial_id" in df.columns:
        df = df[~df["trial_id"].astype(str).str.startswith("daphnet_")].reset_index(drop=True)
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


def _metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    return compute_discriminative_metrics(
        y_true, y_pred, y_score=y_score, config={"dataset": {"label_mode": "binary"}}
    )


def _score_block(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    *,
    method: str,
    fold_threshold_mean: float,
    fold_threshold_std: float,
    n_threshold_folds: int,
    feature_selection_protocol: str = PROTOCOL_ANOMALY_LOSO,
    threshold_source: str = "loso_healthy_train_percentile",
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)
    valid = np.isfinite(y_score)
    yt, ys, yp = y_true[valid], y_score[valid], y_pred[valid]
    row: dict[str, Any] = {
        "method": method,
        "evaluation_mode": "loso_trial_oof",
        "feature_selection_protocol": feature_selection_protocol,
        "n_trials_scored": int(valid.sum()),
        "threshold_source": threshold_source,
        "n_threshold_folds": int(n_threshold_folds),
    }
    if len(np.unique(yt)) < 2:
        row.update(nan_discriminative_metrics())
        row.update(
            {
                "auc": float("nan"),
                "auc_pr": float("nan"),
                "threshold_youden": float("nan"),
                "threshold_youden_std": float("nan"),
            }
        )
        return row
    auc = float(roc_auc_score(yt, ys))
    auc_pr = float(average_precision_score(yt, ys))
    disc = _metrics_from_predictions(yt, yp, y_score=ys)
    row.update(
        {
            "auc": auc,
            "auroc": auc,
            "auc_pr": auc_pr,
            "threshold_youden": float(fold_threshold_mean),
            "threshold_youden_std": float(fold_threshold_std),
            **disc,
        }
    )
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
    oof_method_preds: dict[str, np.ndarray] = {
        m: np.full(n, np.nan, dtype=float) for m in (*ANOMALY_METHODS, "ensemble")
    }
    fold_thresholds: dict[str, list[float]] = {m: [] for m in (*ANOMALY_METHODS, "ensemble")}
    threshold_source_report = "loso_healthy_train_percentile"

    unique_pids = np.unique(groups)
    logger.info("ANOM-001: LOSO anomaly evaluation over {} participants", len(unique_pids))

    for pid in progress_bar(unique_pids, desc="Anomaly LOSO", unit="participant"):
        test_mask = groups == pid
        train_mask = ~test_mask
        assert_loso_fold_disjoint(
            groups[train_mask],
            groups[test_mask],
            held_out_subject=str(pid),
            context="anomaly LOSO",
        )
        healthy_train = train_mask & (cohorts == "Healthy")
        if int(healthy_train.sum()) < MIN_HEALTHY_TRAIN:
            logger.warning(
                "Skipping LOSO fold {} — only {} Healthy train trials",
                pid,
                int(healthy_train.sum()),
            )
            continue

        X_h = X[healthy_train]
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y_true[train_mask]
        healthy_on_train = (cohorts[train_mask] == "Healthy")

        norm_train_layers: list[np.ndarray] = []
        norm_test_layers: list[np.ndarray] = []

        for method in ANOMALY_METHODS:
            sq_test, sr, _, model, scaler = fit_method_scores(
                X_h, X_test, method, random_state=rs
            )
            sq_train = score_fitted_method(model, scaler, X_train)
            thresh, thresh_src = fit_anomaly_threshold(
                sq_train,
                config,
                healthy_train_mask=healthy_on_train,
                y_train=y_train,
            )
            threshold_source_report = thresh_src

            oof_method_scores[method][test_mask] = sq_test
            oof_method_preds[method][test_mask] = (sq_test >= thresh).astype(float)
            fold_thresholds[method].append(float(thresh))

            norm_train_layers.append(normalise_scores(sq_train, sr))
            norm_test_layers.append(normalise_scores(sq_test, sr))

        ens_train = np.mean(np.stack(norm_train_layers, axis=0), axis=0)
        ens_test = np.mean(np.stack(norm_test_layers, axis=0), axis=0)
        ens_thresh, ens_src = fit_anomaly_threshold(
            ens_train,
            config,
            healthy_train_mask=healthy_on_train,
            y_train=y_train,
        )
        threshold_source_report = ens_src

        oof_method_scores["ensemble"][test_mask] = ens_test
        oof_method_preds["ensemble"][test_mask] = (ens_test >= ens_thresh).astype(float)
        fold_thresholds["ensemble"].append(float(ens_thresh))

    rows = []
    for method in (*ANOMALY_METHODS, "ensemble"):
        th_list = fold_thresholds[method]
        th_mean = float(np.mean(th_list)) if th_list else float("nan")
        th_std = float(np.std(th_list)) if len(th_list) > 1 else float("nan")
        rows.append(
            _score_block(
                y_true,
                oof_method_scores[method],
                oof_method_preds[method],
                method=method,
                fold_threshold_mean=th_mean,
                fold_threshold_std=th_std,
                n_threshold_folds=len(th_list),
                threshold_source=threshold_source_report,
            )
        )

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
            "threshold_source": str(r.get("threshold_source", threshold_source_report)),
        }
        thresh_path = metrics_dir / "anomaly_threshold.json"
        thresh_payload = {
            "primary_method": "ensemble",
            "probability": float(r["threshold_youden"])
            if pd.notna(r.get("threshold_youden"))
            else 0.5,
            "source": str(r.get("threshold_source", threshold_source_report)),
            "eval_positive_definition": "non_Healthy_trial",
        }
        thresh_path.write_text(json.dumps(thresh_payload, indent=2), encoding="utf-8")

    write_anomaly_primary_artifacts(metrics_dir, registry)

    oof_path = metrics_dir / "anomaly_loso_oof_scores.csv"
    oof_df = metadata.copy()
    for method, scores in oof_method_scores.items():
        oof_df[f"{method}_score"] = scores
        oof_df[f"{method}_pred_train_youden"] = oof_method_preds[method]
    oof_df["eval_non_healthy"] = y_true
    oof_df.to_csv(oof_path, index=False)

    from src.evaluation.threshold_validation import run_threshold_validation

    run_threshold_validation(config)

    return metrics_df
