"""
LOSO evaluation — BiLSTM-AE + 3-method one-class ensemble (primary novelty).

Trains on Healthy subjects in each LOSO train fold only. Pathological gait never
appears in AE / IF / OCSVM fitting.

Outputs ``bilstm_ae_anomaly_metrics.csv`` with one row per method + ensemble row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.dataset.subject_split import assert_loso_fold_disjoint
from src.evaluation.anomaly_loso_evaluator import _score_block
from src.evaluation.anomaly_threshold_policy import fit_anomaly_threshold
from src.evaluation.primary_endpoint import (
    ENDPOINT_BILSTM_AE_ENSEMBLE,
    PROTOCOL_BILSTM_AE_LOSO,
    write_bilstm_ae_primary_artifacts,
)
from src.models.anomaly_scoring import eval_binary_labels
from src.models.bilstm_ae_scoring import (
    ENSEMBLE_METHODS,
    METHOD_ENSEMBLE,
    build_fold_trial_scores,
    load_voisard_trial_windows,
)
from src.utils.reproducibility import get_pipeline_seed
from src.utils.progress import progress_bar

ALL_METHODS = (*ENSEMBLE_METHODS, METHOD_ENSEMBLE)
MIN_HEALTHY_TRAIN_TRIALS = 3


def run_bilstm_ae_loso_evaluation(config: dict) -> pd.DataFrame:
    pcfg = (config.get("primary_model") or {}).get("bilstm_ae_ensemble") or {}
    if not pcfg.get("enabled", True):
        logger.info("BiLSTM-AE ensemble disabled in config")
        return pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs = get_pipeline_seed(config)
    bundle = load_voisard_trial_windows(config, require_all_sensors=True)

    trial_to_idx = {tid: i for i, tid in enumerate(bundle.trial_ids)}
    groups = bundle.participant_ids
    cohorts = bundle.cohorts
    y_true = eval_binary_labels(cohorts)
    n = len(bundle.trial_ids)

    oof_scores: dict[str, np.ndarray] = {m: np.full(n, np.nan) for m in ALL_METHODS}
    oof_preds: dict[str, np.ndarray] = {m: np.full(n, np.nan) for m in ALL_METHODS}
    fold_thresholds: dict[str, list[float]] = {m: [] for m in ALL_METHODS}
    fold_threshold_sources: list[str] = []

    unique_pids = np.unique(groups)
    logger.info(
        "BiLSTM-AE 3-method LOSO over {} participants, {} trials",
        len(unique_pids),
        n,
    )

    for pid in progress_bar(unique_pids, desc="BiLSTM-AE LOSO", unit="participant"):
        test_mask = groups == pid
        train_mask = ~test_mask
        assert_loso_fold_disjoint(
            groups[train_mask], groups[test_mask], held_out_subject=str(pid)
        )

        train_tids = [bundle.trial_ids[i] for i in np.where(train_mask)[0]]
        test_tids = [bundle.trial_ids[i] for i in np.where(test_mask)[0]]
        healthy_train_tids = [
            bundle.trial_ids[i]
            for i in np.where(train_mask & (cohorts == "Healthy"))[0]
        ]
        if len(healthy_train_tids) < MIN_HEALTHY_TRAIN_TRIALS:
            logger.warning("Skipping fold {} — insufficient Healthy train trials", pid)
            continue

        train_scores, test_scores = build_fold_trial_scores(
            bundle,
            train_tids,
            test_tids,
            healthy_train_tids,
            config,
            device=device,
            random_state=rs,
        )
        y_train = y_true[train_mask]
        healthy_on_train = np.array([tid in healthy_train_tids for tid in train_tids], dtype=bool)

        for method in ALL_METHODS:
            thresh, thresh_src = fit_anomaly_threshold(
                train_scores[method],
                config,
                healthy_train_mask=healthy_on_train,
                y_train=y_train,
            )
            if not fold_threshold_sources:
                fold_threshold_sources.append(thresh_src)
            fold_thresholds[method].append(float(thresh))
            for tid, score in zip(test_tids, test_scores[method], strict=True):
                idx = trial_to_idx[tid]
                oof_scores[method][idx] = score
                oof_preds[method][idx] = float(score >= thresh)

    rows: list[dict[str, Any]] = []
    for method in ALL_METHODS:
        th_list = fold_thresholds[method]
        rows.append(
            _score_block(
                y_true,
                oof_scores[method],
                oof_preds[method],
                method=method,
                fold_threshold_mean=float(np.mean(th_list)) if th_list else float("nan"),
                fold_threshold_std=float(np.std(th_list)) if len(th_list) > 1 else float("nan"),
                n_threshold_folds=len(th_list),
                feature_selection_protocol=PROTOCOL_BILSTM_AE_LOSO,
                threshold_source=fold_threshold_sources[0] if fold_threshold_sources else "loso_healthy_train_percentile",
            )
        )

    metrics_df = pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "bilstm_ae_anomaly_metrics.csv"
    metrics_df.to_csv(out_path, index=False)
    logger.info("BiLSTM-AE ensemble metrics → {}", out_path)

    ens = metrics_df[metrics_df["method"] == METHOD_ENSEMBLE]
    registry = {
        "primary_endpoint": ENDPOINT_BILSTM_AE_ENSEMBLE,
        "manuscript_guidance": (
            "Primary endpoint: strict LOSO BiLSTM-AE reconstruction + latent "
            "Isolation Forest + latent One-Class SVM weighted ensemble. "
            "Trained on Healthy gait only; 4-sensor HE+LB+LF+RF input."
        ),
        "registered_endpoints": {},
    }
    if not ens.empty:
        r = ens.iloc[0]
        registry["registered_endpoints"][ENDPOINT_BILSTM_AE_ENSEMBLE] = {
            "model": METHOD_ENSEMBLE,
            "auc": float(r["auc"]) if pd.notna(r["auc"]) else None,
            "auc_pr": float(r["auc_pr"]) if pd.notna(r.get("auc_pr")) else None,
            "metric_source": "loso_trial_oof_bilstm_ae",
            "feature_selection_protocol": PROTOCOL_BILSTM_AE_LOSO,
            "methods": list(ALL_METHODS),
            "threshold_youden": float(r["threshold_youden"])
            if pd.notna(r.get("threshold_youden"))
            else None,
        }
    write_bilstm_ae_primary_artifacts(metrics_dir, registry)

    oof_df = pd.DataFrame(
        {
            "trial_id": bundle.trial_ids,
            "participant_id": groups,
            "cohort": cohorts,
            "eval_non_healthy": y_true,
        }
    )
    for method in ALL_METHODS:
        oof_df[f"{method}_score"] = oof_scores[method]
        oof_df[f"{method}_pred"] = oof_preds[method]
    oof_df.to_csv(metrics_dir / "bilstm_ae_loso_oof_scores.csv", index=False)

    from src.evaluation.loso_oof_scores import build_oof_export_frame, save_model_oof

    ens_scores = oof_scores[METHOD_ENSEMBLE]
    save_model_oof(
        metrics_dir,
        ENDPOINT_BILSTM_AE_ENSEMBLE,
        build_oof_export_frame(
            bundle.trial_ids,
            groups,
            y_true,
            ens_scores,
            cohorts=cohorts,
        ),
    )

    gain_path = metrics_dir / "bilstm_ae_ensemble_gain.json"
    if not ens.empty and not metrics_df.empty:
        base_aucs = {
            str(row["method"]): float(row["auc"])
            for _, row in metrics_df.iterrows()
            if str(row["method"]) != METHOD_ENSEMBLE and pd.notna(row["auc"])
        }
        ens_auc = float(ens.iloc[0]["auc"]) if pd.notna(ens.iloc[0]["auc"]) else None
        gain_path.write_text(
            json.dumps(
                {
                    "ensemble_auc": ens_auc,
                    "per_method_auc": base_aucs,
                    "gain_vs_best_single": (
                        ens_auc - max(base_aucs.values()) if ens_auc and base_aucs else None
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    from src.evaluation.threshold_validation import run_threshold_validation

    run_threshold_validation(config)

    return metrics_df
