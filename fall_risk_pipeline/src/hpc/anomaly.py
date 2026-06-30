"""Sharded BiLSTM-AE LOSO fold execution and OOF reduction."""

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
from src.evaluation.bilstm_ae_loso_evaluator import (
    ALL_METHODS,
    MIN_HEALTHY_TRAIN_TRIALS,
    run_bilstm_ae_loso_evaluation,
)
from src.evaluation.primary_endpoint import (
    ENDPOINT_BILSTM_AE_ENSEMBLE,
    PROTOCOL_BILSTM_AE_LOSO,
    write_bilstm_ae_primary_artifacts,
)
from src.hpc.paths import anomaly_fold_path, oof_root
from src.models.anomaly_scoring import eval_binary_labels
from src.models.bilstm_ae_scoring import (
    METHOD_ENSEMBLE,
    build_fold_trial_scores,
    load_voisard_trial_windows,
)
from src.utils.reproducibility import get_pipeline_seed


def run_anomaly_fold(config: dict, held_out_participant_id: str) -> Path | None:
    """
    Run one LOSO fold for the primary BiLSTM-AE ensemble; write fold OOF shard.
    """
    pcfg = (config.get("primary_model") or {}).get("bilstm_ae_ensemble") or {}
    if not pcfg.get("enabled", True):
        logger.info("BiLSTM-AE disabled — skipping fold")
        return None

    pid = str(held_out_participant_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs = get_pipeline_seed(config)
    bundle = load_voisard_trial_windows(config, require_all_sensors=True)
    groups = bundle.participant_ids
    cohorts = bundle.cohorts
    y_true = eval_binary_labels(cohorts)

    test_mask = groups == pid
    if not np.any(test_mask):
        logger.warning("Fold {} not in bundle — skip", pid)
        return None
    train_mask = ~test_mask
    assert_loso_fold_disjoint(groups[train_mask], groups[test_mask], held_out_subject=pid)

    train_tids = [bundle.trial_ids[i] for i in np.where(train_mask)[0]]
    test_tids = [bundle.trial_ids[i] for i in np.where(test_mask)[0]]
    healthy_train_tids = [
        bundle.trial_ids[i] for i in np.where(train_mask & (cohorts == "Healthy"))[0]
    ]
    if len(healthy_train_tids) < MIN_HEALTHY_TRAIN_TRIALS:
        logger.warning("Fold {} — insufficient Healthy train trials", pid)
        return None

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
    trial_to_idx = {tid: i for i, tid in enumerate(bundle.trial_ids)}

    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    threshold_source = "loso_healthy_train_percentile"
    for method in ALL_METHODS:
        thresh, thresh_src = fit_anomaly_threshold(
            train_scores[method],
            config,
            healthy_train_mask=healthy_on_train,
            y_train=y_train,
        )
        thresholds[method] = float(thresh)
        threshold_source = thresh_src
        for tid, score in zip(test_tids, test_scores[method], strict=True):
            idx = trial_to_idx[tid]
            rows.append(
                {
                    "held_out_participant_id": pid,
                    "trial_id": tid,
                    "participant_id": groups[idx],
                    "cohort": cohorts[idx],
                    "eval_non_healthy": int(y_true[idx]),
                    "method": method,
                    "score": float(score),
                    "pred": float(score >= thresh),
                    "threshold": float(thresh),
                    "threshold_source": thresh_src,
                }
            )

    out = anomaly_fold_path(config, pid)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    sidecar = Path(str(out).replace(".parquet", ".thresholds.json"))
    sidecar.write_text(
        json.dumps({"thresholds": thresholds, "threshold_source": threshold_source}, indent=2),
        encoding="utf-8",
    )
    logger.info("Anomaly fold shard → {} ({} test trials)", out, len(test_tids))
    return out


def reduce_anomaly_folds(config: dict) -> pd.DataFrame:
    """Stitch fold shards and write final BiLSTM-AE metrics + OOF CSV."""
    fold_dir = oof_root(config) / "anomaly"
    if not fold_dir.is_dir():
        raise FileNotFoundError(f"No anomaly fold shards in {fold_dir}")

    bundle = load_voisard_trial_windows(config, require_all_sensors=True)
    n = len(bundle.trial_ids)
    trial_to_idx = {tid: i for i, tid in enumerate(bundle.trial_ids)}
    groups = bundle.participant_ids
    cohorts = bundle.cohorts
    y_true = eval_binary_labels(cohorts)

    oof_scores: dict[str, np.ndarray] = {m: np.full(n, np.nan) for m in ALL_METHODS}
    oof_preds: dict[str, np.ndarray] = {m: np.full(n, np.nan) for m in ALL_METHODS}
    fold_thresholds: dict[str, list[float]] = {m: [] for m in ALL_METHODS}
    fold_threshold_sources: list[str] = []

    for fold_file in sorted(fold_dir.glob("fold_*.parquet")):
        df = pd.read_parquet(fold_file)
        if df.empty:
            continue
        sidecar = Path(str(fold_file).replace(".parquet", ".thresholds.json"))
        if sidecar.is_file():
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            fold_threshold_sources.append(str(meta.get("threshold_source", "")))
            for method, thresh in (meta.get("thresholds") or {}).items():
                if method in fold_thresholds:
                    fold_thresholds[method].append(float(thresh))
        for row in df.itertuples(index=False):
            method = str(row.method)
            tid = str(row.trial_id)
            idx = trial_to_idx.get(tid)
            if idx is None or method not in oof_scores:
                continue
            oof_scores[method][idx] = float(row.score)
            oof_preds[method][idx] = float(row.pred)

    rows: list[dict[str, Any]] = []
    thresh_src = (
        fold_threshold_sources[0] if fold_threshold_sources else "loso_healthy_train_percentile"
    )
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
                threshold_source=thresh_src,
            )
        )

    metrics_df = pd.DataFrame(rows).sort_values("auc", ascending=False, na_position="last")
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / "bilstm_ae_anomaly_metrics.csv"
    metrics_df.to_csv(out_path, index=False)
    logger.info("Reduced BiLSTM-AE metrics → {}", out_path)

    ens = metrics_df[metrics_df["method"] == METHOD_ENSEMBLE]
    registry = {
        "primary_endpoint": ENDPOINT_BILSTM_AE_ENSEMBLE,
        "manuscript_guidance": (
            "Primary endpoint: strict LOSO BiLSTM-AE reconstruction + latent "
            "Isolation Forest + latent One-Class SVM weighted ensemble."
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

    save_model_oof(
        metrics_dir,
        ENDPOINT_BILSTM_AE_ENSEMBLE,
        build_oof_export_frame(
            bundle.trial_ids,
            groups,
            y_true,
            oof_scores[METHOD_ENSEMBLE],
            cohorts=cohorts,
        ),
    )

    from src.evaluation.threshold_validation import run_threshold_validation

    run_threshold_validation(config)
    return metrics_df


def run_anomaly_stage(config: dict, *, sharded: bool = False) -> pd.DataFrame:
    """Dispatch full or sharded anomaly (BiLSTM-AE primary) evaluation."""
    if sharded:
        raise RuntimeError("Use reduce_anomaly_folds after submitting fold jobs")
    return run_bilstm_ae_loso_evaluation(config)
