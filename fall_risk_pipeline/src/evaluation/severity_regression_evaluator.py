"""
Navita 2025-style severity regression — MAE, MSE, R² on BiLSTM-AE latent head.

Per LOSO fold: train Healthy-only AE → pooled trial latents → MLP regression head
fit on train-fold trials → OOF severity predictions on held-out participant.

Targets (no per-participant UPDRS in Voisard ingest):
  - ``ordinal_severity`` — HS=0, ortho=1, neuro=2
  - ``fall_probability_pct`` — cohort reference fall-risk % (continuous UPDRS proxy)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dataset.severity_targets import (
    attach_severity_targets,
    cohort_in_scope,
    resolve_regression_target,
)
from src.dataset.subject_split import assert_loso_fold_disjoint
from src.models.bilstm_ae_scoring import build_fold_trial_latents, load_voisard_trial_windows
from src.models.latent_severity_regressor import predict_latent_regression, train_latent_regression_head
from src.utils.reproducibility import get_pipeline_seed
from src.utils.progress import progress_bar

MODEL_LATENT_HEAD = "bilstm_ae_latent_regressor"
MIN_HEALTHY_TRAIN_TRIALS = 3
VAL_FRACTION = 0.15


def compute_severity_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    n = int(mask.sum())
    if n < 2:
        return {"mae": float("nan"), "mse": float("nan"), "r2": float("nan"), "n": n}
    yt = y_true[mask]
    yp = y_pred[mask]
    return {
        "mae": float(mean_absolute_error(yt, yp)),
        "mse": float(mean_squared_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "n": n,
    }


def _trial_target_series(meta: pd.DataFrame, target_name: str, config: dict) -> pd.Series:
    enriched = attach_severity_targets(meta, config)
    return enriched.apply(lambda row: resolve_regression_target(row, target_name, config), axis=1)


def _split_train_val(
    Z: np.ndarray,
    y: np.ndarray,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    n = len(Z)
    if n < 8:
        return Z, y, None, None
    rng = np.random.default_rng(random_state)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = max(1, int(n * VAL_FRACTION))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    if len(tr_idx) == 0:
        return Z, y, None, None
    return Z[tr_idx], y[tr_idx], Z[val_idx], y[val_idx]


def _valid_latent_rows(Z: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_ok = np.isfinite(y) & np.all(np.isfinite(Z), axis=1)
    return Z[row_ok], y[row_ok]


def run_severity_regression_evaluation(config: dict) -> pd.DataFrame:
    scfg = config.get("severity_regression") or {}
    if not scfg.get("enabled", True):
        logger.info("Severity regression disabled in config")
        return pd.DataFrame()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs = get_pipeline_seed(config)
    bundle = load_voisard_trial_windows(config, require_all_sensors=True)

    processed = Path(config["paths"]["processed_data"])
    meta = pd.read_csv(processed / "trial_metadata.csv")
    meta = meta[~meta["trial_id"].astype(str).str.startswith("daphnet_")].reset_index(drop=True)
    meta_by_tid = meta.set_index("trial_id")

    targets = list(scfg.get("targets") or ["ordinal_severity", "fall_probability_pct"])
    scopes = list(scfg.get("cohort_scopes") or ["all_8", "neuro_ortho"])
    navita_ref = scfg.get("navita_reference") or {}

    trial_to_idx = {tid: i for i, tid in enumerate(bundle.trial_ids)}
    groups = bundle.participant_ids
    cohorts = bundle.cohorts
    n = len(bundle.trial_ids)

    target_arrays: dict[str, np.ndarray] = {}
    for tname in targets:
        series = _trial_target_series(meta_by_tid.loc[bundle.trial_ids], tname, config)
        target_arrays[tname] = series.astype(float).values

    oof_preds: dict[str, np.ndarray] = {t: np.full(n, np.nan) for t in targets}
    unique_pids = np.unique(groups)

    logger.info(
        "Severity regression LOSO over {} participants, {} trials (latent MLP head)",
        len(unique_pids),
        n,
    )

    for pid in progress_bar(unique_pids, desc="Severity regression LOSO", unit="participant"):
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

        lat_tr, lat_te = build_fold_trial_latents(
            bundle,
            train_tids,
            test_tids,
            healthy_train_tids,
            config,
            device=device,
            random_state=rs,
        )

        for tname in targets:
            y_all = target_arrays[tname]
            y_tr = y_all[train_mask]
            y_te = y_all[test_mask]

            Z_fit, y_fit = _valid_latent_rows(lat_tr, y_tr)
            if len(Z_fit) < 3:
                continue

            Z_tr, y_tr_fit, Z_val, y_val = _split_train_val(Z_fit, y_fit, random_state=rs)
            head = train_latent_regression_head(
                Z_tr,
                y_tr_fit,
                config,
                Z_val=Z_val,
                y_val=y_val,
            )

            te_ok = np.isfinite(y_te) & np.all(np.isfinite(lat_te), axis=1)
            if not te_ok.any():
                continue
            pred_te = predict_latent_regression(head, lat_te[te_ok], device=device)
            for tid, pred in zip(
                np.array(test_tids)[te_ok],
                pred_te,
                strict=True,
            ):
                oof_preds[tname][trial_to_idx[tid]] = float(pred)

    rows: list[dict[str, Any]] = []
    for tname in targets:
        y_true = target_arrays[tname]
        y_pred = oof_preds[tname]
        for scope in scopes:
            scope_mask = np.array([cohort_in_scope(c, scope) for c in cohorts])
            valid = scope_mask & np.isfinite(y_true) & np.isfinite(y_pred)
            metrics = compute_severity_regression_metrics(y_true[valid], y_pred[valid])
            navita_mae = navita_ref.get("mae")
            beats_navita = None
            if navita_mae is not None and pd.notna(navita_mae) and pd.notna(metrics["mae"]):
                beats_navita = bool(float(metrics["mae"]) < float(navita_mae))
            rows.append(
                {
                    "model": MODEL_LATENT_HEAD,
                    "target": tname,
                    "cohort_scope": scope,
                    "mae": metrics["mae"],
                    "mse": metrics["mse"],
                    "r2": metrics["r2"],
                    "n_trials": metrics["n"],
                    "navita_reference_mae": navita_mae,
                    "beats_navita_mae": beats_navita,
                    "protocol": "loso_trial_oof_latent_regression",
                }
            )

    metrics_df = pd.DataFrame(rows)
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_csv = metrics_dir / "severity_regression_metrics.csv"
    metrics_df.to_csv(out_csv, index=False)
    logger.info("Severity regression metrics → {}", out_csv)

    oof_df = pd.DataFrame(
        {
            "trial_id": bundle.trial_ids,
            "participant_id": groups,
            "cohort": cohorts,
        }
    )
    for tname in targets:
        oof_df[f"severity_true_{tname}"] = target_arrays[tname]
        oof_df[f"severity_pred_{tname}"] = oof_preds[tname]
    oof_df.to_csv(metrics_dir / "severity_regression_oof_predictions.csv", index=False)

    comparison = {
        "gaitguard_model": MODEL_LATENT_HEAD,
        "navita_reference": navita_ref,
        "metrics_by_target_scope": rows,
        "manuscript_guidance": (
            "Navita 2025 benchmark: AdaBoost/Gradient Boost on gait features → UPDRS "
            "(MAE, MSE, R²). GaitGuard: MLP regression head on BiLSTM-AE pooled latent "
            "activations; ordinal cohort severity when UPDRS unavailable."
        ),
    }
    comp_path = metrics_dir / "severity_regression_navita_comparison.json"
    comp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    md_lines = [
        "# Severity regression — GaitGuard vs Navita 2025",
        "",
        "| Model | Target | Cohort scope | MAE ↓ | MSE ↓ | R² ↑ | n | vs Navita MAE |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in metrics_df.iterrows():
        mae_s = f"{row['mae']:.4f}" if pd.notna(row["mae"]) else "—"
        mse_s = f"{row['mse']:.4f}" if pd.notna(row["mse"]) else "—"
        r2_s = f"{row['r2']:.4f}" if pd.notna(row["r2"]) else "—"
        vs = "—"
        if row.get("beats_navita_mae") is True:
            vs = "beat"
        elif row.get("beats_navita_mae") is False:
            vs = "below"
        md_lines.append(
            f"| {row['model']} | {row['target']} | {row['cohort_scope']} | "
            f"{mae_s} | {mse_s} | {r2_s} | {int(row['n_trials'])} | {vs} |"
        )
    if navita_ref.get("mae") is not None:
        md_lines.extend(
            [
                "",
                f"Navita 2025 reference MAE: **{navita_ref['mae']}** "
                f"(set in `severity_regression.navita_reference`).",
            ]
        )
    (metrics_dir / "severity_regression_navita_comparison.md").write_text(
        "\n".join(md_lines) + "\n",
        encoding="utf-8",
    )

    return metrics_df
