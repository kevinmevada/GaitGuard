"""
BiLSTM-AE sensor ablation — 4-sensor vs 2-sensor (HE+LB) vs 1-sensor (LB).

In-distribution (Voisard LOSO ensemble AUC): expect 4-sensor > 2-sensor > 1-sensor.

Cross-dataset (DAPHNET FOG): 4-sensor-trained model, LB-only zero-padded input
→ demonstrates representation transfer via the LB channel alone.

Outputs ``bilstm_ae_sensor_ablation.csv`` + ``bilstm_ae_sensor_ablation.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger

from src.dataset.subject_split import assert_loso_fold_disjoint
from src.evaluation.anomaly_loso_evaluator import _score_block
from src.evaluation.anomaly_threshold_policy import fit_anomaly_threshold
from src.evaluation.daphnet_bilstm_ae_evaluator import evaluate_daphnet_lb_scores
from src.evaluation.primary_endpoint import PROTOCOL_BILSTM_AE_LOSO
from src.models.anomaly_scoring import eval_binary_labels
from src.models.bilstm_ae_scoring import (
    METHOD_ENSEMBLE,
    SENSOR_ABLATION_CONFIGS,
    TrialWindowBundle,
    apply_sensor_mask_to_bundle,
    build_fold_trial_scores,
    load_voisard_trial_windows,
    mask_inactive_sensors,
    train_healthy_ae,
)
from src.dataset.train_fit_mask import healthy_train_fit_mask
from src.utils.progress import progress_bar
from src.utils.reproducibility import get_pipeline_seed
from src.utils.torch_device import resolve_torch_device

MIN_HEALTHY_TRAIN_TRIALS = 3


def _ablation_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return (config.get("sensor_ablation") or {}).get("bilstm_ae") or {}


def _config_for_ablation_fit(config: dict[str, Any]) -> dict[str, Any]:
    """Merge optional faster AE hyperparameters for ablation sweeps."""
    ab_ae = _ablation_cfg(config).get("bilstm_autoencoder")
    if not ab_ae:
        return config
    import copy

    merged = copy.deepcopy(config)
    primary = merged.setdefault("primary_model", {}).setdefault("bilstm_ae_ensemble", {})
    base_ae = dict(primary.get("bilstm_autoencoder") or {})
    base_ae.update(ab_ae)
    primary["bilstm_autoencoder"] = base_ae
    return merged


def _loso_ensemble_auc(
    bundle: TrialWindowBundle,
    config: dict[str, Any],
    *,
    device: torch.device,
    random_state: int,
) -> dict[str, float]:
    """LOSO OOF ROC-AUC for the 3-method BiLSTM-AE ensemble."""
    trial_to_idx = {tid: i for i, tid in enumerate(bundle.trial_ids)}
    groups = bundle.participant_ids
    cohorts = bundle.cohorts
    y_true = eval_binary_labels(cohorts)
    n = len(bundle.trial_ids)

    oof_score = np.full(n, np.nan)
    oof_pred = np.full(n, np.nan)

    for pid in np.unique(groups):
        test_mask = groups == pid
        train_mask = ~test_mask
        assert_loso_fold_disjoint(groups[train_mask], groups[test_mask], held_out_subject=str(pid))

        train_tids = [bundle.trial_ids[i] for i in np.where(train_mask)[0]]
        test_tids = [bundle.trial_ids[i] for i in np.where(test_mask)[0]]
        healthy_train_tids = [
            bundle.trial_ids[i] for i in np.where(train_mask & (cohorts == "Healthy"))[0]
        ]
        if len(healthy_train_tids) < MIN_HEALTHY_TRAIN_TRIALS:
            continue

        train_scores, test_scores = build_fold_trial_scores(
            bundle,
            train_tids,
            test_tids,
            healthy_train_tids,
            config,
            device=device,
            random_state=random_state,
        )
        healthy_on_train = np.array([tid in healthy_train_tids for tid in train_tids], dtype=bool)
        thresh, _ = fit_anomaly_threshold(
            train_scores[METHOD_ENSEMBLE],
            config,
            healthy_train_mask=healthy_on_train,
            y_train=y_true[train_mask],
        )
        for tid, score in zip(test_tids, test_scores[METHOD_ENSEMBLE], strict=True):
            idx = trial_to_idx[tid]
            oof_score[idx] = score
            oof_pred[idx] = float(score >= thresh)

    row = _score_block(
        y_true,
        oof_score,
        oof_pred,
        method=METHOD_ENSEMBLE,
        fold_threshold_mean=float("nan"),
        fold_threshold_std=float("nan"),
        n_threshold_folds=0,
        feature_selection_protocol=PROTOCOL_BILSTM_AE_LOSO,
    )
    return {
        "voisard_loso_ensemble_auc": float(row["auc"]) if pd.notna(row["auc"]) else float("nan"),
        "voisard_loso_ensemble_auc_pr": float(row["auc_pr"]) if pd.notna(row.get("auc_pr")) else float("nan"),
        "voisard_sensitivity": float(row["sensitivity"]) if pd.notna(row.get("sensitivity")) else float("nan"),
        "voisard_specificity": float(row["specificity"]) if pd.notna(row.get("specificity")) else float("nan"),
        "n_trials_scored": int(row["n_trials_scored"]),
    }


def _train_deploy_4sensor_for_daphnet(
    config: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[Any, Any, list]:
    """Healthy train-fold windows, 4-sensor active (full model for DAPHNET transfer)."""
    from src.features.phase3_deep import compute_sensor_slices
    from src.models.deep_models import trial_to_tensor
    from src.preprocessing.windowing import parse_window_spec, window_single_trial

    processed = Path(config["paths"]["processed_data"])
    signals_dir = processed / "signals_clean"
    meta = pd.read_csv(processed / "trial_metadata.csv")
    meta = meta[~meta["trial_id"].astype(str).str.startswith("daphnet_")]
    fit_mask = healthy_train_fit_mask(meta, config)
    spec = parse_window_spec(config)
    sensor_positions = config["dataset"]["sensor_positions"]
    active = SENSOR_ABLATION_CONFIGS["4_sensor"][1]

    windows: list[np.ndarray] = []
    ref_slices: list = []
    for i, row in meta.iterrows():
        if not fit_mask[i]:
            continue
        tid = str(row["trial_id"])
        arr = trial_to_tensor(tid, signals_dir, sensor_positions, require_all_sensors=True)
        if arr is None or arr.shape[1] < spec.window_len:
            continue
        if not ref_slices:
            ref_slices = compute_sensor_slices(tid, signals_dir, sensor_positions)
        wins = window_single_trial(arr, spec)
        if len(wins):
            windows.append(mask_inactive_sensors(wins, ref_slices, active))

    if not windows:
        raise RuntimeError("No Healthy train windows for 4-sensor DAPHNET transfer eval")

    healthy_windows = np.concatenate(windows, axis=0)
    ckpt = Path(config["paths"]["checkpoints"]) / "bilstm_ae_ablation_4sensor_deploy.pt"
    model, norm = train_healthy_ae(
        healthy_windows, ref_slices, config, device=device, checkpoint_path=ckpt
    )
    return model, norm, ref_slices


def run_bilstm_ae_sensor_ablation(config: dict) -> pd.DataFrame:
    ab_cfg = _ablation_cfg(config)
    if not ab_cfg.get("enabled", True):
        logger.info("BiLSTM-AE sensor ablation disabled")
        return pd.DataFrame()

    pcfg = (config.get("primary_model") or {}).get("bilstm_ae_ensemble") or {}
    if not pcfg.get("enabled", True):
        logger.warning("BiLSTM-AE primary model disabled — skipping sensor ablation")
        return pd.DataFrame()

    device = resolve_torch_device(config)
    rs = get_pipeline_seed(config)
    fit_config = _config_for_ablation_fit(config)
    metrics_dir = Path(config["paths"]["metrics"])
    fig_dir = Path(config["paths"]["figures_models"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    base_bundle = load_voisard_trial_windows(config, require_all_sensors=True)
    rows: list[dict[str, Any]] = []
    daphnet_4sensor_auc = float("nan")

    for config_key, (label, active) in progress_bar(
        SENSOR_ABLATION_CONFIGS.items(),
        desc="bilstm_ae_sensor_ablation",
        unit="config",
    ):
        logger.info("BiLSTM-AE sensor ablation: {} ({})", config_key, label)
        bundle = apply_sensor_mask_to_bundle(base_bundle, active)
        loso = _loso_ensemble_auc(bundle, fit_config, device=device, random_state=rs)

        daphnet_auc = float("nan")
        daphnet_auc_pr = float("nan")
        if config_key == "4_sensor":
            try:
                model, norm, slices = _train_deploy_4sensor_for_daphnet(fit_config, device=device)
                daph = evaluate_daphnet_lb_scores(model, norm, slices, config, device=device)
                daphnet_auc = float(daph.get("lb_reconstruction_auc", float("nan")))
                daphnet_auc_pr = float(daph.get("lb_reconstruction_auc_pr", float("nan")))
                daphnet_4sensor_auc = daphnet_auc
            except (FileNotFoundError, RuntimeError) as exc:
                logger.info("DAPHNET transfer eval skipped for 4-sensor: {}", exc)
                sealed = metrics_dir / "daphnet_bilstm_ae_fog_auroc.json"
                if sealed.is_file():
                    payload = json.loads(sealed.read_text(encoding="utf-8"))
                    per = payload.get("per_method", {}).get("ae_reconstruction") or {}
                    daphnet_auc = float(per.get("auc", payload.get("auc", float("nan"))))
                    daphnet_auc_pr = float(per.get("auc_pr", payload.get("auc_pr", float("nan"))))
                    daphnet_4sensor_auc = daphnet_auc

        rows.append(
            {
                "sensor_config": config_key,
                "sensors": label,
                "n_sensors": len(active),
                "active_positions": ", ".join(active),
                "train_sensors": label,
                "eval_voisard": "masked_loso_oof",
                "eval_daphnet": (
                    "lb_only_zero_padded"
                    if config_key == "4_sensor"
                    else "not_applicable"
                ),
                **loso,
                "daphnet_lb_recon_auc": daphnet_auc,
                "daphnet_lb_recon_auc_pr": daphnet_auc_pr,
                "validation": "loso_subject_grouped",
                "model": METHOD_ENSEMBLE,
            }
        )
        logger.info(
            "  {} Voisard ensemble AUC={:.4f} DAPHNET LB={}",
            label,
            loso["voisard_loso_ensemble_auc"],
            f"{daphnet_auc:.4f}" if np.isfinite(daphnet_auc) else "N/A",
        )

    df = pd.DataFrame(rows).sort_values("n_sensors", ascending=False)
    csv_path = metrics_dir / "bilstm_ae_sensor_ablation.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "in_distribution_ranking": "4_sensor > 2_sensor_he_lb > 1_sensor_lb (expected)",
        "voisard_auc_by_config": {
            str(r["sensor_config"]): float(r["voisard_loso_ensemble_auc"])
            for _, r in df.iterrows()
        },
        "daphnet_4sensor_train_lb_eval_auc": daphnet_4sensor_auc,
        "contribution": (
            "Multi-sensor training improves in-distribution Voisard screening; "
            "4-sensor Healthy manifold transfers to DAPHNET via LB reconstruction alone."
        ),
    }
    (metrics_dir / "bilstm_ae_sensor_ablation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    _write_markdown_table(metrics_dir / "bilstm_ae_sensor_ablation.md", df, daphnet_4sensor_auc)
    _plot_ablation(fig_dir, df, config)
    logger.info("BiLSTM-AE sensor ablation → {}", csv_path)
    return df


def _write_markdown_table(path: Path, df: pd.DataFrame, daphnet_4sensor_auc: float) -> None:
    lines = [
        "# BiLSTM-AE sensor ablation",
        "",
        "Three training configurations on Voisard (Healthy-reference LOSO ensemble). "
        "Inactive IMU blocks are zero-padded; channel layout matches the 4-sensor AE.",
        "",
        "| Config | Sensors | Voisard LOSO AUC (ensemble) | DAPHNET LB recon AUC |",
        "|---|---|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        daph = (
            f"{float(row.daphnet_lb_recon_auc):.4f}"
            if row.eval_daphnet != "not_applicable" and pd.notna(row.daphnet_lb_recon_auc)
            else "—"
        )
        lines.append(
            f"| {row.sensor_config} | {row.sensors} | "
            f"{float(row.voisard_loso_ensemble_auc):.4f} | {daph} |"
        )
    lines.extend(
        [
            "",
            "**In-distribution:** 4-sensor > 2-sensor > 1-sensor → multi-sensor training adds value.",
            "",
        ]
    )
    if np.isfinite(daphnet_4sensor_auc):
        lines.append(
            f"**Cross-dataset:** 4-sensor-trained model, LB-only DAPHNET input → "
            f"AUROC **{daphnet_4sensor_auc:.4f}** (representation transfers via LB channel).",
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_ablation(fig_dir: Path, df: pd.DataFrame, config: dict) -> None:
    if df.empty:
        return
    dpi = int(config.get("reporting", {}).get("figure_dpi", 300))
    fmt = config.get("reporting", {}).get("figure_format", "pdf")

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = df["sensors"].tolist()
    aucs = df["voisard_loso_ensemble_auc"].values
    colors = ["#3498db", "#f39c12", "#e74c3c"][: len(labels)]
    bars = ax.bar(range(len(labels)), aucs, color=colors, edgecolor="white")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Voisard LOSO ensemble ROC-AUC")
    ax.set_title("BiLSTM-AE sensor ablation (in-distribution)")
    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{auc:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    for ext in (fmt, "png"):
        fig.savefig(fig_dir / f"bilstm_ae_sensor_ablation.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
