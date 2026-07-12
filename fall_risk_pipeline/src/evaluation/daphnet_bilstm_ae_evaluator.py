"""
DAPHNET cross-dataset eval — BiLSTM-AE LB reconstruction + 3-method ensemble.

Trains on Voisard Healthy train-fold windows (4-sensor AE). Scores DAPHNET with
LB-channel-only input (zero-padded HE/LF/RF). Primary score: LB reconstruction error.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from src.dataset.train_fit_mask import healthy_train_fit_mask
from src.evaluation.metrics_ci import grouped_bootstrap_binary_auc_ci
from src.ingestion.daphnet_label_mapping import FOG_LABEL_MANIFEST, fog_labels_path, load_fog_labels_npz
from src.models.bilstm_ae_scoring import (
    METHOD_AE_RECON,
    METHOD_ENSEMBLE,
    METHOD_IF_LATENT,
    METHOD_OCSVM_LATENT,
    combine_ensemble_scores,
    fit_latent_one_class_models,
    lb_slice_from_slices,
    score_latent_one_class,
    score_windows,
    train_healthy_ae,
)
from src.models.deep_models import CHANNEL_ORDER, trial_to_tensor
from src.preprocessing.windowing import parse_window_spec, window_single_trial
from src.utils.reproducibility import get_pipeline_seed
from src.utils.torch_device import resolve_torch_device

SEALED_OUTPUT_NAME = "daphnet_bilstm_ae_fog_auroc.json"


class DaphnetBilstmAeEvalError(RuntimeError):
    pass


def _collect_healthy_train_windows(config: dict[str, Any]) -> tuple[np.ndarray, list]:
    processed = Path(config["paths"]["processed_data"])
    signals_dir = processed / "signals_clean"
    meta = pd.read_csv(processed / "trial_metadata.csv")
    meta = meta[~meta["trial_id"].astype(str).str.startswith("daphnet_")]
    fit_mask = healthy_train_fit_mask(meta, config)
    spec = parse_window_spec(config)
    sensor_positions = config["dataset"]["sensor_positions"]

    from src.features.phase3_deep import compute_sensor_slices

    windows: list[np.ndarray] = []
    ref_slices = []
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
            windows.append(wins)
    if not windows:
        raise DaphnetBilstmAeEvalError("No Healthy train-fold windows for DAPHNET BiLSTM-AE eval")
    return np.concatenate(windows, axis=0), ref_slices


def _daphnet_lb_windows(
    processed_dir: Path,
    subject_id: str,
    *,
    n_channels: int,
    lb_slice,
    window_len: int,
    overlap: float,
) -> np.ndarray:
    from src.preprocessing.windowing import WindowSpec

    path = processed_dir / "signals" / f"daphnet_{subject_id}_lower_back.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    cols = [c for c in CHANNEL_ORDER if c in df.columns]
    if not cols:
        cols = [c for c in ("acc_x", "acc_y", "acc_z") if c in df.columns]
    lb = df[cols].values.T.astype(np.float32)
    full = np.zeros((n_channels, lb.shape[1]), dtype=np.float32)
    n_lb = lb_slice.end - lb_slice.start
    full[lb_slice.start : lb_slice.start + min(n_lb, lb.shape[0]), :] = lb[:n_lb]
    spec = WindowSpec(window_len=window_len, overlap=overlap, fs_hz=100.0)
    return window_single_trial(full, spec)


def evaluate_daphnet_lb_scores(
    model,
    norm,
    sensor_slices: list,
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, float]:
    """Score DAPHNET with LB-only zero-padded windows; return LB reconstruction AUROC."""
    spec = parse_window_spec(config)
    overlap = float(config["deep_learning"]["overlap"])
    lb_slice = lb_slice_from_slices(sensor_slices)
    if lb_slice is None:
        raise DaphnetBilstmAeEvalError("No lower_back slice")

    sealed = (config.get("ingestion") or {}).get("daphnet", {}).get("sealed_fog_eval") or {}
    labels_path = Path(
        sealed.get("labels_path") or fog_labels_path(Path(config["paths"]["processed_data"]))
    )
    y_by_subject = load_fog_labels_npz(labels_path)
    processed = Path(config["paths"]["processed_data"])

    all_y: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for subject_id, y_true in sorted(y_by_subject.items()):
        try:
            wins = _daphnet_lb_windows(
                processed,
                subject_id,
                n_channels=model.n_channels,
                lb_slice=lb_slice,
                window_len=spec.window_len,
                overlap=overlap,
            )
        except FileNotFoundError:
            continue
        if len(wins) == 0:
            continue
        n = min(len(wins), len(y_true))
        wins, y_true = wins[:n], y_true[:n]
        recon, _ = score_windows(model, wins, norm, device=device, lb_slice=lb_slice)
        all_y.append(y_true)
        all_scores.append(recon)

    if not all_y:
        raise FileNotFoundError("No DAPHNET LB windows for transfer eval")

    y_cat = np.concatenate(all_y)
    sc = np.concatenate(all_scores)
    if len(np.unique(y_cat)) < 2:
        return {"lb_reconstruction_auc": float("nan"), "lb_reconstruction_auc_pr": float("nan")}

    return {
        "lb_reconstruction_auc": float(roc_auc_score(y_cat, sc)),
        "lb_reconstruction_auc_pr": float(average_precision_score(y_cat, sc)),
        "n_samples": int(len(y_cat)),
    }


def run_daphnet_bilstm_ae_fog_eval(config: dict, *, force: bool = False) -> dict[str, Any]:
    """Sealed DAPHNET FOG AUROC using BiLSTM-AE LB reconstruction + 3-method ensemble."""
    pcfg = (config.get("primary_model") or {}).get("bilstm_ae_ensemble") or {}
    sealed = (config.get("ingestion") or {}).get("daphnet", {}).get("sealed_fog_eval") or {}
    if not pcfg.get("enabled", True):
        return {}

    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / SEALED_OUTPUT_NAME
    if out_path.is_file() and not force and not sealed.get("allow_rerun", False):
        raise DaphnetBilstmAeEvalError(f"Sealed result exists: {out_path}")

    device = resolve_torch_device(config)
    spec = parse_window_spec(config)
    overlap = float(config["deep_learning"]["overlap"])

    healthy_windows, sensor_slices = _collect_healthy_train_windows(config)
    lb_slice = lb_slice_from_slices(sensor_slices)
    if lb_slice is None:
        raise DaphnetBilstmAeEvalError("No lower_back slice in sensor layout")

    ckpt = Path(config["paths"]["checkpoints"]) / "bilstm_ae_deploy.pt"
    model, norm = train_healthy_ae(
        healthy_windows, sensor_slices, config, device=device, checkpoint_path=ckpt
    )

    _, lat_fit = score_windows(model, healthy_windows, norm, device=device)
    oc = fit_latent_one_class_models(lat_fit, random_state=42)
    recon_fit, _ = score_windows(model, healthy_windows, norm, device=device, lb_slice=lb_slice)
    if_fit, svm_fit = score_latent_one_class(oc, lat_fit)
    ref = {
        METHOD_AE_RECON: recon_fit,
        METHOD_IF_LATENT: if_fit,
        METHOD_OCSVM_LATENT: svm_fit,
    }

    labels_path = Path(sealed.get("labels_path") or fog_labels_path(Path(config["paths"]["processed_data"])))
    y_by_subject = load_fog_labels_npz(labels_path)
    processed = Path(config["paths"]["processed_data"])

    all_y: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    scores_by_method: dict[str, list[np.ndarray]] = {m: [] for m in (*ref.keys(), METHOD_ENSEMBLE)}

    for subject_id, y_true in sorted(y_by_subject.items()):
        try:
            wins = _daphnet_lb_windows(
                processed,
                subject_id,
                n_channels=model.n_channels,
                lb_slice=lb_slice,
                window_len=spec.window_len,
                overlap=overlap,
            )
        except FileNotFoundError:
            continue
        if len(wins) == 0:
            continue
        n = min(len(wins), len(y_true))
        wins, y_true = wins[:n], y_true[:n]
        recon, lat = score_windows(model, wins, norm, device=device, lb_slice=lb_slice)
        if_s, svm_s = score_latent_one_class(oc, lat)
        methods = {
            METHOD_AE_RECON: recon,
            METHOD_IF_LATENT: if_s,
            METHOD_OCSVM_LATENT: svm_s,
        }
        ens = combine_ensemble_scores(methods, config, reference_scores=ref)
        methods[METHOD_ENSEMBLE] = ens

        all_y.append(y_true)
        all_groups.append(np.full(len(y_true), subject_id, dtype=object))
        for m, s in methods.items():
            scores_by_method[m].append(s)

    y_cat = np.concatenate(all_y)
    groups_cat = np.concatenate(all_groups)
    result: dict[str, Any] = {
        "endpoint": "daphnet_fog_bilstm_ae_auroc",
        "protocol": "sealed_lb_reconstruction",
        "train_reference": "voisard_healthy_train_fold_4sensor_ae",
        "eval_input": "daphnet_lb_zero_padded",
        "label_mapping": FOG_LABEL_MANIFEST["mapping"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "per_method": {},
    }
    for method, chunks in scores_by_method.items():
        if not chunks:
            continue
        sc = np.concatenate(chunks)
        if len(np.unique(y_cat)) >= 2:
            auc_full, ci_low, ci_high, ci_status = grouped_bootstrap_binary_auc_ci(
                y_cat, sc, groups_cat, seed=get_pipeline_seed(config)
            )
            result["per_method"][method] = {
                "auc": float(roc_auc_score(y_cat, sc)),
                "auc_pr": float(average_precision_score(y_cat, sc)),
                "auc_ci_low": ci_low,
                "auc_ci_high": ci_high,
                "auc_ci_method": ci_status,
            }
    primary = result["per_method"].get(METHOD_AE_RECON) or result["per_method"].get(METHOD_ENSEMBLE)
    if primary:
        result["auc"] = primary["auc"]
        result["auc_pr"] = primary["auc_pr"]
        result["auc_ci_low"] = primary["auc_ci_low"]
        result["auc_ci_high"] = primary["auc_ci_high"]
        result["auc_ci_method"] = primary["auc_ci_method"]
    result["n_samples"] = int(len(y_cat))
    result["n_fog_positive"] = int((y_cat == 1).sum())

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        metrics_dir / "daphnet_bilstm_ae_scores.npz",
        y_true=y_cat.astype(np.int8),
        **{f"{m}_scores": np.concatenate(chunks).astype(np.float32) for m, chunks in scores_by_method.items() if chunks},
    )
    logger.info("DAPHNET BiLSTM-AE FOG AUROC (LB recon) = {:.4f}", result.get("auc", float("nan")))
    return result
