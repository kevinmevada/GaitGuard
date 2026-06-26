"""
Sealed DAPHNET FOG evaluation — headline cross-dataset AUROC (run once).

Trains Healthy-reference anomaly scorers on **Voisard LB accelerometry only**
(never on DAPHNET). Scores DAPHNET LB samples; compares to ``fog_labels.npz``
via ``roc_auc_score(y_true, anomaly_scores)`` at eval time only.

Do not tune models against this metric.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.spectral_utils import sample_freezing_index_series
from src.ingestion.daphnet_label_mapping import (
    FOG_LABEL_MANIFEST,
    fog_labels_path,
    load_fog_labels_npz,
)
from src.models.anomaly_scoring import (
    ANOMALY_METHODS,
    fit_method_scores,
    normalise_scores,
)
from src.utils.reproducibility import get_pipeline_seed

SEALED_OUTPUT_NAME = "daphnet_fog_auroc.json"
FI_OUTPUT_NAME = "daphnet_fog_fi_auroc.json"
LB_ACC_COLS = ("acc_x", "acc_y", "acc_z")
MAX_HEALTHY_FIT_SAMPLES = 200_000


class DaphnetSealedEvalError(RuntimeError):
    pass


def _is_daphnet_trial(trial_id: str) -> bool:
    return str(trial_id).startswith("daphnet_")


def load_voisard_healthy_lb_samples(processed_dir: Path) -> np.ndarray:
    """Concatenate Voisard Healthy lower-back acc rows (excludes DAPHNET trials)."""
    meta = pd.read_csv(processed_dir / "trial_metadata.csv")
    signals_dir = processed_dir / "signals"
    healthy = meta[
        (meta["cohort"] == "Healthy") & (~meta["trial_id"].astype(str).apply(_is_daphnet_trial))
    ]
    chunks: list[np.ndarray] = []
    for trial_id in healthy["trial_id"]:
        path = signals_dir / f"{trial_id}_lower_back.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        if not all(c in df.columns for c in LB_ACC_COLS):
            continue
        arr = df[list(LB_ACC_COLS)].to_numpy(dtype=np.float32)
        arr = arr[np.isfinite(arr).all(axis=1)]
        if len(arr):
            chunks.append(arr)
    if not chunks:
        raise DaphnetSealedEvalError("No Voisard Healthy lower_back signals for anomaly fit")
    X = np.vstack(chunks)
    if len(X) > MAX_HEALTHY_FIT_SAMPLES:
        rng = np.random.default_rng(42)
        X = X[rng.choice(len(X), MAX_HEALTHY_FIT_SAMPLES, replace=False)]
    return X


def load_daphnet_lb_samples(processed_dir: Path, subject_id: str) -> np.ndarray:
    path = processed_dir / "signals" / f"daphnet_{subject_id}_lower_back.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    return df[list(LB_ACC_COLS)].to_numpy(dtype=np.float32)


def score_lb_samples_ensemble(
    X_healthy_train: np.ndarray,
    X_query: np.ndarray,
    *,
    random_state: int,
) -> np.ndarray:
    """Healthy-fit anomaly ensemble mean score (higher = more anomalous)."""
    norm_layers: list[np.ndarray] = []
    for method in ANOMALY_METHODS:
        sq, sr, _, _, _ = fit_method_scores(
            X_healthy_train, X_query, method, random_state=random_state
        )
        norm_layers.append(normalise_scores(sq, sr))
    return np.mean(np.stack(norm_layers, axis=0), axis=0)


def load_daphnet_lb_z_axis(processed_dir: Path, subject_id: str) -> np.ndarray:
    """Lower-back vertical acceleration for freezing-index detector."""
    path = processed_dir / "signals" / f"daphnet_{subject_id}_lower_back.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    for col in ("acc_z_grav_free", "acc_z"):
        if col in df.columns:
            return df[col].to_numpy(dtype=np.float32)
    raise DaphnetSealedEvalError(f"No vertical acc column in {path}")


def score_lb_freezing_index(
    z_acc: np.ndarray,
    fs: float,
    *,
    window_s: float = 2.0,
    locomotion_band_hz: tuple[float, float] = (0.5, 3.0),
    freezing_band_hz: tuple[float, float] = (3.0, 8.0),
) -> np.ndarray:
    """Per-sample freezing index (higher → more high-frequency trembling / FOG-like)."""
    return sample_freezing_index_series(
        z_acc,
        fs,
        window_s=window_s,
        locomotion_band_hz=locomotion_band_hz,
        freezing_band_hz=freezing_band_hz,
    ).astype(np.float32)


def run_daphnet_freezing_index_detector_eval(
    config: dict,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """
    Standalone freezing-index FOG detector on DAPHNET LB Z-axis.

    Compare against sealed anomaly AUROC; does not use Voisard training data.
    """
    processed_dir = Path(config["paths"]["processed_data"])
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / FI_OUTPUT_NAME

    dcfg = (config.get("ingestion") or {}).get("daphnet") or {}
    sealed_cfg = dcfg.get("sealed_fog_eval") or {}
    if not sealed_cfg.get("enabled", True):
        logger.info("DAPHNET FOG eval disabled — skipping freezing-index detector")
        return {}

    if out_path.is_file() and not force and not sealed_cfg.get("allow_rerun", False):
        raise DaphnetSealedEvalError(
            f"Freezing-index detector result already exists: {out_path}. "
            "Delete manually or set allow_rerun: true."
        )

    feat_cfg = (config.get("features") or {}).get("phase2_kinematic_frequency") or {}
    fi_cfg = feat_cfg.get("freezing_index") or {}
    loco = fi_cfg.get("locomotion_band_hz", [0.5, 3.0])
    freeze = fi_cfg.get("freezing_band_hz", [3.0, 8.0])
    window_s = float(fi_cfg.get("sample_window_s", 2.0))
    fs = float(config.get("dataset", {}).get("sampling_rate", 100))

    labels_path = Path(sealed_cfg.get("labels_path") or fog_labels_path(processed_dir))
    y_by_subject = load_fog_labels_npz(labels_path)

    all_y: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    per_subject: list[dict[str, Any]] = []

    for subject_id, y_true in sorted(y_by_subject.items()):
        z_acc = load_daphnet_lb_z_axis(processed_dir, subject_id)
        n = min(len(z_acc), len(y_true))
        if n == 0:
            continue
        z_acc, y_true = z_acc[:n], y_true[:n]
        scores = score_lb_freezing_index(
            z_acc,
            fs,
            window_s=window_s,
            locomotion_band_hz=(float(loco[0]), float(loco[1])),
            freezing_band_hz=(float(freeze[0]), float(freeze[1])),
        )
        mask = np.isfinite(scores)
        if not mask.any():
            continue
        all_y.append(y_true[mask])
        all_scores.append(scores[mask])
        if len(np.unique(y_true[mask])) >= 2:
            per_subject.append(
                {
                    "subject_id": subject_id,
                    "n_samples": int(mask.sum()),
                    "auc": float(roc_auc_score(y_true[mask], scores[mask])),
                }
            )

    y_cat = np.concatenate(all_y)
    scores_cat = np.concatenate(all_scores)
    if len(np.unique(y_cat)) < 2:
        raise DaphnetSealedEvalError("FOG labels single-class — cannot compute FI AUROC")

    auc = float(roc_auc_score(y_cat, scores_cat))
    auc_pr = float(average_precision_score(y_cat, scores_cat))

    result: dict[str, Any] = {
        "endpoint": "daphnet_fog_freezing_index_auroc",
        "protocol": "standalone_physics_detector",
        "metric": "roc_auc_score(y_true, freezing_index_z)",
        "labels_source": str(labels_path),
        "labels_never_used_in_training": True,
        "detector": "freezing_index_psd_ratio",
        "bands_hz": {"locomotion": list(loco), "freezing": list(freeze)},
        "sample_window_s": window_s,
        "auc": auc,
        "auc_pr": auc_pr,
        "n_samples": int(len(y_cat)),
        "n_fog_positive": int((y_cat == 1).sum()),
        "per_subject": per_subject,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        metrics_dir / "daphnet_fog_fi_scores.npz",
        y_true=y_cat.astype(np.int8),
        freezing_index=scores_cat.astype(np.float32),
    )
    logger.info("DAPHNET freezing-index AUROC = {:.4f} → {}", auc, out_path)
    return result


def run_daphnet_sealed_fog_eval(config: dict, *, force: bool = False) -> dict[str, Any]:
    """
    Sealed test: one-shot ``roc_auc_score(y_true, anomaly_scores)`` on DAPHNET.

    Refuses to overwrite an existing sealed result unless ``force=True``.
    """
    processed_dir = Path(config["paths"]["processed_data"])
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / SEALED_OUTPUT_NAME

    dcfg = (config.get("ingestion") or {}).get("daphnet") or {}
    sealed_cfg = dcfg.get("sealed_fog_eval") or {}
    if not sealed_cfg.get("enabled", True):
        logger.info("DAPHNET sealed FOG eval disabled in config")
        return {}

    if out_path.is_file() and not force and not sealed_cfg.get("allow_rerun", False):
        raise DaphnetSealedEvalError(
            f"Sealed DAPHNET FOG result already exists: {out_path}. "
            "Delete manually or set ingestion.daphnet.sealed_fog_eval.allow_rerun: true. "
            "Do not tune models against this metric."
        )

    labels_path = Path(sealed_cfg.get("labels_path") or fog_labels_path(processed_dir))
    y_by_subject = load_fog_labels_npz(labels_path)
    X_healthy = load_voisard_healthy_lb_samples(processed_dir)
    rs = get_pipeline_seed(config)

    all_y: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    per_subject_rows: list[dict[str, Any]] = []

    for subject_id, y_true in sorted(y_by_subject.items()):
        X = load_daphnet_lb_samples(processed_dir, subject_id)
        n = min(len(X), len(y_true))
        if n == 0:
            continue
        X, y_true = X[:n], y_true[:n]
        scores = score_lb_samples_ensemble(X_healthy, X, random_state=rs)
        all_y.append(y_true)
        all_scores.append(scores)
        if len(np.unique(y_true)) >= 2:
            per_subject_rows.append(
                {
                    "subject_id": subject_id,
                    "n_samples": int(n),
                    "n_fog": int((y_true == 1).sum()),
                    "auc": float(roc_auc_score(y_true, scores)),
                }
            )

    y_cat = np.concatenate(all_y)
    scores_cat = np.concatenate(all_scores)
    if len(np.unique(y_cat)) < 2:
        raise DaphnetSealedEvalError("DAPHNET FOG labels single-class — cannot compute AUROC")

    auc = float(roc_auc_score(y_cat, scores_cat))
    auc_pr = float(average_precision_score(y_cat, scores_cat))

    result: dict[str, Any] = {
        "endpoint": "daphnet_fog_zero_shot_auroc",
        "protocol": "sealed_single_run",
        "metric": "roc_auc_score(y_true, anomaly_scores)",
        "labels_source": str(labels_path),
        "labels_never_used_in_training": True,
        "train_reference": "voisard_healthy_lower_back_acc_only",
        "eval_input": "daphnet_lower_back_acc_only",
        "auc": auc,
        "auc_pr": auc_pr,
        "n_samples": int(len(y_cat)),
        "n_fog_positive": int((y_cat == 1).sum()),
        "n_subjects": len(y_by_subject),
        "label_mapping": FOG_LABEL_MANIFEST["mapping"],
        "manuscript_note": (
            "Headline DAPHNET AUROC — sealed test. Run once, report once. "
            "Do not tune anomaly models against this holdout."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "per_subject": per_subject_rows,
    }

    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    scores_path = metrics_dir / "daphnet_fog_anomaly_scores.npz"
    np.savez_compressed(
        scores_path,
        y_true=y_cat.astype(np.int8),
        anomaly_scores=scores_cat.astype(np.float32),
        subject_ids=np.array(sorted(y_by_subject.keys())),
    )
    logger.info(
        "DAPHNET sealed FOG AUROC = {:.4f} (n={}) → {}",
        auc,
        len(y_cat),
        out_path,
    )

    # Freezing-index physics detector (comparison; no Voisard training).
    try:
        fi_result = run_daphnet_freezing_index_detector_eval(config, force=force)
        if fi_result:
            result["comparison_freezing_index_detector"] = {
                "auc": fi_result.get("auc"),
                "auc_pr": fi_result.get("auc_pr"),
                "endpoint": fi_result.get("endpoint"),
                "artifact": str(metrics_dir / FI_OUTPUT_NAME),
            }
            out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except DaphnetSealedEvalError as exc:
        logger.warning("Skipping freezing-index comparison: {}", exc)

    return result
