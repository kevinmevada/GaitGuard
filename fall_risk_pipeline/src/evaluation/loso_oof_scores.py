"""
Standardized LOSO out-of-fold score exports for statistical benchmark comparisons.

Each model writes ``results/metrics/oof_scores/{model}.csv`` with columns:
  trial_id, participant_id, y_true, score [, cohort]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_auc_score

from src.evaluation.primary_endpoint import ENDPOINT_BILSTM_AE_ENSEMBLE
from src.models.bilstm_ae_scoring import METHOD_ENSEMBLE

OOF_COLUMNS = ("trial_id", "participant_id", "y_true", "score")


def oof_scores_dir(metrics_dir: Path) -> Path:
    return metrics_dir / "oof_scores"


def save_model_oof(metrics_dir: Path, model: str, df: pd.DataFrame) -> Path:
    out_dir = oof_scores_dir(metrics_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c for c in (*OOF_COLUMNS, "cohort") if c in df.columns]
    path = out_dir / f"{model}.csv"
    df[cols].to_csv(path, index=False)
    logger.debug("OOF scores → {}", path)
    return path


def load_model_oof(metrics_dir: Path, model: str) -> pd.DataFrame | None:
    path = oof_scores_dir(metrics_dir) / f"{model}.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    required = {"participant_id", "y_true", "score"}
    if not required.issubset(df.columns):
        raise ValueError(f"OOF file {path} missing columns {required - set(df.columns)}")
    return df


def load_bilstm_ae_oof(metrics_dir: Path) -> pd.DataFrame | None:
    cached = load_model_oof(metrics_dir, ENDPOINT_BILSTM_AE_ENSEMBLE)
    if cached is not None:
        return cached

    legacy = metrics_dir / "bilstm_ae_loso_oof_scores.csv"
    if not legacy.is_file():
        return None

    df = pd.read_csv(legacy)
    score_col = f"{METHOD_ENSEMBLE}_score"
    if score_col not in df.columns:
        return None
    out = pd.DataFrame(
        {
            "trial_id": df["trial_id"].astype(str),
            "participant_id": df["participant_id"].astype(str),
            "cohort": df.get("cohort"),
            "y_true": df["eval_non_healthy"].astype(int),
            "score": df[score_col].astype(float),
        }
    )
    save_model_oof(metrics_dir, ENDPOINT_BILSTM_AE_ENSEMBLE, out)
    return out


def discover_oof_models(metrics_dir: Path, reference: str) -> list[str]:
    oof_dir = oof_scores_dir(metrics_dir)
    models: list[str] = []
    if oof_dir.is_dir():
        models = sorted(p.stem for p in oof_dir.glob("*.csv"))
    if reference not in models and load_bilstm_ae_oof(metrics_dir) is not None:
        models = sorted(set(models) | {reference})
    return models


def binary_positive_score(y_proba: np.ndarray, *, binary: bool) -> np.ndarray:
    """Map classifier probabilities to a single score for AUROC."""
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 1:
        return proba.ravel()
    if binary:
        return proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
    return proba.max(axis=1)


def leave_one_participant_out_aurocs(
    y_true: np.ndarray,
    scores: np.ndarray,
    participant_ids: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Jackknife AUROC: for each participant *p*, AUROC on all trials except *p*.

    Yields paired vectors of length ≤ n_participants for Wilcoxon / CD ranks.
    """
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pids = np.asarray(participant_ids, dtype=str)

    aucs: list[float] = []
    used: list[str] = []
    for pid in np.unique(pids):
        mask = pids != pid
        y_sub = y[mask]
        s_sub = s[mask]
        if len(np.unique(y_sub)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_sub, s_sub)))
        used.append(str(pid))
    return np.asarray(aucs, dtype=float), used


def align_jackknife_aurocs(
    model_aucs: dict[str, tuple[np.ndarray, list[str]]],
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Intersect participant jackknife folds across models."""
    if not model_aucs:
        return [], {}
    common = set(next(iter(model_aucs.values()))[1])
    for _, pids in model_aucs.values():
        common &= set(pids)
    common_list = sorted(common)
    if not common_list:
        return [], {}

    aligned: dict[str, np.ndarray] = {}
    for model, (aucs, pids) in model_aucs.items():
        idx = {p: i for i, p in enumerate(pids)}
        aligned[model] = np.asarray([aucs[idx[p]] for p in common_list], dtype=float)
    return common_list, aligned


def build_oof_export_frame(
    trial_ids: list[str] | np.ndarray,
    participant_ids: np.ndarray,
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    cohorts: np.ndarray | None = None,
) -> pd.DataFrame:
    frame: dict[str, Any] = {
        "trial_id": np.asarray(trial_ids, dtype=str),
        "participant_id": np.asarray(participant_ids, dtype=str),
        "y_true": np.asarray(y_true, dtype=int),
        "score": np.asarray(scores, dtype=float),
    }
    if cohorts is not None:
        frame["cohort"] = np.asarray(cohorts, dtype=str)
    return pd.DataFrame(frame)
