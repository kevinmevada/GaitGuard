"""
Phase 1 + Phase 2 handcrafted feature column selection (competitor paradigm 1).

Phase 1: spatiotemporal + variability (`temporal`, `spatial`, `asymmetry` config groups).
Phase 2: kinematic / frequency (`phase2_kinematic` config group).

Excludes legacy spectral, wavelet, trunk dynamics, orientation, turning, and Phase 3 deep features.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features.feature_groups import patient_columns_for_group

PHASE12_FEATURE_GROUPS = ("temporal", "spatial", "asymmetry", "phase2_kinematic")

_TRIAL_META = {
    "trial_id",
    "session",
    "participant_id",
    "cohort",
    "risk_label",
    "multiclass_label",
    "fall_probability",
    "laterality_biased",
}


def phase12_trial_bases(config: dict[str, Any]) -> list[str]:
    feat_cfg = config.get("features") or {}
    bases: list[str] = []
    for group in PHASE12_FEATURE_GROUPS:
        bases.extend(feat_cfg.get(group) or [])
    return list(dict.fromkeys(bases))


def trial_column_matches_base(column: str, base: str) -> bool:
    if column == base:
        return True
    if column.endswith(f"_{base}"):
        return True
    for prefix in ("left_", "right_", "lb_", "head_", "lf_", "rf_"):
        if column == f"{prefix}{base}":
            return True
    return False


def phase12_trial_columns(df_columns: list[str], config: dict[str, Any]) -> list[str]:
    bases = phase12_trial_bases(config)
    cols = [
        c
        for c in df_columns
        if c not in _TRIAL_META
        and any(trial_column_matches_base(c, base) for base in bases)
    ]
    return sorted(cols)


def phase12_patient_columns(all_columns: list[str], config: dict[str, Any]) -> list[str]:
    cols: list[str] = []
    for group in PHASE12_FEATURE_GROUPS:
        cols.extend(patient_columns_for_group(all_columns, group, config))
    return sorted(dict.fromkeys(cols))


def load_phase12_trial_matrix(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    from pathlib import Path

    feat_dir = Path(config["paths"]["features"])
    df = pd.read_parquet(feat_dir / "trial_features.parquet")
    df = df[~df["trial_id"].astype(str).str.startswith("daphnet_")].reset_index(drop=True)

    feat_cols = phase12_trial_columns(df.columns.tolist(), config)
    if not feat_cols:
        raise ValueError("No Phase 1+2 trial feature columns matched config groups")

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).values.astype(np.float32)

    label_mode = str(config.get("dataset", {}).get("label_mode", "multiclass")).lower()
    if label_mode == "multiclass" and "multiclass_label" in df.columns:
        y = df["multiclass_label"].values.astype(int)
    else:
        y = df["risk_label"].values.astype(int)

    groups = df["participant_id"].astype(str).values
    return X, y, groups, feat_cols, df


def load_phase12_patient_matrix(
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    from pathlib import Path

    from src.features.feature_matrix import get_numeric_feature_columns

    feat_dir = Path(config["paths"]["features"])
    df = pd.read_parquet(feat_dir / "patient_features.parquet")
    numeric = get_numeric_feature_columns(df)
    feat_cols = [c for c in phase12_patient_columns(numeric, config) if c in df.columns]
    if not feat_cols:
        raise ValueError("No Phase 1+2 patient feature columns matched config groups")

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).values.astype(np.float32)

    label_mode = str(config.get("dataset", {}).get("label_mode", "multiclass")).lower()
    if label_mode == "multiclass" and "multiclass_label" in df.columns:
        y = df["multiclass_label"].values.astype(int)
    else:
        y = df["risk_label"].values.astype(int)

    groups = df["participant_id"].astype(str).values
    return X, y, groups, feat_cols, df
