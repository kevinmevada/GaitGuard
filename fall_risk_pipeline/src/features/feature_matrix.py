"""
Shared patient-level feature matrix loading with optional selection mask.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NON_FEATURE_COLS = {
    "participant_id",
    "cohort",
    "risk_label",
    "multiclass_label",
    "fall_probability",
    "laterality_biased",
    "n_trials",
}

SELECTED_FEATURES_FILE = "selected_features.json"


def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    return df[feat_cols].select_dtypes(include=np.number).columns.tolist()


def load_selected_feature_names(feat_dir: Path, config: dict) -> list[str] | None:
    fscfg = config.get("feature_selection", {})
    if not fscfg.get("enabled", False):
        return None

    path = feat_dir / SELECTED_FEATURES_FILE
    if not path.exists():
        return None

    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)

    features = payload.get("features")
    if not isinstance(features, list) or not features:
        return None

    return [str(name) for name in features]


def load_patient_feature_matrix(
    config: dict,
    *,
    feat_dir: Path | None = None,
    use_selected: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Load (X, y, groups, feature_names, dataframe).

    When feature selection is enabled and ``selected_features.json`` exists,
    columns are restricted to the selected set (order preserved).
    """
    feat_dir = feat_dir or Path(config["paths"]["features"])
    path = feat_dir / "patient_features.parquet"
    df = pd.read_parquet(path)

    feat_cols = get_numeric_feature_columns(df)
    if use_selected:
        selected = load_selected_feature_names(feat_dir, config)
        if selected is not None:
            missing = [c for c in selected if c not in feat_cols]
            if missing:
                raise ValueError(
                    f"Selected features missing from patient matrix: {missing[:5]} "
                    f"({len(missing)} total)"
                )
            feat_cols = selected

    X = df[feat_cols].values.astype(np.float32)
    y = df["risk_label"].values.astype(int)
    groups = df["participant_id"].values

    return X, y, groups, feat_cols, df
