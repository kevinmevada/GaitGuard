"""
Shared patient-level feature matrix loading with optional selection mask.
"""

from __future__ import annotations

import json
from pathlib import Path

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

# Cohort-level constants that must never be persisted in feature parquets (HIGH-003).
TARGET_PROXY_COLS = frozenset({"fall_probability", "laterality_biased"})

SELECTED_FEATURES_FILE = "selected_features.json"


def drop_target_proxies_from_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Remove label-proxy columns before writing or analyzing feature parquets."""
    drop = [c for c in TARGET_PROXY_COLS if c in df.columns]
    if not drop:
        return df
    return df.drop(columns=drop)


def assert_no_target_proxies_in_feature_frame(
    df: pd.DataFrame,
    *,
    context: str = "feature frame",
) -> None:
    present = sorted(TARGET_PROXY_COLS & set(df.columns))
    if present:
        raise AssertionError(
            f"Target proxy column(s) in {context}: {present}. "
            "Store fall_probability and laterality_biased in trial_metadata.csv only."
        )


def sanitize_feature_parquet_artifacts(feat_dir: Path) -> list[str]:
    """Drop target-proxy columns from on-disk feature parquets (migration helper)."""
    updated: list[str] = []
    for name in ("trial_features.parquet", "patient_features.parquet"):
        path = feat_dir / name
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        cleaned = drop_target_proxies_from_feature_frame(df)
        if list(cleaned.columns) != list(df.columns):
            cleaned.to_parquet(path, index=False)
            updated.append(name)
    return updated


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
    assert_no_target_proxies_in_feature_frame(df, context=str(path))

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

    label_mode = config.get("dataset", {}).get("label_mode", "multiclass")
    if label_mode == "multiclass" and "multiclass_label" in df.columns:
        y = df["multiclass_label"].values.astype(int)
    else:
        y = df["risk_label"].values.astype(int)

    groups = df["participant_id"].values

    return X, y, groups, feat_cols, df


def column_indices(feat_cols: list[str], selected: list[str]) -> list[int]:
    """Map selected feature names to column indices (preserves order)."""
    return [feat_cols.index(name) for name in selected]


def nested_rfecv_column_indices(
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat_cols: list[str],
    train_mask: np.ndarray,
) -> list[int]:
    """
    RFECV on train rows only (ML-014). Returns column indices into ``feat_cols``.
    """
    from src.features.feature_selector import FeatureSelector

    train_mask = np.asarray(train_mask, dtype=bool)
    fscfg = config.get("feature_selection", {})
    if not fscfg.get("enabled", False):
        return list(range(len(feat_cols)))

    fs = FeatureSelector(config)
    selected = fs.select_feature_names(
        X[train_mask],
        y[train_mask],
        groups[train_mask],
        feat_cols,
        n_jobs=1,
    )
    return column_indices(feat_cols, selected)


def intersect_nested_rfecv_columns(
    config: dict,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat_cols: list[str],
    train_mask: np.ndarray,
    scenario_indices: list[int],
) -> list[int]:
    """Intersect scenario column indices with per-fold nested RFECV (ML-025)."""
    fscfg = config.get("feature_selection", {})
    if not fscfg.get("nested_in_ablation", True):
        return list(scenario_indices)
    nested_set = set(
        nested_rfecv_column_indices(config, X, y, groups, feat_cols, train_mask)
    )
    return [idx for idx in scenario_indices if idx in nested_set]
