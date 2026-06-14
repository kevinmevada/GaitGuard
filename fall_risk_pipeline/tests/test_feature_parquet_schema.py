"""Schema guards for feature parquets (HIGH-003 target-proxy leakage)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.features.feature_extractor import FeatureExtractor
from src.features.feature_matrix import (
    TARGET_PROXY_COLS,
    assert_no_target_proxies_in_feature_frame,
    drop_target_proxies_from_feature_frame,
    sanitize_feature_parquet_artifacts,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATIENT_PARQUET = (
    REPO_ROOT / "fall_risk_pipeline" / "data" / "features" / "patient_features.parquet"
)
DEFAULT_TRIAL_PARQUET = (
    REPO_ROOT / "fall_risk_pipeline" / "data" / "features" / "trial_features.parquet"
)


def test_drop_target_proxies_from_feature_frame():
    df = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "fall_probability": [67.3],
            "laterality_biased": [True],
            "feat_a": [1.0],
        }
    )
    cleaned = drop_target_proxies_from_feature_frame(df)
    assert not TARGET_PROXY_COLS.intersection(cleaned.columns)
    assert "feat_a" in cleaned.columns


def test_assert_no_target_proxies_raises():
    df = pd.DataFrame({"fall_probability": [5.2]})
    with pytest.raises(AssertionError, match="fall_probability"):
        assert_no_target_proxies_in_feature_frame(df, context="test frame")


def test_aggregate_to_patient_excludes_target_proxies():
    trial_df = pd.DataFrame(
        {
            "participant_id": ["p1", "p1"],
            "trial_id": ["t1", "t2"],
            "session": [1, 2],
            "cohort": ["PD", "PD"],
            "risk_label": [2, 2],
            "multiclass_label": [5, 5],
            "signal_feat": [0.1, 0.2],
        }
    )
    cfg = {
        "paths": {"features": ".", "processed_data": ".", "metrics": "."},
        "dataset": {"sampling_rate": 100},
        "features": {},
    }
    patient_df = FeatureExtractor(cfg)._aggregate_to_patient(trial_df)
    patient_df = drop_target_proxies_from_feature_frame(patient_df)
    assert_no_target_proxies_in_feature_frame(patient_df, context="patient aggregation")
    assert "signal_feat_mean" in patient_df.columns
    assert "risk_label" in patient_df.columns


def test_sanitize_feature_parquet_artifacts(tmp_path: Path):
    feat_dir = tmp_path / "features"
    feat_dir.mkdir()
    df = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "fall_probability": [5.2],
            "laterality_biased": [False],
            "feat": [1.0],
        }
    )
    df.to_parquet(feat_dir / "patient_features.parquet", index=False)

    updated = sanitize_feature_parquet_artifacts(feat_dir)
    assert updated == ["patient_features.parquet"]

    cleaned = pd.read_parquet(feat_dir / "patient_features.parquet")
    assert_no_target_proxies_in_feature_frame(cleaned)


@pytest.mark.parametrize(
    "parquet_path",
    [DEFAULT_PATIENT_PARQUET, DEFAULT_TRIAL_PARQUET],
)
def test_no_target_proxy_in_feature_parquet(parquet_path: Path):
    if not parquet_path.exists():
        pytest.skip(f"{parquet_path.name} not present (run feature extraction first)")

    df = pd.read_parquet(parquet_path)
    for col in sorted(TARGET_PROXY_COLS):
        assert col not in df.columns, f"Target proxy {col} found in {parquet_path.name}"
