"""Unit tests for src/models/anomaly_detector.py (ISSUE-40).

The full ``GaitAnomalyDetector.run()`` orchestration pulls in subject-split
manifests, LOSO evaluation, and (optionally) the DAPHNET zero-shot transfer
path — exercising it end-to-end belongs in a slower integration/fixture
suite. These tests instead give the previously-untested self-contained
pieces (data loading/cleaning and healthy-only scaler fitting) direct
coverage, plus a construction/config sanity check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.models.anomaly_detector import GaitAnomalyDetector  # noqa: E402


def _make_config(tmp_path: Path) -> dict:
    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    results_dir = tmp_path / "results"
    return {
        "paths": {"features": str(feat_dir), "results": str(results_dir)},
        "reproducibility": {"seed": 42},
    }, feat_dir


def _write_synthetic_trial_features(feat_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(30):
        cohort = "Healthy" if i < 20 else "PD"
        rows.append({
            "trial_id": f"t_{i}" if i < 28 else f"daphnet_{i}",
            "participant_id": f"p{i}",
            "cohort": cohort,
            "risk_label": 0 if cohort == "Healthy" else 1,
            "multiclass_label": 0 if cohort == "Healthy" else 2,
            "fall_probability": 0.0,
            "laterality_biased": False,
            "feat_a": float(rng.normal(0 if cohort == "Healthy" else 2, 0.3)),
            "feat_b": float(rng.normal(0 if cohort == "Healthy" else 2, 0.3)),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(feat_dir / "trial_features.parquet", index=False)
    return df


def test_detector_construction_creates_results_dir(tmp_path):
    config, _ = _make_config(tmp_path)
    detector = GaitAnomalyDetector(config)
    assert detector.results_dir.is_dir()
    assert detector.random_state == 42
    assert detector.models == {}
    assert detector.trial_feature_columns == []


def test_load_data_excludes_daphnet_trials_and_cleans_nans(tmp_path):
    config, feat_dir = _make_config(tmp_path)
    _write_synthetic_trial_features(feat_dir)
    detector = GaitAnomalyDetector(config)

    X, metadata, feature_cols = detector._load_data()

    # daphnet_28 and daphnet_29 excluded; 28 voisard trials remain (t_0..t_27).
    assert "daphnet_28" not in metadata["trial_id"].astype(str).tolist()
    assert "daphnet_29" not in metadata["trial_id"].astype(str).tolist()
    assert len(metadata) == 28
    assert X.shape == (28, len(feature_cols))
    assert set(feature_cols) >= {"feat_a", "feat_b"}
    # No NaN/inf should remain after cleaning.
    assert np.isfinite(X).all()


def test_fit_healthy_scaler_fits_only_on_train_fold_rows(tmp_path):
    config, feat_dir = _make_config(tmp_path)
    _write_synthetic_trial_features(feat_dir)
    detector = GaitAnomalyDetector(config)
    X, metadata, _ = detector._load_data()

    fit_mask = (metadata["cohort"] == "Healthy").to_numpy()
    scaler, X_scaled = detector._fit_healthy_scaler(X, fit_mask)

    # Scaler mean/scale should reflect only the Healthy rows it was fit on.
    assert np.allclose(scaler.mean_, X[fit_mask].mean(axis=0))
    assert X_scaled.shape == X.shape


def test_load_data_missing_feature_file_raises(tmp_path):
    config, _ = _make_config(tmp_path)
    detector = GaitAnomalyDetector(config)
    with pytest.raises(FileNotFoundError):
        detector._load_data()
