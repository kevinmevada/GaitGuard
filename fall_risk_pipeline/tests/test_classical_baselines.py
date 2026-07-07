"""Tests for Phase 1+2 classical baseline evaluator."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def _synthetic_trial_features(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    rows = []
    pid = 0
    for cohort, label in (("Healthy", 0), ("Healthy", 0), ("PD", 2), ("PD", 2)):
        for trial in range(4):
            row = {
                "trial_id": f"t_{pid}_{trial}",
                "participant_id": f"p{pid}",
                "cohort": cohort,
                "risk_label": 0 if label == 0 else 1,
                "multiclass_label": label,
            }
            base = 0.0 if label == 0 else 2.0
            for col in (
                "stride_duration_s",
                "step_length_m",
                "si_stride_duration",
                "lb_peak_gyro_deg_s",
                "hip_angle_rom_deg",
            ):
                row[col] = float(base + rng.normal(0, 0.2))
            rows.append(row)
        pid += 1

    feat_dir = tmp_path / "data" / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(feat_dir / "trial_features.parquet", index=False)
    return feat_dir


def test_phase12_trial_columns_match_phase_features():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.features.phase12_features import phase12_trial_columns

    cols = [
        "trial_id",
        "stride_duration_s",
        "left_stride_duration_s",
        "lb_peak_gyro_deg_s",
        "lb_spectral_entropy_mean",
        "ae_latent_h00_mean",
    ]
    config = {
        "features": {
            "temporal": ["stride_duration_s"],
            "spatial": [],
            "asymmetry": [],
            "phase2_kinematic": ["peak_gyro_deg_s"],
        }
    }
    matched = phase12_trial_columns(cols, config)
    assert "stride_duration_s" in matched
    assert "left_stride_duration_s" in matched
    assert "lb_peak_gyro_deg_s" in matched
    assert "lb_spectral_entropy_mean" not in matched
    assert "ae_latent_h00_mean" not in matched


def test_run_classical_baselines_exports_matrix(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.classical_baseline_evaluator import run_classical_baselines

    feat_dir = _synthetic_trial_features(tmp_path)
    metrics_dir = tmp_path / "results" / "metrics"
    config = {
        "paths": {"features": str(feat_dir), "metrics": str(metrics_dir)},
        "reproducibility": {"seed": 42},
        "dataset": {"label_mode": "multiclass"},
        "classical_baselines": {
            "enabled": True,
            "evaluation_level": "trial",
            "models": ["svm_rbf", "logistic_regression_l2", "knn"],
            "random_forest": {"optuna_trials": 2},
        },
        "features": {
            "temporal": ["stride_duration_s"],
            "spatial": ["step_length_m"],
            "asymmetry": ["si_stride_duration"],
            "phase2_kinematic": ["peak_gyro_deg_s", "hip_angle_rom_deg"],
        },
    }

    with patch(
        "src.evaluation.classical_baseline_evaluator._tune_random_forest",
        return_value={"n_estimators": 50, "max_depth": 5},
    ):
        df = run_classical_baselines(config)

    assert len(df) == 3
    assert (metrics_dir / "classical_baseline_metrics.csv").is_file()
    assert (metrics_dir / "classical_baseline_metrics.md").is_file()
    for col in ("f1", "balanced_accuracy"):
        assert col in df.columns
        assert df[col].notna().any()
    assert "auroc" in df.columns


def test_tune_random_forest_nested_cv_scoring_binary(tmp_path):
    """Regression test for ISSUE-2 (NameError: roc_auc_score not imported).

    ``test_run_classical_baselines_exports_matrix`` above mocks out
    ``_tune_random_forest`` entirely, so it never actually executes the
    nested-CV Optuna objective that calls ``roc_auc_score`` — this is the
    exact reason the missing-import bug shipped without a failing test.
    This test calls ``_tune_random_forest`` directly (unmocked) on a small
    synthetic binary dataset to exercise that scoring path for real.
    """
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.classical_baseline_evaluator import _tune_random_forest

    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 4))
    y = np.zeros(n, dtype=int)
    y[n // 2 :] = 1
    X[y == 1] += 1.5  # separable-ish signal so AUROC is well-defined

    config = {
        "reproducibility": {"seed": 42},
        "dataset": {"label_mode": "binary"},
        "classical_baselines": {"random_forest": {"optuna_trials": 2}},
    }

    best_params = _tune_random_forest(X, y, config)
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert "max_depth" in best_params


def test_tune_random_forest_nested_cv_scoring_multiclass():
    """Regression test for ISSUE-2, multiclass branch of the same objective."""
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.classical_baseline_evaluator import _tune_random_forest

    rng = np.random.default_rng(1)
    n = 90
    X = rng.normal(size=(n, 4))
    y = np.repeat([0, 1, 2], n // 3)
    X[y == 1] += 1.5
    X[y == 2] += 3.0

    config = {
        "reproducibility": {"seed": 42},
        "dataset": {"label_mode": "multiclass"},
        "classical_baselines": {"random_forest": {"optuna_trials": 2}},
    }

    best_params = _tune_random_forest(X, y, config)
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
