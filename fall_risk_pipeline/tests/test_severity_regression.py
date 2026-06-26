"""Tests for Navita-style severity regression (MAE, MSE, R² on latent head)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def _minimal_config(tmp_path: Path) -> dict:
    return {
        "paths": {
            "processed_data": str(tmp_path / "data" / "processed"),
            "metrics": str(tmp_path / "results" / "metrics"),
            "checkpoints": str(tmp_path / "checkpoints"),
        },
        "reproducibility": {"seed": 42},
        "primary_model": {
            "bilstm_ae_ensemble": {
                "enabled": True,
                "bilstm_autoencoder": {"max_epochs": 2, "early_stopping_patience": 1},
            }
        },
        "deep_learning": {"sequence_length": 20, "overlap": 0.5},
        "dataset": {"sensor_positions": ["head", "lower_back", "left_foot", "right_foot"]},
        "severity_regression": {
            "enabled": True,
            "targets": ["ordinal_severity"],
            "cohort_scopes": ["all_8", "neuro_ortho"],
            "navita_reference": {"mae": 5.0, "mse": None, "r2": None},
            "regression_head": {
                "hidden_dim": 8,
                "max_epochs": 5,
                "early_stopping_patience": 2,
                "batch_size": 8,
            },
        },
    }


def _synthetic_bundle(n_trials: int = 16) -> MagicMock:
    rng = np.random.default_rng(0)
    trial_ids = [f"t{i}" for i in range(n_trials)]
    cohorts = np.array(
        ["Healthy"] * (n_trials // 2) + ["PD"] * (n_trials - n_trials // 2),
        dtype=object,
    )
    pids = np.array([f"p{i // 4}" for i in range(n_trials)], dtype=object)
    windows = {
        tid: rng.normal(0, 1, size=(3, 12, 20)).astype(np.float32) for tid in trial_ids
    }
    from src.models.bilstm_autoencoder import SensorChannelSlice

    slices = [
        SensorChannelSlice("head", 0, 3),
        SensorChannelSlice("lower_back", 3, 6),
        SensorChannelSlice("left_foot", 6, 9),
        SensorChannelSlice("right_foot", 9, 12),
    ]
    bundle = MagicMock()
    bundle.trial_ids = trial_ids
    bundle.participant_ids = pids
    bundle.cohorts = cohorts
    bundle.windows = windows
    bundle.sensor_slices = slices
    bundle.n_channels = 12
    bundle.window_len = 20
    return bundle


def _trial_metadata(bundle: MagicMock) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": bundle.trial_ids,
            "participant_id": bundle.participant_ids,
            "cohort": bundle.cohorts,
            "fall_probability": [5.2 if c == "Healthy" else 45.0 for c in bundle.cohorts],
        }
    )


def test_ordinal_severity_mapping():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.dataset.severity_targets import (
        COHORT_ORDINAL_SEVERITY,
        cohort_in_scope,
        ordinal_severity_from_cohort,
    )

    assert ordinal_severity_from_cohort("Healthy") == 0.0
    assert ordinal_severity_from_cohort("HipOA") == 1.0
    assert ordinal_severity_from_cohort("PD") == 2.0
    assert cohort_in_scope("Healthy", "neuro_ortho") is False
    assert cohort_in_scope("PD", "neuro_ortho") is True
    assert len(COHORT_ORDINAL_SEVERITY) == 8


def test_compute_severity_regression_metrics():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.severity_regression_evaluator import compute_severity_regression_metrics

    y = np.array([0.0, 1.0, 2.0, 1.0])
    pred = np.array([0.1, 0.9, 2.1, 1.2])
    m = compute_severity_regression_metrics(y, pred)
    assert m["n"] == 4
    assert m["mae"] < 0.2
    assert np.isfinite(m["r2"])


def test_run_severity_regression_exports_artifacts(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.severity_regression_evaluator import run_severity_regression_evaluation

    config = _minimal_config(tmp_path)
    proc = Path(config["paths"]["processed_data"])
    proc.mkdir(parents=True, exist_ok=True)
    bundle = _synthetic_bundle()
    meta = _trial_metadata(bundle)
    meta.to_csv(proc / "trial_metadata.csv", index=False)

    latent_dim = 16

    def fake_latents(bundle, train_tids, test_tids, healthy_train_tids, config, **kwargs):
        rng = np.random.default_rng(hash(tuple(test_tids)) % 10_000)
        lat_tr = rng.normal(size=(len(train_tids), latent_dim)).astype(np.float32)
        lat_te = rng.normal(size=(len(test_tids), latent_dim)).astype(np.float32)
        return lat_tr, lat_te

    with (
        patch(
            "src.evaluation.severity_regression_evaluator.load_voisard_trial_windows",
            return_value=bundle,
        ),
        patch(
            "src.evaluation.severity_regression_evaluator.build_fold_trial_latents",
            side_effect=fake_latents,
        ),
    ):
        metrics_df = run_severity_regression_evaluation(config)

    metrics_dir = Path(config["paths"]["metrics"])
    assert (metrics_dir / "severity_regression_metrics.csv").is_file()
    assert (metrics_dir / "severity_regression_oof_predictions.csv").is_file()
    assert (metrics_dir / "severity_regression_navita_comparison.json").is_file()
    assert (metrics_dir / "severity_regression_navita_comparison.md").is_file()

    assert not metrics_df.empty
    assert "mae" in metrics_df.columns
    assert "r2" in metrics_df.columns
    assert set(metrics_df["cohort_scope"]) == {"all_8", "neuro_ortho"}

    comp = json.loads((metrics_dir / "severity_regression_navita_comparison.json").read_text())
    assert comp["gaitguard_model"] == "bilstm_ae_latent_regressor"


def test_latent_regression_head_trains_on_tiny_data():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.models.latent_severity_regressor import predict_latent_regression, train_latent_regression_head

    config = {
        "reproducibility": {"seed": 0},
        "severity_regression": {
            "regression_head": {"hidden_dim": 4, "max_epochs": 20, "batch_size": 4},
        },
    }
    Z = np.random.default_rng(1).normal(size=(24, 8)).astype(np.float32)
    y = Z[:, 0] * 0.5 + 1.0
    model = train_latent_regression_head(Z, y, config)
    pred = predict_latent_regression(model, Z)
    assert pred.shape == (24,)
    assert np.isfinite(pred).all()
