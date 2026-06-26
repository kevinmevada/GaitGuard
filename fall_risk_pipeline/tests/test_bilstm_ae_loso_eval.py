"""Tests for BiLSTM-AE 3-method LOSO primary endpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import yaml

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
        "models": {"evaluation": {"primary_endpoint": "bilstm_ae_ensemble"}},
        "primary_model": {
            "bilstm_ae_ensemble": {
                "enabled": True,
                "ae_reconstruction_weight": 0.40,
                "isolation_forest_latent_weight": 0.33,
                "one_class_svm_latent_weight": 0.27,
            }
        },
        "deep_learning": {"sequence_length": 20, "overlap": 0.5},
        "dataset": {"sensor_positions": ["head", "lower_back", "left_foot", "right_foot"]},
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


def test_run_bilstm_ae_loso_exports_metrics_table(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.bilstm_ae_loso_evaluator import run_bilstm_ae_loso_evaluation
    from src.models.bilstm_ae_scoring import (
        METHOD_AE_RECON,
        METHOD_ENSEMBLE,
        METHOD_IF_LATENT,
        METHOD_OCSVM_LATENT,
    )

    config = _minimal_config(tmp_path)
    bundle = _synthetic_bundle()

    def fake_fold_scores(bundle, train_tids, test_tids, healthy_train_tids, config, **kwargs):
        n_tr, n_te = len(train_tids), len(test_tids)
        rng = np.random.default_rng(1)
        base_tr = rng.uniform(0, 1, n_tr)
        base_te = rng.uniform(0, 1, n_te)
        train_methods = {
            METHOD_AE_RECON: base_tr,
            METHOD_IF_LATENT: base_tr + 0.1,
            METHOD_OCSVM_LATENT: base_tr + 0.2,
        }
        test_methods = {
            METHOD_AE_RECON: base_te,
            METHOD_IF_LATENT: base_te + 0.1,
            METHOD_OCSVM_LATENT: base_te + 0.2,
        }
        train_methods[METHOD_ENSEMBLE] = base_tr + 0.05
        test_methods[METHOD_ENSEMBLE] = base_te + 0.05
        return train_methods, test_methods

    with (
        patch(
            "src.evaluation.bilstm_ae_loso_evaluator.load_voisard_trial_windows",
            return_value=bundle,
        ),
        patch(
            "src.evaluation.bilstm_ae_loso_evaluator.build_fold_trial_scores",
            side_effect=fake_fold_scores,
        ),
    ):
        metrics_df = run_bilstm_ae_loso_evaluation(config)

    metrics_dir = Path(config["paths"]["metrics"])
    assert (metrics_dir / "bilstm_ae_anomaly_metrics.csv").is_file()
    assert (metrics_dir / "bilstm_ae_loso_oof_scores.csv").is_file()
    assert (metrics_dir / "primary_endpoint.json").is_file()

    methods = set(metrics_df["method"])
    assert methods == {
        METHOD_AE_RECON,
        METHOD_IF_LATENT,
        METHOD_OCSVM_LATENT,
        METHOD_ENSEMBLE,
    }
    reg = json.loads((metrics_dir / "primary_endpoint.json").read_text(encoding="utf-8"))
    assert reg["primary_endpoint"] == "bilstm_ae_ensemble"


def test_pipeline_config_primary_endpoint_bilstm_ae():
    cfg_path = PIPELINE_ROOT / "configs" / "pipeline_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["models"]["evaluation"]["primary_endpoint"] == "bilstm_ae_ensemble"
    assert cfg["primary_model"]["bilstm_ae_ensemble"]["enabled"] is True


def test_ensemble_weights_normalised():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.models.bilstm_ae_scoring import ensemble_weights

    w = ensemble_weights(
        {
            "primary_model": {
                "bilstm_ae_ensemble": {
                    "ae_reconstruction_weight": 0.4,
                    "isolation_forest_latent_weight": 0.33,
                    "one_class_svm_latent_weight": 0.27,
                }
            }
        }
    )
    assert abs(sum(w.values()) - 1.0) < 1e-6
