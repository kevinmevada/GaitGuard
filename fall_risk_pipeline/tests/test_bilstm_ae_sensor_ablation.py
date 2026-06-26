"""Tests for BiLSTM-AE 4/2/1 sensor ablation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_mask_inactive_sensors_zeros_foot_blocks():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.models.bilstm_autoencoder import SensorChannelSlice
    from src.models.bilstm_ae_scoring import mask_inactive_sensors

    slices = [
        SensorChannelSlice("head", 0, 3),
        SensorChannelSlice("lower_back", 3, 6),
        SensorChannelSlice("left_foot", 6, 9),
        SensorChannelSlice("right_foot", 9, 12),
    ]
    x = np.ones((2, 12, 20), dtype=np.float32)
    masked = mask_inactive_sensors(x, slices, ("lower_back",))
    assert masked[:, 3:6, :].sum() > 0
    assert masked[:, :3, :].sum() == 0
    assert masked[:, 6:, :].sum() == 0


def test_run_bilstm_ae_sensor_ablation_table(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.bilstm_ae_sensor_ablation import run_bilstm_ae_sensor_ablation
    from src.models.bilstm_ae_scoring import METHOD_ENSEMBLE

    metrics_dir = tmp_path / "metrics"
    config = {
        "paths": {
            "processed_data": str(tmp_path / "processed"),
            "metrics": str(metrics_dir),
            "checkpoints": str(tmp_path / "checkpoints"),
            "figures_models": str(tmp_path / "figures"),
        },
        "reproducibility": {"seed": 42},
        "models": {"evaluation": {"primary_endpoint": "bilstm_ae_ensemble"}},
        "primary_model": {"bilstm_ae_ensemble": {"enabled": True}},
        "sensor_ablation": {"bilstm_ae": {"enabled": True}},
        "deep_learning": {"sequence_length": 20, "overlap": 0.5},
        "dataset": {"sensor_positions": ["head", "lower_back", "left_foot", "right_foot"]},
    }

    bundle = MagicMock()
    bundle.trial_ids = [f"t{i}" for i in range(8)]
    bundle.participant_ids = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"])
    bundle.cohorts = np.array(["Healthy"] * 4 + ["PD"] * 4)
    bundle.windows = {tid: np.zeros((2, 12, 20), np.float32) for tid in bundle.trial_ids}
    bundle.sensor_slices = []
    bundle.n_channels = 12
    bundle.window_len = 20

    fake_loso_calls = {"n": 0}

    def loso_side_effect(bundle, config, **kwargs):
        fake_loso_calls["n"] += 1
        order_auc = [0.88, 0.78, 0.68]
        auc = order_auc[fake_loso_calls["n"] - 1]
        return {
            "voisard_loso_ensemble_auc": auc,
            "voisard_loso_ensemble_auc_pr": auc - 0.05,
            "voisard_sensitivity": 0.7,
            "voisard_specificity": 0.8,
            "n_trials_scored": 8,
        }

    with (
        patch(
            "src.evaluation.bilstm_ae_sensor_ablation.load_voisard_trial_windows",
            return_value=bundle,
        ),
        patch(
            "src.evaluation.bilstm_ae_sensor_ablation.apply_sensor_mask_to_bundle",
            side_effect=lambda b, a: b,
        ),
        patch(
            "src.evaluation.bilstm_ae_sensor_ablation._loso_ensemble_auc",
            side_effect=loso_side_effect,
        ),
        patch(
            "src.evaluation.bilstm_ae_sensor_ablation._train_deploy_4sensor_for_daphnet",
            return_value=(MagicMock(), MagicMock(), []),
        ),
        patch(
            "src.evaluation.bilstm_ae_sensor_ablation.evaluate_daphnet_lb_scores",
            return_value={"lb_reconstruction_auc": 0.79, "lb_reconstruction_auc_pr": 0.55},
        ),
    ):
        df = run_bilstm_ae_sensor_ablation(config)

    assert len(df) == 3
    assert list(df.sort_values("n_sensors", ascending=False)["voisard_loso_ensemble_auc"]) == [
        0.88,
        0.78,
        0.68,
    ]
    row4 = df[df["sensor_config"] == "4_sensor"].iloc[0]
    assert row4["daphnet_lb_recon_auc"] == 0.79
    row1 = df[df["sensor_config"] == "1_sensor_lb"].iloc[0]
    assert row1["eval_daphnet"] == "not_applicable"
    assert (metrics_dir / "bilstm_ae_sensor_ablation.csv").is_file()
    assert (metrics_dir / "bilstm_ae_sensor_ablation.md").is_file()
    summary = json.loads(
        (metrics_dir / "bilstm_ae_sensor_ablation_summary.json").read_text(encoding="utf-8")
    )
    assert summary["daphnet_4sensor_train_lb_eval_auc"] == 0.79
