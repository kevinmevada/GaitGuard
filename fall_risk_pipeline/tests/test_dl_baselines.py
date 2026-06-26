"""Tests for DL baseline evaluator (ROCKET / MINIROCKET / InceptionTime / DeepConvLSTM)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_deep_conv_lstm_registered():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.models.deep_models import DEEP_MODEL_REGISTRY, DeepConvLSTM, build_deep_model

    assert "deep_conv_lstm" in DEEP_MODEL_REGISTRY
    model = build_deep_model("deep_conv_lstm", n_channels=12, seq_len=200, n_classes=3)
    assert isinstance(model, DeepConvLSTM)
    x = np.random.randn(2, 12, 200).astype(np.float32)
    import torch

    with torch.no_grad():
        out = model(torch.tensor(x))
    assert out.shape == (2, 3)


def test_run_dl_baselines_rocket_path(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.dl_baseline_evaluator import run_dl_baselines

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "bilstm_ae_anomaly_metrics.csv").write_text(
        "method,auc,f1,balanced_accuracy,n_trials_scored\n"
        "bilstm_ae_ensemble,0.85,0.72,0.70,100\n",
        encoding="utf-8",
    )
    config = {
        "paths": {
            "processed_data": str(tmp_path / "processed"),
            "metrics": str(metrics_dir),
        },
        "reproducibility": {"seed": 42},
        "dataset": {"label_mode": "multiclass", "sensor_positions": ["head", "lower_back", "left_foot", "right_foot"]},
        "deep_learning": {"sequence_length": 20, "overlap": 0.5},
        "dl_baselines": {
            "enabled": True,
            "include_bilstm_ae": True,
            "models": ["minirocket"],
            "minirocket": {"n_kernels": 16},
        },
    }

    trial_ids = [f"t{i}" for i in range(8)]
    groups = np.array(["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"])
    labels = np.array([0, 0, 0, 0, 2, 2, 2, 2])
    windows = {tid: np.random.randn(3, 12, 20).astype(np.float32) for tid in trial_ids}

    fake_metrics = {"f1": 0.7, "auroc": 0.8, "balanced_accuracy": 0.65}

    with (
        patch(
            "src.evaluation.dl_baseline_evaluator._trial_window_records",
            return_value=(trial_ids, groups, labels, windows),
        ),
        patch(
            "src.evaluation.dl_baseline_evaluator._loso_rocket_ridge",
            return_value=fake_metrics,
        ),
    ):
        df = run_dl_baselines(config)

    assert len(df) == 2  # minirocket + bilstm_ae
    assert (metrics_dir / "dl_baseline_metrics.csv").is_file()
    assert (metrics_dir / "dl_competitor_matrix.md").is_file()
    bilstm = df[df["model"] == "bilstm_ae_ensemble"].iloc[0]
    assert float(bilstm["auroc"]) == 0.85
