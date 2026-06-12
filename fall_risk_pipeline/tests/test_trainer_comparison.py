"""Tests for model comparison export (nested CV vs deployed params)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.trainer import ModelTrainer


@pytest.fixture
def trainer(tmp_path: Path) -> ModelTrainer:
    config = {
        "paths": {
            "features": str(tmp_path / "features"),
            "checkpoints": str(tmp_path / "checkpoints"),
            "metrics": str(tmp_path / "metrics"),
        },
        "models": {
            "tuning": {"n_trials": 1, "timeout_per_model": 1, "cv_folds": 2},
            "evaluation": {"random_state": 42},
            "run": ["lightgbm"],
            "ensemble": {"top_k": 2, "methods": []},
        },
    }
    return ModelTrainer(config)


def test_save_comparison_reports_nested_cv_and_deployed_params(trainer: ModelTrainer):
    results = {
        "lightgbm": {
            "cv_auc": 0.72,
            "cv_std": 0.03,
            "tuning_cv_auc": 0.78,
            "params": {"n_estimators": 120, "max_depth": 4},
        },
        "ensemble_stacking": {
            "cv_auc": 0.75,
            "cv_std": 0.02,
            "params": {"ensemble_method": "stacking", "top_models": ["lightgbm"]},
        },
    }
    trainer._save_comparison(results)

    csv_path = trainer.metrics_dir / "model_comparison_cv.csv"
    df = pd.read_csv(csv_path)
    lgb_row = df.loc[df["model"] == "lightgbm"].iloc[0]
    assert lgb_row["cv_auc"] == pytest.approx(0.72)
    assert lgb_row["tuning_cv_auc"] == pytest.approx(0.78)
    assert lgb_row["cv_auc_source"] == "nested_cv"
    assert json.loads(lgb_row["deployed_params"]) == {
        "max_depth": 4,
        "n_estimators": 120,
    }

    ens_row = df.loc[df["model"] == "ensemble_stacking"].iloc[0]
    assert ens_row["cv_auc_source"] == "ensemble_cv"

    params_path = trainer.metrics_dir / "model_deployed_params.json"
    saved = json.loads(params_path.read_text(encoding="utf-8"))
    assert saved["lightgbm"]["n_estimators"] == 120
