"""ISSUE-008: XGBoost multiclass inverse-frequency sample weights."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.trainer import ModelTrainer


@pytest.fixture
def multiclass_trainer(tmp_path) -> ModelTrainer:
    config = {
        "paths": {
            "features": str(tmp_path / "features"),
            "checkpoints": str(tmp_path / "checkpoints"),
            "metrics": str(tmp_path / "metrics"),
        },
        "dataset": {"label_mode": "multiclass", "high_risk_threshold": 1},
        "models": {
            "tuning": {"n_trials": 1, "timeout_per_model": 1, "cv_folds": 2},
            "evaluation": {"random_state": 42},
            "run": ["xgboost"],
            "ensemble": {"methods": []},
        },
    }
    return ModelTrainer(config)


def test_xgb_sample_weights_upweight_minority_class(multiclass_trainer: ModelTrainer):
    y = np.array([0] * 80 + [1] * 20 + [2] * 80, dtype=int)
    weights = ModelTrainer._xgb_sample_weights(y, multiclass_trainer.config)
    assert weights is not None
    assert weights.shape == y.shape
    assert weights[y == 1].mean() > weights[y == 0].mean()
    assert weights[y == 1].mean() > weights[y == 2].mean()


def test_xgb_sample_weights_none_for_binary(multiclass_trainer: ModelTrainer):
    config = {
        **multiclass_trainer.config,
        "dataset": {"label_mode": "binary", "high_risk_threshold": 1},
    }
    y = np.array([0, 0, 1, 1], dtype=int)
    assert ModelTrainer._xgb_sample_weights(y, config) is None


def test_fit_pipeline_passes_clf_sample_weight(multiclass_trainer: ModelTrainer):
    y = np.array([0, 0, 1, 2, 2], dtype=int)
    X = np.random.default_rng(0).standard_normal((len(y), 4))
    pipe = Pipeline([("scaler", StandardScaler())])
    pipe.fit = MagicMock(return_value=pipe)

    multiclass_trainer.fit_pipeline("xgboost", pipe, X, y)

    pipe.fit.assert_called_once()
    _, kwargs = pipe.fit.call_args
    assert "clf__sample_weight" in kwargs
    assert len(kwargs["clf__sample_weight"]) == len(y)


def test_fit_pipeline_skips_weights_for_lightgbm(multiclass_trainer: ModelTrainer):
    y = np.array([0, 0, 1, 2, 2], dtype=int)
    X = np.random.default_rng(0).standard_normal((len(y), 4))
    pipe = Pipeline([("scaler", StandardScaler())])
    pipe.fit = MagicMock(return_value=pipe)

    multiclass_trainer.fit_pipeline("lightgbm", pipe, X, y)

    pipe.fit.assert_called_once_with(X, y)
