"""Tests for MLP training safeguards (ML-004 / ML-008)."""

from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

from src.models.trainer import ModelTrainer


def _trainer(tmp_path: Path) -> ModelTrainer:
    return ModelTrainer(
        {
            "paths": {
                "features": str(tmp_path / "features"),
                "checkpoints": str(tmp_path / "checkpoints"),
                "metrics": str(tmp_path / "metrics"),
            },
            "dataset": {"label_mode": "multiclass"},
            "models": {
                "run": ["mlp"],
                "tuning": {"n_trials": 1, "timeout_per_model": 10, "cv_folds": 3},
                "evaluation": {"random_state": 42},
            },
        }
    )


def test_mlp_sample_weights_favor_minority_class(tmp_path: Path):
    trainer = _trainer(tmp_path)
    y = np.array([0, 0, 0, 0, 1], dtype=int)
    weights = trainer._balanced_sample_weights(y)
    assert weights[y == 1].mean() > weights[y == 0].mean()


def test_mlp_pipeline_fit_params_include_sample_weight(tmp_path: Path):
    trainer = _trainer(tmp_path)
    y = np.array([0, 0, 1, 2], dtype=int)
    params = trainer._pipeline_fit_params("mlp", y)
    assert "clf__sample_weight" in params
    assert len(params["clf__sample_weight"]) == len(y)


def test_mlp_classifier_disables_early_stopping(tmp_path: Path):
    trainer = _trainer(tmp_path)
    clf = trainer._build_classifier_from_params(
        "mlp",
        {"units": 64, "alpha": 0.001, "learning_rate_init": 0.001},
        y=np.array([0, 1, 2]),
    )
    assert isinstance(clf, MLPClassifier)
    assert clf.early_stopping is False
