"""Tests for LOSO-appropriate binary AUC confidence intervals."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics_ci import subject_bootstrap_binary_auc_ci


@pytest.fixture
def evaluator(tmp_path) -> Evaluator:
    config = {
        "paths": {
            "features": str(tmp_path / "features"),
            "checkpoints": str(tmp_path / "checkpoints"),
            "metrics": str(tmp_path / "metrics"),
            "figures_models": str(tmp_path / "figures_models"),
            "figures_shap": str(tmp_path / "figures_shap"),
        },
        "reporting": {"figure_format": "png", "figure_dpi": 100},
        "models": {
            "tuning": {"n_trials": 1, "timeout_per_model": 1, "cv_folds": 2},
            "evaluation": {"random_state": 42, "strategy": "nested_group_cv"},
            "run": ["xgboost"],
            "ensemble": {"top_k": 1, "methods": []},
        },
        "reproducibility": {"seed": 42},
    }
    return Evaluator(config)


def test_subject_bootstrap_auc_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    n = 80
    y_true = rng.integers(0, 2, size=n)
    y_prob = rng.random(n)
    y_prob[y_true == 1] += 0.35
    y_prob = np.clip(y_prob, 0.01, 0.99)

    auc = roc_auc_score(y_true, y_prob)
    auc_full, low, high, method = subject_bootstrap_binary_auc_ci(y_true, y_prob, seed=42)
    assert method == "subject_bootstrap"
    assert auc_full == pytest.approx(auc)
    assert low <= auc <= high
    assert np.isfinite(low) and np.isfinite(high)


def test_subject_bootstrap_auc_ci_reproducible():
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9, 0.4, 0.6, 0.35, 0.85, 0.15, 0.75])
    a = subject_bootstrap_binary_auc_ci(y_true, y_prob, seed=42)
    b = subject_bootstrap_binary_auc_ci(y_true, y_prob, seed=42)
    assert a == b


def test_build_metric_payload_uses_subject_bootstrap(evaluator: Evaluator):
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.7, 0.9, 0.4, 0.6, 0.35, 0.85, 0.15, 0.75])
    payload = evaluator._build_metric_payload("test", y_true, y_prob)
    assert payload["auc_ci_method"] == "subject_bootstrap"
    assert payload["auc_ci_low"] <= payload["auc"] <= payload["auc_ci_high"]
