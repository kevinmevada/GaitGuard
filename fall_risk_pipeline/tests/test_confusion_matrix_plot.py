"""HIGH-005: publishable normalized confusion matrix figures."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.evaluation.evaluator import Evaluator


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


def test_row_normalized_confusion_matrix():
    cm = np.array([[10, 0, 0], [2, 6, 2], [0, 1, 9]], dtype=float)
    norm = Evaluator._row_normalized_confusion_matrix(cm)
    assert np.allclose(norm.sum(axis=1), [1.0, 1.0, 1.0])
    assert np.isclose(norm[0, 0], 1.0)
    assert np.isclose(norm[1, 1], 0.6)


def test_confusion_matrix_tick_labels_multiclass():
    labels = Evaluator._confusion_matrix_tick_labels(3, multiclass=True)
    assert len(labels) == 3
    assert "Healthy" in labels[0]


def test_plot_confusion_matrices_writes_annotated_figure(evaluator: Evaluator):
    results = {
        "xgboost": {
            "confusion_matrix": np.array([[10, 2, 1], [1, 8, 2], [0, 1, 9]]),
            "label_mode": "multiclass",
        }
    }
    evaluator._plot_confusion_matrices(results)
    out = evaluator.fig_models / "cm_xgboost.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_confusion_matrices_uses_seaborn_not_imshow():
    source = inspect.getsource(Evaluator._plot_confusion_matrices)
    assert "sns.heatmap" in source
    assert "imshow" not in source
