"""HIGH-005: publishable normalized confusion matrix figures."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from src.evaluation.evaluator import Evaluator
from src.evaluation.evaluation_plots import (
    EvaluationPlotter,
    confusion_matrix_tick_labels,
    row_normalized_confusion_matrix,
)


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
    # ISSUE (god-class decomposition): moved from Evaluator._row_normalized_confusion_matrix
    # (a @staticmethod) to a module-level function in evaluation_plots.py, since it has
    # no dependency on Evaluator state. Evaluator still delegates to the same plotting
    # module for its _plot_confusion_matrices method — see the two tests below.
    cm = np.array([[10, 0, 0], [2, 6, 2], [0, 1, 9]], dtype=float)
    norm = row_normalized_confusion_matrix(cm)
    assert np.allclose(norm.sum(axis=1), [1.0, 1.0, 1.0])
    assert np.isclose(norm[0, 0], 1.0)
    assert np.isclose(norm[1, 1], 0.6)


def test_confusion_matrix_tick_labels_multiclass():
    # See note above — moved from Evaluator._confusion_matrix_tick_labels.
    labels = confusion_matrix_tick_labels(3, multiclass=True)
    assert len(labels) == 3
    assert "Healthy" in labels[0]


def test_plot_confusion_matrices_writes_annotated_figure(evaluator: Evaluator):
    results = {
        "xgboost": {
            "confusion_matrix": np.array([[10, 2, 1], [1, 8, 2], [0, 1, 9]]),
            "label_mode": "multiclass",
        }
    }
    # Evaluator._plot_confusion_matrices is now a thin delegating wrapper around
    # EvaluationPlotter.plot_confusion_matrices — this test exercises that
    # delegation end-to-end (real figure written to evaluator.fig_models),
    # not just the extracted module in isolation.
    evaluator._plot_confusion_matrices(results)
    out = evaluator.fig_models / "cm_xgboost.png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_confusion_matrices_uses_seaborn_not_imshow():
    # The actual plotting implementation now lives in EvaluationPlotter
    # (evaluation_plots.py) rather than directly in Evaluator — check the
    # real implementation, not the one-line delegating wrapper in Evaluator.
    source = inspect.getsource(EvaluationPlotter.plot_confusion_matrices)
    assert "sns.heatmap" in source
    assert "imshow" not in source


def test_evaluator_delegates_confusion_matrix_plotting_to_extracted_module(evaluator: Evaluator):
    """Guard against silently reintroducing a second, diverging implementation
    directly in Evaluator instead of delegating to EvaluationPlotter."""
    assert isinstance(evaluator._plotter, EvaluationPlotter)
    source = inspect.getsource(Evaluator._plot_confusion_matrices)
    assert "self._plotter.plot_confusion_matrices" in source
