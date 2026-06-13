"""Tests for threshold comparison CSV labeling."""

from __future__ import annotations

import pandas as pd

from src.evaluation.evaluator import Evaluator


def test_save_threshold_comparison_includes_threshold_source(tmp_path):
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
    ev = Evaluator(config)
    ev.metrics_dir.mkdir(parents=True, exist_ok=True)
    ev._save_threshold_comparison({
        "xgboost": {
            "label_mode": "binary",
            "accuracy": 0.8,
            "f1": 0.75,
            "sensitivity": 0.7,
            "specificity": 0.85,
            "threshold_train_youden_mean": 0.42,
            "threshold_eval_youden": 0.38,
            "accuracy_at_0.5": 0.72,
            "f1_at_0.5": 0.68,
            "sensitivity_at_0.5": 0.65,
            "specificity_at_0.5": 0.78,
            "accuracy_eval_youden": 0.88,
            "f1_eval_youden": 0.84,
            "sensitivity_eval_youden": 0.9,
            "specificity_eval_youden": 0.86,
            "delta_accuracy_eval_minus_train": 0.08,
            "delta_f1_eval_minus_train": 0.09,
            "primary_threshold_source": "unbiased_train_fold",
        }
    })

    df = pd.read_csv(ev.metrics_dir / "metrics_threshold_comparison.csv")
    assert set(df["threshold_source"]) == {
        "unbiased_train_fold",
        "fixed_0.5",
        "optimistic_eval_set",
    }
    primary = df.loc[df["threshold_source"] == "unbiased_train_fold"].iloc[0]
    optimistic = df.loc[df["threshold_source"] == "optimistic_eval_set"].iloc[0]
    assert primary["accuracy"] == 0.8
    assert optimistic["accuracy"] == 0.88
    assert optimistic["delta_accuracy_vs_train_fold"] == 0.08
