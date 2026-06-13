"""Tests for nested (per-fold) feature selection helpers."""

import numpy as np

from src.evaluation.evaluator import Evaluator
from src.features.feature_selector import FeatureSelector


def _base_config(**fs_overrides) -> dict:
    fs = {
        "enabled": True,
        "nested_in_evaluation": True,
        "max_features": 5,
        "min_features": 2,
        "primary_method": "rfecv",
        "required_feature_substrings": [],
        "cv_folds": 3,
    }
    fs.update(fs_overrides)
    return {
        "paths": {
            "features": "data/features",
            "checkpoints": "results/checkpoints",
            "metrics": "results/metrics",
            "figures_models": "results/figures/models",
            "figures_shap": "results/figures/shap",
        },
        "dataset": {"label_mode": "multiclass"},
        "models": {
            "run": ["xgboost"],
            "tuning": {"cv_folds": 3, "n_trials": 1, "timeout_per_model": 10},
            "evaluation": {"random_state": 42, "strategy": "nested_group_cv"},
            "ensemble": {"enabled": False},
        },
        "feature_selection": fs,
        "explainability": {"shap_enabled": False},
        "reporting": {"figure_format": "png", "figure_dpi": 100},
    }


def test_select_feature_names_respects_max_features():
    rng = np.random.default_rng(42)
    n = 24
    p = 12
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = np.array([0] * 8 + [1] * 8 + [2] * 8)
    groups = np.array([f"p{i // 3}" for i in range(n)])
    feat_cols = [f"f{i}" for i in range(p)]

    selector = FeatureSelector(_base_config(max_features=4, min_features=2))
    selected = selector.select_feature_names(X, y, groups, feat_cols, n_jobs=1)
    assert 2 <= len(selected) <= 4
    assert all(name in feat_cols for name in selected)


def test_evaluator_fold_masks_vary_by_held_out_subject():
    rng = np.random.default_rng(0)
    n = 24
    p = 8
    # Binary labels avoid multiclass ROC failures in tiny StratifiedGroupKFold folds.
    y = np.array([0, 0, 0, 1, 1, 1] * 4)
    X = rng.normal(size=(n, p)).astype(np.float32)
    X[:, 0] += y * 2.0
    groups = np.array([f"s{i // 4}" for i in range(n)])
    feat_cols = [f"f{i}" for i in range(p)]

    cfg = _base_config(max_features=3, cv_folds=3)
    cfg["dataset"] = {"label_mode": "binary", "binary_strategy": "threshold_ge_2"}
    ev = Evaluator(cfg)
    ev._init_fold_feature_masks(X, y, groups, feat_cols)
    assert len(ev._fold_col_idx) >= 2


def test_evaluator_align_matrix_to_features():
    ev = Evaluator(_base_config())
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    aligned = ev._align_matrix_to_features(matrix, ["b", "a"], ["a", "b", "c"])
    assert aligned.shape == (2, 3)
    assert aligned[0, 0] == 2.0
    assert aligned[0, 1] == 1.0
    assert aligned[0, 2] == 0.0


def test_nested_fs_disabled_uses_identity_columns():
    ev = Evaluator(_base_config(nested_in_evaluation=False))
    X = np.arange(12, dtype=float).reshape(4, 3)
    assert np.array_equal(ev._X_for_fold(X, "s0"), X)
