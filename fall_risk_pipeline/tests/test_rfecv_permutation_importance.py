"""CRIT-01: RFECV must not rank by Gini importances when p >> n."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.features.feature_selector import (
    FeatureSelector,
    PermutationImportanceRandomForest,
    _rfe_pipeline_importance,
)


def _config(**fs_overrides) -> dict:
    fs = {
        "enabled": True,
        "max_features": 3,
        "min_features": 2,
        "primary_method": "rfecv",
        "required_feature_substrings": [],
        "cv_folds": 3,
        "rfecv_importance_method": "permutation",
        "rfecv_permutation_n_repeats": 3,
    }
    fs.update(fs_overrides)
    return {
        "paths": {"features": "data/features", "metrics": "results/metrics"},
        "models": {
            "tuning": {"cv_folds": 3},
            "evaluation": {"random_state": 42},
        },
        "feature_selection": fs,
    }


def test_permutation_rf_overwrites_gini_importances():
    rng = np.random.default_rng(0)
    n = 80
    signal = rng.normal(size=n)
    X = np.column_stack([
        signal + rng.normal(scale=0.3, size=n),
        rng.normal(scale=50.0, size=n),
        rng.normal(scale=50.0, size=n),
    ])
    y = (signal > 0).astype(int)

    gini_rf = RandomForestClassifier(
        n_estimators=40, max_depth=4, random_state=0, n_jobs=1
    )
    gini_rf.fit(X, y)
    perm_rf = PermutationImportanceRandomForest(
        n_estimators=40,
        max_depth=4,
        random_state=0,
        n_jobs=1,
        permutation_n_repeats=5,
    )
    perm_rf.fit(X, y)

    assert perm_rf._permutation_importances_[0] > perm_rf._permutation_importances_[1]
    assert not np.allclose(gini_rf.feature_importances_, perm_rf._permutation_importances_)


def test_rfecv_pipeline_uses_permutation_classifier():
    selector = FeatureSelector(_config())
    pipe = selector._rfecv_pipeline()
    assert isinstance(pipe.named_steps["clf"], PermutationImportanceRandomForest)


def test_rfecv_pipeline_gini_fallback():
    selector = FeatureSelector(_config(rfecv_importance_method="gini"))
    pipe = selector._rfecv_pipeline()
    assert isinstance(pipe.named_steps["clf"], RandomForestClassifier)


def test_rfecv_prefers_signal_over_high_variance_noise():
    rng = np.random.default_rng(7)
    n = 36
    groups = np.array([f"p{i // 6}" for i in range(n)])
    signal = rng.normal(size=n)
    feat_cols = ["signal_mean", "noise_a", "noise_b", "noise_c"]
    X = np.column_stack([
        signal + rng.normal(scale=0.2, size=n),
        rng.normal(scale=40.0, size=n),
        rng.normal(scale=40.0, size=n),
        rng.normal(scale=40.0, size=n),
    ])
    y = np.array([0, 0, 1, 1, 2, 2] * 6)

    selector = FeatureSelector(
        _config(max_features=2, min_features=2, cv_folds=3, rfecv_permutation_n_repeats=3)
    )
    selected, detail = selector._select_rfecv(
        X, y, groups, feat_cols, n_jobs=1
    )
    assert detail["importance_method"] == "permutation"
    assert "signal_mean" in selected
    assert sum(name.startswith("noise_") for name in selected) <= 1


def test_importance_getter_reads_pipeline_clf():
    selector = FeatureSelector(_config())
    pipe = selector._rfecv_pipeline()
    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 4))
    y = np.array([0, 0, 1, 1, 2, 2] * 4)
    pipe.fit(X, y)
    importances = _rfe_pipeline_importance(pipe)
    assert importances.shape == (4,)
    assert np.isfinite(importances).all()
