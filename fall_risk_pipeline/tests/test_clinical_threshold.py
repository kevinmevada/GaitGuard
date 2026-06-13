"""Tests for Youden threshold and risk-level assignment."""

from __future__ import annotations

import numpy as np

from src.evaluation.clinical_threshold import (
    DEFAULT_DECISION_THRESHOLD,
    assign_risk_level,
    collapse_labels_binary,
    elevated_risk_probability,
    fixed_threshold_when_inner_cv_unavailable,
    metrics_at_threshold,
    threshold_from_inner_oof,
    youden_threshold,
)


def _multiclass_config() -> dict:
    return {
        "dataset": {
            "label_mode": "multiclass",
            "high_risk_threshold": 1,
            "binary_strategy": "threshold_ge_1",
        }
    }


def test_youden_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    t = youden_threshold(y, p)
    assert 0.2 <= t <= 0.9


def test_metrics_at_threshold_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = metrics_at_threshold(y, p, 0.5)
    assert m["sensitivity"] >= 0.999
    assert m["specificity"] >= 0.999


def test_elevated_probability_multiclass():
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    p = elevated_risk_probability(proba, _multiclass_config())
    np.testing.assert_allclose(p, [0.3, 0.9])


def test_assign_risk_level_relative_to_youden():
    assert assign_risk_level(0.6, 0.5) == "high"
    assert assign_risk_level(0.3, 0.5) == "moderate"
    assert assign_risk_level(0.1, 0.5) == "low"


def test_collapse_multiclass_labels():
    y = np.array([0, 1, 2])
    b = collapse_labels_binary(y, _multiclass_config())
    np.testing.assert_array_equal(b, [0, 1, 1])


def test_ml037_inner_oof_helpers():
    thresh, strategy = fixed_threshold_when_inner_cv_unavailable()
    assert thresh == DEFAULT_DECISION_THRESHOLD
    y = np.repeat([0, 1], 15)
    oof = np.full(len(y), np.nan)
    oof[-5:] = np.linspace(0.2, 0.8, 5)
    assert threshold_from_inner_oof(y, oof, n_splits=5)[0] == DEFAULT_DECISION_THRESHOLD
