"""Tests for multiclass evaluation payloads."""

import numpy as np

from src.evaluation.multiclass_metrics import (
    _bootstrap_multiclass_auc_ci,
    build_multiclass_metric_payload,
    is_multiclass_metric_result,
)


def test_build_multiclass_metric_payload_per_class():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_proba = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.2, 0.1, 0.7],
    ])
    payload = build_multiclass_metric_payload("test", y_true, y_proba, seed=42)

    assert payload["label_mode"] == "multiclass"
    assert payload["macro_f1"] == payload["f1"]
    assert len(payload["per_class_metrics"]) == 3
    assert payload["accuracy"] > 0.5
    assert "y_prob" not in payload
    assert payload["y_proba_full"].shape == (6, 3)
    assert "y_prob_class_1" in payload


def test_is_multiclass_metric_result():
    assert is_multiclass_metric_result({"label_mode": "multiclass"})
    assert is_multiclass_metric_result(
        {"y_proba_full": np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])}
    )
    assert not is_multiclass_metric_result(
        {"label_mode": "binary", "y_prob": np.array([0.2, 0.8])}
    )


def test_bootstrap_auc_ci_reproducible_with_same_seed():
    rng = np.random.default_rng(0)
    n = 60
    y_true = rng.integers(0, 3, size=n)
    y_proba = rng.random((n, 3))
    y_proba /= y_proba.sum(axis=1, keepdims=True)
    labels = [0, 1, 2]
    ci_a = _bootstrap_multiclass_auc_ci(
        y_true, y_proba, labels, seed=7, n_bootstrap=500
    )
    ci_b = _bootstrap_multiclass_auc_ci(
        y_true, y_proba, labels, seed=7, n_bootstrap=500
    )
    ci_c = _bootstrap_multiclass_auc_ci(
        y_true, y_proba, labels, seed=99, n_bootstrap=500
    )
    assert ci_a == ci_b
    assert ci_a != ci_c
