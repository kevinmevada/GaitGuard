"""Tests for multiclass evaluation payloads."""

import numpy as np

from src.evaluation.multiclass_metrics import build_multiclass_metric_payload


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
    payload = build_multiclass_metric_payload("test", y_true, y_proba)

    assert payload["label_mode"] == "multiclass"
    assert payload["macro_f1"] == payload["f1"]
    assert len(payload["per_class_metrics"]) == 3
    assert payload["accuracy"] > 0.5
