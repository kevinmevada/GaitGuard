"""Anomaly scaler / trial column schema validation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.models.anomaly_feature_schema import (
    validate_trial_columns_for_anomaly_scalers,
)


def _scaler(n: int) -> StandardScaler:
    s = StandardScaler()
    s.fit(np.zeros((3, n)))
    return s


def test_validate_matching_schema_and_scalers():
    cols = ["a", "b", "c"]
    scalers = {"if": _scaler(3), "lof": _scaler(3)}
    schema = {"feature_columns": cols, "n_features": 3}
    ok, msg = validate_trial_columns_for_anomaly_scalers(cols, scalers, schema)
    assert ok
    assert msg == "ok"


def test_validate_count_mismatch():
    cols = ["a", "b"]
    scalers = {"if": _scaler(3)}
    ok, msg = validate_trial_columns_for_anomaly_scalers(cols, scalers, None)
    assert not ok
    assert "n_features_in_" in msg


def test_validate_column_order_mismatch():
    cols = ["b", "a", "c"]
    scalers = {"if": _scaler(3)}
    schema = {"feature_columns": ["a", "b", "c"], "n_features": 3}
    ok, _ = validate_trial_columns_for_anomaly_scalers(cols, scalers, schema)
    assert not ok


def test_scaler_dimension_disagreement():
    scalers = {"if": _scaler(3), "lof": _scaler(4)}
    ok, msg = validate_trial_columns_for_anomaly_scalers(["a", "b", "c"], scalers, None)
    assert not ok
    assert "mismatch" in msg.lower()
