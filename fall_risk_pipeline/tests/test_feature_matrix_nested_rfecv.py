"""nested_rfecv_column_indices must accept sklearn integer train indices."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.features.feature_matrix import drop_daphnet_rows, nested_rfecv_column_indices


def test_drop_daphnet_rows_filters_trial_id_prefix():
    df = pd.DataFrame(
        {
            "trial_id": ["HS_1_1", "daphnet_S02", "ACL_1_1"],
            "participant_id": ["HS_1", "S02", "ACL_1"],
            "x": [1.0, 2.0, 3.0],
        }
    )
    out = drop_daphnet_rows(df)
    assert list(out["trial_id"]) == ["HS_1_1", "ACL_1_1"]


def test_drop_daphnet_rows_filters_patient_ids_via_metadata(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    pd.DataFrame(
        {
            "trial_id": ["HS_1_1", "daphnet_S02", "daphnet_S03"],
            "participant_id": ["HS_1", "S02", "S03"],
        }
    ).to_csv(processed / "trial_metadata.csv", index=False)
    df = pd.DataFrame(
        {
            "participant_id": ["HS_1", "S02", "S03", "PD_1"],
            "x": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = drop_daphnet_rows(df, {"paths": {"processed_data": str(processed)}})
    assert list(out["participant_id"]) == ["HS_1", "PD_1"]


def test_nested_rfecv_accepts_integer_train_indices():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    groups = np.arange(10)
    feat_cols = ["a", "b", "c"]
    train_idx = np.array([0, 1, 2, 3, 4, 5, 6])
    config = {"feature_selection": {"enabled": True}}

    mock_fs = MagicMock()
    mock_fs.select_feature_names.return_value = ["a", "c"]

    with patch("src.features.feature_selector.FeatureSelector", return_value=mock_fs):
        col_idx = nested_rfecv_column_indices(
            config, X, y, groups, feat_cols, train_idx
        )

    assert col_idx == [0, 2]
    args = mock_fs.select_feature_names.call_args[0]
    assert args[0].shape == (7, 3)
    assert args[1].shape == (7,)
    assert args[2].shape == (7,)


def test_nested_rfecv_accepts_boolean_train_mask():
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    groups = np.arange(10)
    feat_cols = ["a", "b", "c"]
    train_mask = np.array([True, True, True, True, False, False, False, False, False, False])
    config = {"feature_selection": {"enabled": True}}

    mock_fs = MagicMock()
    mock_fs.select_feature_names.return_value = ["b"]

    with patch("src.features.feature_selector.FeatureSelector", return_value=mock_fs):
        col_idx = nested_rfecv_column_indices(
            config, X, y, groups, feat_cols, train_mask
        )

    assert col_idx == [1]
    args = mock_fs.select_feature_names.call_args[0]
    assert args[0].shape == (4, 3)
