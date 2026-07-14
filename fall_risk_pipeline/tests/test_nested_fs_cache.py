"""Nested FS disk cache — resume and LOSO held-out inference."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.features.feature_matrix import nested_rfecv_column_indices
from src.features.nested_fs_cache import (
    count_cached_folds,
    infer_held_out_subject,
    load_fold_selected,
    nested_fs_cache_dir,
    save_fold_selected,
)


def test_infer_held_out_subject_loso():
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    train_idx = np.array([0, 1, 2, 3])  # hold out c
    assert infer_held_out_subject(groups, train_idx) == "c"
    train_mask = np.array([True, True, True, True, False, False])
    assert infer_held_out_subject(groups, train_mask) == "c"


def test_infer_held_out_subject_rejects_multi():
    groups = np.array(["a", "b", "c", "d"])
    train_idx = np.array([0, 1])  # hold out c and d
    assert infer_held_out_subject(groups, train_idx) is None


def test_save_load_fold_roundtrip(tmp_path: Path):
    cfg = {
        "paths": {"metrics": str(tmp_path)},
        "feature_selection": {"enabled": True, "max_features": 20},
        "reproducibility": {"seed": 42},
    }
    feat_cols = ["f0", "f1", "f2"]
    cache_dir = nested_fs_cache_dir(cfg, feat_cols, n_samples=10, n_groups=3)
    save_fold_selected(cache_dir, "subj_1", ["f0", "f2"])
    assert load_fold_selected(cache_dir, "subj_1") == ["f0", "f2"]
    assert count_cached_folds(cache_dir) == 1


def test_nested_rfecv_hits_disk_cache(tmp_path: Path):
    X = np.arange(30, dtype=float).reshape(10, 3)
    y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    groups = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 4)
    feat_cols = ["a", "b", "c"]
    train_idx = np.where(groups != "c")[0]
    config = {
        "paths": {"metrics": str(tmp_path)},
        "feature_selection": {"enabled": True, "max_features": 20},
        "reproducibility": {"seed": 7},
    }

    mock_fs = MagicMock()
    mock_fs.select_feature_names.return_value = ["a", "c"]

    with patch("src.features.feature_selector.FeatureSelector", return_value=mock_fs):
        first = nested_rfecv_column_indices(
            config, X, y, groups, feat_cols, train_idx, held_out_subject="c"
        )
        second = nested_rfecv_column_indices(
            config, X, y, groups, feat_cols, train_idx, held_out_subject="c"
        )

    assert first == [0, 2]
    assert second == [0, 2]
    # RFECV called once; second fold served from disk
    assert mock_fs.select_feature_names.call_count == 1
