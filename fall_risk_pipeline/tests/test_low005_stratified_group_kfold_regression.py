"""LOW-005: StratifiedGroupKFold split regression guard (sklearn 1.8.0)."""

from __future__ import annotations

import numpy as np
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

# Golden splits recorded with scikit-learn==1.8.0, random_state=42, n_splits=5.
# Update intentionally when bumping sklearn in requirements-lock.txt.
_EXPECTED_SKLEARN = "1.8.0"
_N_SPLITS = 5
_RANDOM_STATE = 42

# 26 participants × 2 rows; binary labels balanced across groups.
_GOLDEN_FOLD_SIZES: list[tuple[int, int]] = [
    (40, 12),
    (42, 10),
    (42, 10),
    (42, 10),
    (42, 10),
]

_GOLDEN_TRAIN_HEAD: list[list[int]] = [
    [0, 1, 2, 3, 6],
    [0, 1, 2, 3, 4],
    [2, 3, 4, 5, 8],
    [0, 1, 4, 5, 6],
    [0, 1, 2, 3, 4],
]

_GOLDEN_VAL_HEAD: list[list[int]] = [
    [4, 5, 12, 13, 16],
    [14, 15, 18, 19, 24],
    [0, 1, 6, 7, 20],
    [2, 3, 8, 9, 28],
    [10, 11, 22, 23, 30],
]


def _synthetic_grouped_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    n_groups = 26
    rows_per = 2
    groups = np.repeat(np.arange(n_groups), rows_per)
    y = np.repeat(np.array([0] * 13 + [1] * 13), rows_per)
    X = rng.normal(size=(len(groups), 4))
    return X, y, groups


def _pipeline_stratified_group_kfold() -> StratifiedGroupKFold:
    """Match ModelTrainer / FeatureSelector defaults."""
    return StratifiedGroupKFold(
        n_splits=_N_SPLITS,
        shuffle=True,
        random_state=_RANDOM_STATE,
    )


def test_stratified_group_kfold_sklearn_version_pinned():
    assert sklearn.__version__ == _EXPECTED_SKLEARN, (
        f"Update LOW-005 golden splits when bumping scikit-learn from {_EXPECTED_SKLEARN}"
    )


def test_stratified_group_kfold_fold_sizes_regression():
    X, y, groups = _synthetic_grouped_dataset()
    cv = _pipeline_stratified_group_kfold()
    splits = list(cv.split(X, y, groups))
    assert len(splits) == _N_SPLITS
    sizes = [(len(tr), len(va)) for tr, va in splits]
    assert sizes == _GOLDEN_FOLD_SIZES


def test_stratified_group_kfold_train_val_index_heads_regression():
    X, y, groups = _synthetic_grouped_dataset()
    cv = _pipeline_stratified_group_kfold()
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        assert train_idx[:5].tolist() == _GOLDEN_TRAIN_HEAD[fold_idx]
        assert val_idx[:5].tolist() == _GOLDEN_VAL_HEAD[fold_idx]


def test_stratified_group_kfold_no_group_leakage_between_train_val():
    X, y, groups = _synthetic_grouped_dataset()
    cv = _pipeline_stratified_group_kfold()
    for train_idx, val_idx in cv.split(X, y, groups):
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert train_groups.isdisjoint(val_groups)
