"""Tests for stratified validation split in deep learning LOSO."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import StratifiedShuffleSplit


def _stratified_val_indices(
    y_train: np.ndarray,
    *,
    seed: int,
    test_size: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror DeepLearningPipeline._loso_evaluate val split logic."""
    val_size = max(1, int(test_size * len(y_train)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    try:
        train_idx, val_idx = next(sss.split(np.arange(len(y_train)), y_train))
        return train_idx, val_idx
    except ValueError:
        rng = np.random.default_rng(seed)
        val_idx = rng.choice(len(y_train), val_size, replace=False)
        train_idx = np.setdiff1d(np.arange(len(y_train)), val_idx)
        return train_idx, val_idx


def test_stratified_val_split_preserves_all_classes():
    # Imbalanced but each class has enough windows for a 10% val draw.
    y_train = np.array([0] * 40 + [1] * 10 + [2] * 10, dtype=int)
    _, val_idx = _stratified_val_indices(y_train, seed=42)
    y_val = y_train[val_idx]
    assert len(np.unique(y_val)) == 3


def test_stratified_val_split_reproducible():
    y_train = np.array([0] * 30 + [1] * 10 + [2] * 10, dtype=int)
    _, val_a = _stratified_val_indices(y_train, seed=7)
    _, val_b = _stratified_val_indices(y_train, seed=7)
    np.testing.assert_array_equal(val_a, val_b)


def test_random_fallback_when_stratification_impossible():
    # One sample per class: stratified 10% val cannot satisfy sklearn constraints.
    y_train = np.array([0, 1, 2], dtype=int)
    with pytest.raises(ValueError):
        next(
            StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0).split(
                np.arange(len(y_train)), y_train
            )
        )
    _, val_idx = _stratified_val_indices(y_train, seed=0)
    assert len(val_idx) >= 1
