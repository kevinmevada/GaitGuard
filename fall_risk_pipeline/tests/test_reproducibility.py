"""Tests for global pipeline seed helpers."""

from __future__ import annotations

import numpy as np

from src.utils.reproducibility import get_pipeline_seed, set_global_seed


def test_get_pipeline_seed_prefers_reproducibility_block():
    config = {
        "reproducibility": {"seed": 7},
        "models": {"evaluation": {"random_state": 42}},
    }
    assert get_pipeline_seed(config) == 7


def test_get_pipeline_seed_falls_back_to_evaluation():
    config = {"models": {"evaluation": {"random_state": 99}}}
    assert get_pipeline_seed(config) == 99


def test_set_global_seed_numpy():
    set_global_seed(42)
    a = np.random.rand(3)
    set_global_seed(42)
    b = np.random.rand(3)
    assert np.allclose(a, b)
