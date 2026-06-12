"""Tests for global pipeline seed helpers."""

from __future__ import annotations

import os
import warnings

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


def test_set_global_seed_overrides_pythonhashseed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        set_global_seed(42)
    assert os.environ["PYTHONHASHSEED"] == "42"
    assert any("PYTHONHASHSEED" in str(w.message) for w in caught)
