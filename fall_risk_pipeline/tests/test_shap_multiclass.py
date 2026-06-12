"""Tests for multiclass SHAP parsing helpers."""

import numpy as np

from src.evaluation.shap_multiclass import (
    global_mean_abs_importance,
    global_shap_matrix,
    per_class_mean_abs_importance,
    split_shap_by_class,
)


def test_split_shap_list_to_per_class_cube():
    shap_list = [
        np.ones((4, 3)),
        np.full((4, 3), 2.0),
        np.full((4, 3), 3.0),
    ]
    cube = split_shap_by_class(shap_list, n_classes=3)
    assert cube.shape == (4, 3, 3)
    assert cube[0, 0, 2] == 3.0


def test_global_vs_per_class_importance():
    cube = np.array([
        [[1.0, -2.0, 0.5], [0.0, 1.0, -1.0]],
        [[-1.0, 2.0, 0.5], [1.0, -1.0, 2.0]],
    ])
    global_imp = global_mean_abs_importance(cube)
    per_class = per_class_mean_abs_importance(cube)
    assert global_imp.shape == (2,)
    assert len(per_class) == 3
    assert per_class[1][0] == 2.0
    summary = global_shap_matrix(cube)
    assert summary.shape == (2, 2)
