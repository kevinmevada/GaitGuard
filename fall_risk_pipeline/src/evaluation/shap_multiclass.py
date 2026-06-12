"""Multiclass SHAP value parsing and per-class importance helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.dataset.label_policy import MULTICLASS_NAMES


def multiclass_display_name(class_index: int) -> str:
    return MULTICLASS_NAMES.get(class_index, f"class_{class_index}")


def split_shap_by_class(
    shap_vals: Any, n_classes: int | None = None
) -> np.ndarray:
    """
    Normalize TreeExplainer / KernelExplainer output to
    (n_samples, n_features) for binary or (n_samples, n_features, n_classes).
    """
    if isinstance(shap_vals, list):
        arrays = [np.asarray(v, dtype=float) for v in shap_vals]
        if not arrays:
            raise ValueError("empty SHAP value list")
        if arrays[0].ndim == 1:
            arrays = [a.reshape(1, -1) for a in arrays]
        stacked = np.stack(arrays, axis=0)
        if stacked.shape[0] == len(arrays):
            return np.moveaxis(stacked, 0, -1)
        return stacked

    arr = np.asarray(shap_vals, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if n_classes is not None and arr.shape[-1] == n_classes:
            return arr
        if n_classes is not None and arr.shape[0] == n_classes:
            return np.moveaxis(arr, 0, -1)
        if arr.shape[-1] <= 10:
            return arr
        return np.moveaxis(arr, 0, -1)
    raise ValueError(f"unsupported SHAP array shape: {arr.shape}")


def is_multiclass_shap(per_class: np.ndarray) -> bool:
    return per_class.ndim == 3 and per_class.shape[2] > 1


def global_shap_matrix(per_class: np.ndarray) -> np.ndarray:
    """Class-averaged |SHAP| per sample: (n_samples, n_features)."""
    if not is_multiclass_shap(per_class):
        return per_class
    return np.abs(per_class).mean(axis=2)


def global_mean_abs_importance(per_class: np.ndarray) -> np.ndarray:
    """Global summary importance: mean |SHAP| over samples (and classes if multiclass)."""
    return np.abs(global_shap_matrix(per_class)).mean(axis=0)


def per_class_mean_abs_importance(per_class: np.ndarray) -> dict[int, np.ndarray]:
    """Per-class mean |SHAP| over samples: class_index -> (n_features,)."""
    if not is_multiclass_shap(per_class):
        return {0: np.abs(per_class).mean(axis=0)}
    n_classes = per_class.shape[2]
    return {
        c: np.abs(per_class[:, :, c]).mean(axis=0) for c in range(n_classes)
    }
