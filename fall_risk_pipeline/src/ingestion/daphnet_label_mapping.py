"""
DAPHNET annotation → FOG ground truth (eval-only; never a feature).

Mapping (after calibration rows removed):
  annotation == 1  →  y_true = 0  (normal walking)
  annotation == 2  →  y_true = 1  (freezing of gait / FOG)

Labels are stored in a separate ``.npz`` artifact — never concatenated into
feature tensors, parquets, or model training matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DAPHNET_ANNOTATION_NORMAL = 1
DAPHNET_ANNOTATION_FOG = 2

Y_TRUE_NORMAL = 0
Y_TRUE_FOG = 1

FOG_LABELS_FILENAME = "fog_labels.npz"
FOG_LABEL_MANIFEST: dict[str, Any] = {
    "mapping": {
        str(DAPHNET_ANNOTATION_NORMAL): Y_TRUE_NORMAL,
        str(DAPHNET_ANNOTATION_FOG): Y_TRUE_FOG,
    },
    "annotation_0": "dropped at ingest (calibration)",
    "usage": (
        "Eval-only sealed test: roc_auc_score(y_true, anomaly_scores). "
        "Never used during model training or hyperparameter tuning."
    ),
}


class DaphnetLabelError(ValueError):
    pass


def annotations_to_y_true(annotation: np.ndarray) -> np.ndarray:
    """Map DAPHNET annotations (1/2) to binary FOG labels (0/1)."""
    ann = np.asarray(annotation, dtype=np.int64)
    y = np.full(len(ann), -1, dtype=np.int8)
    y[ann == DAPHNET_ANNOTATION_NORMAL] = Y_TRUE_NORMAL
    y[ann == DAPHNET_ANNOTATION_FOG] = Y_TRUE_FOG
    bad = ann[(ann != DAPHNET_ANNOTATION_NORMAL) & (ann != DAPHNET_ANNOTATION_FOG)]
    if bad.size:
        raise DaphnetLabelError(
            f"Unexpected DAPHNET annotations (expected 1 or 2): {np.unique(bad)}"
        )
    return y


def align_labels_to_resampled_length(
    annotation: np.ndarray,
    n_resampled: int,
) -> np.ndarray:
    """Nearest-index alignment of per-row annotations to resampled signal length."""
    ann = np.asarray(annotation, dtype=np.int64)
    if n_resampled <= 0:
        return np.array([], dtype=np.int8)
    if len(ann) == 0:
        raise DaphnetLabelError("Cannot align labels: empty annotation array")
    if n_resampled == len(ann):
        return annotations_to_y_true(ann)
    idx = np.linspace(0, len(ann) - 1, n_resampled).round().astype(int)
    idx = np.clip(idx, 0, len(ann) - 1)
    return annotations_to_y_true(ann[idx])


def fog_labels_path(processed_dir: Path) -> Path:
    return Path(processed_dir) / "daphnet" / FOG_LABELS_FILENAME


def save_fog_labels_npz(
    per_subject: dict[str, np.ndarray],
    out_path: Path,
    *,
    trial_ids: dict[str, str] | None = None,
) -> Path:
    """
    Write sealed eval labels to ``fog_labels.npz``.

    Arrays are stored per subject key (e.g. ``S01``). No feature data is written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not per_subject:
        raise DaphnetLabelError("No DAPHNET FOG labels to save")

    payload: dict[str, Any] = {
        "subject_ids": np.array(sorted(per_subject.keys())),
        "annotation_map_normal": DAPHNET_ANNOTATION_NORMAL,
        "annotation_map_fog": DAPHNET_ANNOTATION_FOG,
        "y_true_normal": Y_TRUE_NORMAL,
        "y_true_fog": Y_TRUE_FOG,
    }
    if trial_ids:
        payload["trial_ids"] = np.array(
            [trial_ids.get(sid, f"daphnet_{sid}") for sid in sorted(per_subject.keys())]
        )
    for sid, y in sorted(per_subject.items()):
        payload[sid] = np.asarray(y, dtype=np.int8)

    np.savez_compressed(out_path, **payload)
    return out_path


def load_fog_labels_npz(path: Path) -> dict[str, np.ndarray]:
    """Load per-subject ``y_true`` arrays from ``fog_labels.npz``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DAPHNET FOG labels not found: {path}")
    data = np.load(path, allow_pickle=False)
    subject_ids = [str(s) for s in data["subject_ids"]]
    return {sid: np.asarray(data[sid], dtype=np.int8) for sid in subject_ids}


def assert_labels_not_in_feature_columns(columns: list[str], *, context: str = "") -> None:
    """Guardrail: FOG / annotation columns must never enter feature matrices."""
    forbidden = {
        "annotation",
        "y_true",
        "fog_label",
        "fog",
        "daphnet_annotation",
    }
    found = sorted(forbidden & {c.lower() for c in columns})
    if found:
        prefix = f"{context}: " if context else ""
        raise DaphnetLabelError(
            f"{prefix}FOG/annotation labels leaked into feature columns: {found}"
        )
