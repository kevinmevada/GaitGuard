"""Tests for participant-level inner validation split in deep learning LOSO."""

from __future__ import annotations

import numpy as np

from src.models.deep_trainer import DeepLearningPipeline


def _participants_from_labels(labels: dict[str, int], windows_per_pid: int = 5):
    """Synthetic participants dict with fixed window count per participant."""
    participants = {}
    for pid, label in labels.items():
        participants[pid] = {
            "windows": np.arange(windows_per_pid * 32, dtype=np.float32).reshape(
                windows_per_pid, 4, 8
            ),
            "label": label,
            "trial_ids": [f"{pid}_t0"],
        }
    return participants


def test_participant_val_split_no_overlap():
    labels = {f"p{i}": i % 3 for i in range(30)}
    train_pids = list(labels.keys())
    inner_train, inner_val = DeepLearningPipeline.split_inner_train_val_participants(
        train_pids, labels, seed=42, val_fraction=0.1
    )
    assert inner_train and inner_val
    assert set(inner_train).isdisjoint(set(inner_val))
    assert set(inner_train) | set(inner_val) == set(train_pids)


def test_participant_val_split_preserves_classes_when_possible():
    labels = {f"p{i}": i % 3 for i in range(30)}
    train_pids = list(labels.keys())
    _, inner_val = DeepLearningPipeline.split_inner_train_val_participants(
        train_pids, labels, seed=7, val_fraction=0.2
    )
    val_labels = [labels[pid] for pid in inner_val]
    assert len(np.unique(val_labels)) >= 2


def test_participant_val_split_reproducible():
    labels = {f"p{i}": i % 3 for i in range(24)}
    train_pids = list(labels.keys())
    a = DeepLearningPipeline.split_inner_train_val_participants(
        train_pids, labels, seed=99
    )
    b = DeepLearningPipeline.split_inner_train_val_participants(
        train_pids, labels, seed=99
    )
    assert a == b


def test_concat_participant_windows_keeps_participant_boundaries():
    labels = {"a": 0, "b": 1, "c": 2}
    participants = _participants_from_labels(labels, windows_per_pid=4)
    inner_train = ["a", "b"]
    inner_val = ["c"]
    X_tr, y_tr = DeepLearningPipeline.concat_participant_windows(inner_train, participants)
    X_va, y_va = DeepLearningPipeline.concat_participant_windows(inner_val, participants)
    assert X_tr.shape[0] == 8
    assert X_va.shape[0] == 4
    assert set(y_tr.tolist()) == {0, 1}
    assert set(y_va.tolist()) == {2}
