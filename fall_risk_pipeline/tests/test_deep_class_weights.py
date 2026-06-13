"""Tests for participant-level deep learning class weights (ML-009)."""

import numpy as np

from src.models.deep_models import (
    participant_balanced_ce_class_weights,
    window_level_ce_class_weights,
)


def test_participant_weights_ignore_window_duplication():
    # Class 0: 2 participants × 50 windows; class 1: 1 participant × 5 windows.
    y = np.array([0] * 100 + [1] * 5, dtype=int)
    pids = np.array(["p0"] * 50 + ["p1"] * 50 + ["p2"] * 5, dtype=object)

    participant_w = participant_balanced_ce_class_weights(y, pids, n_classes=2)
    window_w = window_level_ce_class_weights(y, n_classes=2)

    # Participant balancing: minority class (1 participant) gets higher per-class weight.
    assert participant_w[1] > participant_w[0]

    # Window weights also upweight the minority class by window count.
    assert window_w[1] > window_w[0]

    # Duplication problem: total weighted mass still favors class 0 despite 2:1 participants.
    participant_mass_0 = int((y == 0).sum()) * float(participant_w[0])
    participant_mass_1 = int((y == 1).sum()) * float(participant_w[1])
    assert participant_mass_0 > participant_mass_1 * 2

    # Window-level inverse-frequency normalizes total mass across classes (~equal).
    window_mass_0 = int((y == 0).sum()) * float(window_w[0])
    window_mass_1 = int((y == 1).sum()) * float(window_w[1])
    assert abs(window_mass_0 - window_mass_1) < 0.2 * max(window_mass_0, window_mass_1)


def test_participant_weights_match_sklearn_style_formula():
    y = np.array([0, 0, 1, 2, 2], dtype=int)
    pids = np.array(["a", "b", "c", "d", "e"], dtype=object)
    weights = participant_balanced_ce_class_weights(y, pids, n_classes=3)
    # N=5, n_classes=3, counts=[2,1,2]
    expected = np.array([5 / (3 * 2), 5 / (3 * 1), 5 / (3 * 2)], dtype=np.float32)
    np.testing.assert_allclose(weights, expected, rtol=1e-5)


def test_window_participant_ids_align_with_concat():
    from src.models.deep_trainer import DeepLearningPipeline

    participants = {
        "a": {"windows": np.zeros((3, 2, 4)), "label": 0},
        "b": {"windows": np.zeros((2, 2, 4)), "label": 1},
    }
    X, y = DeepLearningPipeline.concat_participant_windows(["a", "b"], participants)
    pids = DeepLearningPipeline.window_participant_ids(["a", "b"], participants)
    assert len(X) == len(y) == len(pids) == 5
    assert list(pids[:3]) == ["a", "a", "a"]
    assert list(pids[3:]) == ["b", "b"]
