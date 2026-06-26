"""Subject-grouped splits: no participant leakage across train/val/test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dataset.subject_split import (
    LEAKAGE_MSG,
    SubjectHoldoutSplit,
    assert_loso_fold_disjoint,
    assert_no_subject_leakage,
    build_holdout_from_participants,
    trial_masks_for_split,
)


def _participants(n_healthy: int = 20, n_patho: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(1, n_healthy + 1):
        rows.append({"participant_id": f"HS_{i}", "cohort": "Healthy"})
    for i in range(1, n_patho + 1):
        rows.append({"participant_id": f"PD_{i}", "cohort": "PD"})
    return pd.DataFrame(rows)


def test_holdout_disjoint_and_patho_in_test():
    config = {
        "reproducibility": {"seed": 42},
        "dataset": {
            "subject_split": {
                "healthy_train_fraction": 0.70,
                "healthy_val_fraction": 0.15,
                "healthy_test_fraction": 0.15,
                "random_state": 42,
            }
        },
    }
    split = build_holdout_from_participants(_participants(20, 8), config)
    split.assert_disjoint()
    assert all(pid.startswith("PD_") for pid in split.test_ids if pid.startswith("PD_"))
    patho_in_test = [p for p in split.test_ids if p.startswith("PD_")]
    assert len(patho_in_test) == 8
    assert len(split.train_ids) + len(split.val_ids) + len(
        [p for p in split.test_ids if p.startswith("HS_")]
    ) == 20


def test_assert_no_subject_leakage_raises():
    with pytest.raises(AssertionError, match=LEAKAGE_MSG):
        assert_no_subject_leakage(["S1", "S2"], ["S2", "S3"])
    with pytest.raises(AssertionError, match=LEAKAGE_MSG):
        assert_no_subject_leakage(["S1"], ["S2"], val_ids=["S1"])


def test_loso_fold_disjoint():
    groups = np.array(["A", "A", "B", "B", "C", "C"])
    train_mask = groups != "B"
    test_mask = groups == "B"
    assert_loso_fold_disjoint(
        groups[train_mask],
        groups[test_mask],
        held_out_subject="B",
        context="test",
    )


def test_trial_masks_no_overlap():
    split = SubjectHoldoutSplit(
        train_ids=("HS_1", "HS_2"),
        val_ids=("HS_3",),
        test_ids=("HS_4", "PD_1"),
    )
    groups = np.array(["HS_1", "HS_1", "HS_3", "HS_4", "PD_1"])
    train_m, val_m, test_m = trial_masks_for_split(groups, split)
    assert train_m.sum() == 2
    assert val_m.sum() == 1
    assert test_m.sum() == 2
    assert not np.any(train_m & test_m)


def test_holdout_reproducible():
    config = {"reproducibility": {"seed": 0}, "dataset": {"subject_split": {"random_state": 0}}}
    df = _participants(60, 20)
    a = build_holdout_from_participants(df, config)
    b = build_holdout_from_participants(df, config)
    assert a == b
