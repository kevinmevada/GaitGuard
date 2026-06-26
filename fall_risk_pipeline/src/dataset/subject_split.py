"""
Subject-grouped train/val/test splits (no trial-level leakage).

Trials are always assigned via ``participant_id`` (subject string key). A static
Healthy holdout (default 70% / 15% / 15%) places all pathological participants in
test only. LOSO evaluation asserts disjoint train/test subject sets per fold.

Contrast: Klaver et al. (2023) randomized trials without grouping by subject,
inflating AUC when the same participant appeared in train and test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from src.utils.reproducibility import get_pipeline_seed

HEALTHY_COHORT = "Healthy"
LEAKAGE_MSG = "DATA LEAKAGE: subject in both splits"


@dataclass(frozen=True)
class SubjectHoldoutSplit:
    """Participant-level holdout: Healthy 70/15/15; pathological → test only."""

    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]

    def assert_disjoint(self) -> None:
        assert_no_subject_leakage(self.train_ids, self.test_ids, self.val_ids)


def assert_no_subject_leakage(
    train_ids: Iterable[str],
    test_ids: Iterable[str],
    val_ids: Iterable[str] | None = None,
    *,
    context: str = "",
) -> None:
    """Hard stop if any participant_id appears in more than one split role."""
    train_set = {str(s) for s in train_ids}
    test_set = {str(s) for s in test_ids}
    overlap = train_set & test_set
    prefix = f"{context}: " if context else ""
    assert not overlap, f"{prefix}{LEAKAGE_MSG}: {sorted(overlap)[:8]}"
    if val_ids is not None:
        val_set = {str(s) for s in val_ids}
        overlap_tv = train_set & val_set
        overlap_vt = val_set & test_set
        assert not overlap_tv, f"{prefix}{LEAKAGE_MSG} (train∩val): {sorted(overlap_tv)[:8]}"
        assert not overlap_vt, f"{prefix}{LEAKAGE_MSG} (val∩test): {sorted(overlap_vt)[:8]}"


def assert_loso_fold_disjoint(
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    *,
    held_out_subject: str | None = None,
    context: str = "LOSO fold",
) -> None:
    """Assert a LOSO fold keeps all trials of each subject in one split only."""
    train_ids = np.unique(train_groups)
    test_ids = np.unique(test_groups)
    assert_fold_subject_disjoint(train_ids, test_ids, context=context)
    if held_out_subject is not None:
        held = str(held_out_subject)
        test_set = {str(s) for s in test_ids}
        train_set = {str(s) for s in train_ids}
        assert held in test_set, f"{context}: held-out subject {held!r} missing from test"
        assert held not in train_set, f"{context}: held-out subject {held!r} leaked into train"


def assert_fold_subject_disjoint(
    train_ids: Iterable[str],
    test_ids: Iterable[str],
    *,
    context: str = "fold",
) -> None:
    assert_no_subject_leakage(train_ids, test_ids, context=context)


def _split_cfg(config: dict) -> dict:
    return (config.get("dataset") or {}).get("subject_split") or {}


def build_holdout_from_participants(
    participants: pd.DataFrame,
    config: dict,
    *,
    participant_col: str = "participant_id",
    cohort_col: str = "cohort",
) -> SubjectHoldoutSplit:
    """
    Partition unique participants: Healthy → train/val/test fractions;
    all non-Healthy → test only.
    """
    if participant_col not in participants.columns:
        raise KeyError(f"Missing {participant_col!r} in participant table")
    if cohort_col not in participants.columns:
        raise KeyError(f"Missing {cohort_col!r} in participant table")

    subjects = participants[[participant_col, cohort_col]].drop_duplicates(participant_col)
    subjects[participant_col] = subjects[participant_col].astype(str)
    healthy_ids = sorted(
        subjects.loc[subjects[cohort_col] == HEALTHY_COHORT, participant_col].tolist()
    )
    patho_ids = sorted(
        subjects.loc[subjects[cohort_col] != HEALTHY_COHORT, participant_col].tolist()
    )

    scfg = _split_cfg(config)
    seed = int(scfg.get("random_state", get_pipeline_seed(config)))
    test_frac = float(scfg.get("healthy_test_fraction", 0.15))
    val_frac = float(scfg.get("healthy_val_fraction", 0.15))
    if test_frac <= 0 or val_frac <= 0 or test_frac + val_frac >= 1.0:
        raise ValueError(
            f"Invalid healthy fractions: test={test_frac}, val={val_frac} "
            "(must be > 0 and sum < 1)"
        )

    if len(healthy_ids) == 0:
        train_ids: list[str] = []
        val_ids: list[str] = []
        test_hs: list[str] = []
    elif len(healthy_ids) == 1:
        train_ids, val_ids, test_hs = healthy_ids, [], []
    else:
        train_val, test_hs = train_test_split(
            healthy_ids,
            test_size=test_frac,
            random_state=seed,
        )
        # 15% of total from remaining 85% → val fraction of train_val
        inner_val_frac = val_frac / (1.0 - test_frac)
        if len(train_val) <= 1:
            train_ids, val_ids = list(train_val), []
        else:
            train_ids, val_ids = train_test_split(
                train_val,
                test_size=inner_val_frac,
                random_state=seed,
            )
            train_ids, val_ids = list(train_ids), list(val_ids)

    test_ids = sorted(set(test_hs) | set(patho_ids))
    split = SubjectHoldoutSplit(
        train_ids=tuple(sorted(train_ids)),
        val_ids=tuple(sorted(val_ids)),
        test_ids=tuple(test_ids),
    )
    split.assert_disjoint()
    return split


def trial_masks_for_split(
    groups: np.ndarray,
    split: SubjectHoldoutSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks over trial rows aligned to ``groups`` (participant_id per row)."""
    groups = np.asarray(groups).astype(str)
    train_mask = np.isin(groups, list(split.train_ids))
    val_mask = np.isin(groups, list(split.val_ids))
    test_mask = np.isin(groups, list(split.test_ids))
    assert not np.any(train_mask & test_mask), LEAKAGE_MSG
    assert not np.any(train_mask & val_mask), LEAKAGE_MSG
    assert not np.any(val_mask & test_mask), LEAKAGE_MSG
    return train_mask, val_mask, test_mask


def export_subject_split_manifest(
    split: SubjectHoldoutSplit,
    participants: pd.DataFrame,
    metrics_dir: Path,
    *,
    participant_col: str = "participant_id",
    cohort_col: str = "cohort",
) -> Path:
    """Write ``subject_split.csv`` and ``subject_split.json`` for Methods / audit."""
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    subjects = participants[[participant_col, cohort_col]].drop_duplicates(participant_col)
    subjects[participant_col] = subjects[participant_col].astype(str)

    def _role(pid: str) -> str:
        if pid in split.train_ids:
            return "train"
        if pid in split.val_ids:
            return "val"
        if pid in split.test_ids:
            return "test"
        return "unassigned"

    subjects = subjects.copy()
    subjects["split_role"] = subjects[participant_col].map(_role)
    csv_path = metrics_dir / "subject_split.csv"
    subjects.to_csv(csv_path, index=False)

    summary = {
        "protocol": "subject_grouped_holdout",
        "healthy_fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
        "pathological_split": "test_only",
        "n_train": len(split.train_ids),
        "n_val": len(split.val_ids),
        "n_test": len(split.test_ids),
        "train_ids": list(split.train_ids),
        "val_ids": list(split.val_ids),
        "test_ids": list(split.test_ids),
        "klaver_2023_contrast": (
            "Trials grouped by participant_id before split; no subject in both "
            "train and test (unlike trial-randomized splits criticized in review)."
        ),
    }
    json_path = metrics_dir / "subject_split.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Subject split manifest: train={} val={} test={} → {}",
        len(split.train_ids),
        len(split.val_ids),
        len(split.test_ids),
        csv_path,
    )
    return csv_path


def ensure_subject_split_manifest(config: dict, metrics_dir: Path | None = None) -> SubjectHoldoutSplit:
    """Load participant table, build split, export manifest, assert disjoint."""
    from src.features.feature_matrix import load_patient_feature_matrix

    _, _, _, _, patient_df = load_patient_feature_matrix(config)
    split = build_holdout_from_participants(patient_df, config)
    out_dir = Path(metrics_dir or config["paths"]["metrics"])
    export_subject_split_manifest(split, patient_df, out_dir)
    split.assert_disjoint()
    return split
