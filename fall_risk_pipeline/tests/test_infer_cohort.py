"""Tests for path-token cohort inference (false-positive guards)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion.data_loader import DataLoader


@pytest.fixture
def loader(tmp_path) -> DataLoader:
    config = {
        "paths": {
            "raw_data": str(tmp_path / "raw"),
            "processed_data": str(tmp_path / "processed"),
        },
        "preprocessing": {"min_trial_length_s": 10.0},
        "dataset": {
            "sampling_rate": 100,
            "label_mode": "multiclass",
        },
    }
    return DataLoader(config)


def test_infer_cohort_pd_hyphenated_subject_id(loader: DataLoader):
    trial_dir = Path("raw/PD/group1/pd-001/PD_1_1")
    assert loader._infer_cohort(trial_dir) == "PD"


def test_infer_cohort_does_not_match_pd_inside_padded_gait(loader: DataLoader):
    trial_dir = Path("raw/Healthy/group1/padded_gait/HS_1_1")
    assert loader._infer_cohort(trial_dir) == "Healthy"


def test_infer_cohort_does_not_match_pd_inside_updated_trial(loader: DataLoader):
    trial_dir = Path("raw/Healthy/group1/updated_trial/HS_2_1")
    assert loader._infer_cohort(trial_dir) == "Healthy"


def test_infer_cohort_pd_period_delimited(loader: DataLoader):
    trial_dir = Path("raw/PD/group1/pd.001/PD_2_1")
    assert loader._infer_cohort(trial_dir) == "PD"
