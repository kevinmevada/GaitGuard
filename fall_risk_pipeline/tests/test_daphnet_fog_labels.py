"""DAPHNET FOG label mapping and sealed eval tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion.daphnet_label_mapping import (
    DaphnetLabelError,
    annotations_to_y_true,
    align_labels_to_resampled_length,
    assert_labels_not_in_feature_columns,
    load_fog_labels_npz,
    save_fog_labels_npz,
)
from src.evaluation.daphnet_fog_evaluator import (
    DaphnetSealedEvalError,
    run_daphnet_sealed_fog_eval,
    score_lb_freezing_index,
    score_lb_samples_ensemble,
)


def test_annotation_mapping():
    ann = np.array([1, 1, 2, 2, 1])
    y = annotations_to_y_true(ann)
    assert list(y) == [0, 0, 1, 1, 0]


def test_align_labels_to_resampled_length():
    ann = np.array([1, 1, 2, 2])
    y = align_labels_to_resampled_length(ann, 8)
    assert len(y) == 8
    assert set(y) <= {0, 1}


def test_save_load_fog_labels_roundtrip(tmp_path: Path):
    labels = {"S01": np.array([0, 0, 1, 1], dtype=np.int8)}
    path = save_fog_labels_npz(labels, tmp_path / "fog_labels.npz")
    loaded = load_fog_labels_npz(path)
    assert np.array_equal(loaded["S01"], labels["S01"])


def test_forbidden_feature_columns():
    with pytest.raises(DaphnetLabelError):
        assert_labels_not_in_feature_columns(["acc_x", "annotation"])


def test_sealed_eval_writes_once(tmp_path: Path):
    processed = tmp_path / "processed"
    signals = processed / "signals"
    signals.mkdir(parents=True)
    daphnet_dir = processed / "daphnet"
    daphnet_dir.mkdir(parents=True)

    # Healthy Voisard reference
    meta = pd.DataFrame(
        {
            "trial_id": ["HS_1_1", "daphnet_S01"],
            "cohort": ["Healthy", "PD"],
            "participant_id": ["HS_1", "S01"],
        }
    )
    meta.to_csv(processed / "trial_metadata.csv", index=False)
    rng = np.random.default_rng(0)
    healthy = pd.DataFrame(
        {
            "time": np.arange(500) / 100.0,
            "acc_x": rng.normal(size=500),
            "acc_y": rng.normal(size=500),
            "acc_z": rng.normal(size=500),
        }
    )
    healthy.to_parquet(signals / "HS_1_1_lower_back.parquet", index=False)

    n = 400
    fog = np.zeros(n, dtype=np.int8)
    fog[200:] = 1
    np.savez_compressed(
        daphnet_dir / "fog_labels.npz",
        subject_ids=np.array(["S01"]),
        S01=fog,
    )
    daph = pd.DataFrame(
        {
            "time": np.arange(n) / 100.0,
            "acc_x": rng.normal(size=n),
            "acc_y": rng.normal(size=n),
            "acc_z": np.concatenate([rng.normal(size=200), rng.normal(size=200) + 3.0]),
        }
    )
    daph.to_parquet(signals / "daphnet_S01_lower_back.parquet", index=False)

    config = {
        "reproducibility": {"seed": 42},
        "paths": {
            "processed_data": str(processed),
            "metrics": str(tmp_path / "metrics"),
        },
        "ingestion": {"daphnet": {"sealed_fog_eval": {"enabled": True, "allow_rerun": False}}},
    }

    result = run_daphnet_sealed_fog_eval(config)
    assert "auc" in result
    assert (tmp_path / "metrics" / "daphnet_fog_auroc.json").is_file()
    assert (tmp_path / "metrics" / "daphnet_fog_fi_auroc.json").is_file()
    assert "comparison_freezing_index_detector" in result

    with pytest.raises(DaphnetSealedEvalError):
        run_daphnet_sealed_fog_eval(config)


def test_score_lb_freezing_index_shape():
    fs = 100.0
    sig = np.random.default_rng(2).normal(size=120)
    scores = score_lb_freezing_index(sig, fs, window_s=1.0)
    assert scores.shape == sig.shape


def test_score_lb_ensemble_shape():
    rng = np.random.default_rng(1)
    Xh = rng.normal(size=(200, 3))
    Xq = rng.normal(size=(50, 3))
    scores = score_lb_samples_ensemble(Xh, Xq, random_state=42)
    assert scores.shape == (50,)
