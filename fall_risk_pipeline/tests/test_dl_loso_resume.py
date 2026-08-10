"""Unit tests for deep-learning LOSO fold resume cache."""

from __future__ import annotations

from pathlib import Path

from src.models.dl_loso_resume import (
    build_dl_loso_fingerprint,
    count_cached_folds,
    dl_loso_model_cache_dir,
    load_dl_fold_result,
    save_dl_fold_result,
)


def test_fold_roundtrip(tmp_path: Path):
    cache = dl_loso_model_cache_dir(tmp_path, "abc123", "inception_time")
    save_dl_fold_result(
        cache,
        participant_id="PD_8",
        fold_idx=207,
        y_true=2,
        y_proba=[0.1, 0.2, 0.7],
        learning_rate=0.001,
    )
    loaded = load_dl_fold_result(cache, "PD_8")
    assert loaded is not None
    assert loaded["y_true"] == 2
    assert loaded["y_proba"] == [0.1, 0.2, 0.7]
    assert count_cached_folds(cache) == 1
    assert load_dl_fold_result(cache, "missing") is None


def test_fingerprint_stable_ignores_device():
    cfg = {
        "reproducibility": {"seed": 42},
        "dataset": {"sensor_positions": ["head", "lower_back"]},
        "deep_learning": {
            "sequence_length": 200,
            "overlap": 0.5,
            "learning_rate": 0.001,
            "max_epochs": 80,
            "device": "cuda",
            "models": ["inception_time", "tcn"],
        },
    }
    a = build_dl_loso_fingerprint(
        cfg,
        model_name="inception_time",
        participant_ids=["A", "B"],
        n_channels=12,
        n_windows_total=100,
    )
    cfg["deep_learning"]["device"] = "cpu"
    b = build_dl_loso_fingerprint(
        cfg,
        model_name="inception_time",
        participant_ids=["A", "B"],
        n_channels=12,
        n_windows_total=100,
    )
    assert a == b
    c = build_dl_loso_fingerprint(
        cfg,
        model_name="tcn",
        participant_ids=["A", "B"],
        n_channels=12,
        n_windows_total=100,
    )
    assert a != c


def test_deep_trainer_wires_resume():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "models" / "deep_trainer.py"
    ).read_text(encoding="utf-8")
    assert "save_dl_fold_result" in source
    assert "load_dl_fold_result" in source
    assert "resume cache hit" in source
