"""Unit tests for leakage comparison resume cache."""

from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.leakage_resume import (
    build_leakage_resume_key,
    completed_model_names,
    get_cached_seed_auc,
    load_leakage_resume,
    save_leakage_resume,
    set_cached_seed_auc,
)


def test_seed_auc_roundtrip(tmp_path: Path):
    key = '{"test": true}'
    seed_aucs: dict[str, dict[str, float]] = {}
    set_cached_seed_auc(seed_aucs, "mlp", 0, 0.71)
    set_cached_seed_auc(seed_aucs, "mlp", 1, 0.72)
    rows = [{"model": "xgboost", "auc_inflation": 0.005}]
    save_leakage_resume(tmp_path, key, seed_aucs=seed_aucs, rows=rows)

    loaded = load_leakage_resume(tmp_path, key)
    assert get_cached_seed_auc(loaded["seed_aucs"], "mlp", 0) == 0.71
    assert get_cached_seed_auc(loaded["seed_aucs"], "mlp", 1) == 0.72
    assert get_cached_seed_auc(loaded["seed_aucs"], "mlp", 2) is None
    assert completed_model_names(loaded["rows"]) == {"xgboost"}


def test_resume_key_mismatch_is_ignored(tmp_path: Path):
    save_leakage_resume(
        tmp_path,
        "key-a",
        seed_aucs={"mlp": {"0": 0.5}},
        rows=[{"model": "mlp"}],
    )
    loaded = load_leakage_resume(tmp_path, "key-b")
    assert loaded["seed_aucs"] == {}
    assert loaded["rows"] == []


def test_build_leakage_resume_key_stable():
    cfg = {
        "reproducibility": {"seed": 42},
        "feature_selection": {
            "enabled": True,
            "nested_in_evaluation": True,
            "max_features": 20,
            "nested_fs_n_jobs": 14,
        },
        "models": {
            "evaluation": {
                "leakage_kfold_splits": 10,
                "random_state": 42,
                "leakage_kfold_seed_repeats": 5,
            }
        },
    }
    a = build_leakage_resume_key(cfg, ["f1", "f2"], n_samples=10, n_groups=10)
    b = build_leakage_resume_key(cfg, ["f1", "f2"], n_samples=10, n_groups=10)
    assert a == b
    # Parallelism-only change must not invalidate (nested_fs_n_jobs ignored in FS fp).
    cfg2 = json.loads(json.dumps(cfg))
    cfg2["feature_selection"]["nested_fs_n_jobs"] = 8
    c = build_leakage_resume_key(cfg2, ["f1", "f2"], n_samples=10, n_groups=10)
    assert a == c
