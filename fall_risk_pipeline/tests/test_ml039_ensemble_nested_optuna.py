"""ML-039: ensemble nested CV must re-tune base models per outer fold."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = REPO_ROOT / "fall_risk_pipeline" / "src" / "models" / "trainer.py"


def _nested_ensemble_cv_source() -> str:
    tree = ast.parse(TRAINER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_nested_ensemble_cv":
            return ast.get_source_segment(
                TRAINER_PATH.read_text(encoding="utf-8"), node
            ) or ""
    raise AssertionError("_nested_ensemble_cv not found")


def test_nested_ensemble_cv_runs_optuna_per_outer_fold():
    source = _nested_ensemble_cv_source()
    assert "_run_optuna" in source
    assert 'res["params"]' not in source
    assert "results[name]" not in source


def test_nested_ensemble_cv_invokes_optuna_once_per_base_per_fold():
    from src.models.trainer import ModelTrainer

    config = {
        "paths": {
            "features": str(REPO_ROOT / "fall_risk_pipeline" / "data" / "features"),
            "checkpoints": str(REPO_ROOT / "fall_risk_pipeline" / "checkpoints"),
            "metrics": str(REPO_ROOT / "fall_risk_pipeline" / "metrics"),
        },
        "models": {
            "tuning": {"n_trials": 1, "timeout_per_model": 1, "cv_folds": 2},
            "evaluation": {"random_state": 42},
            "ensemble": {"stacking": {"cv_folds": 2}},
        },
        "dataset": {"label_mode": "binary"},
    }
    trainer = ModelTrainer(config)
    trainer.cv_folds = 2
    trainer.n_trials = 1
    trainer.timeout = 1

    rng = np.random.default_rng(42)
    n = 24
    X_full = rng.normal(size=(n, 4))
    y = np.array([0, 0, 0, 1, 1, 1] * 4)
    groups = np.repeat(np.arange(6), 4)

    top_models = [("m_a", {}), ("m_b", {})]
    optuna_calls: list[str] = []

    def fake_optuna(name, X, y_train, groups_train, n_trials, timeout):
        optuna_calls.append(name)
        return {"n_estimators": 10}, 0.5

    def fake_build(name, params, y_train):
        pipe = MagicMock()
        pipe.predict_proba.return_value = np.column_stack(
            [np.full(len(y_train), 0.4), np.full(len(y_train), 0.6)]
        )
        return pipe

    def fake_ensemble(fold_bases, method, cv_folds, random_state):
        ens = MagicMock()
        ens.predict_proba.return_value = np.column_stack(
            [np.full(6, 0.4), np.full(6, 0.6)]
        )
        return ens

    with (
        patch.object(trainer, "_run_optuna", side_effect=fake_optuna),
        patch.object(trainer, "_build_pipeline_from_params", side_effect=fake_build),
        patch.object(trainer, "fit_pipeline", side_effect=lambda *a, **k: a[1]),
        patch(
            "src.models.trainer.nested_rfecv_column_indices",
            return_value=np.arange(X_full.shape[1]),
        ),
        patch("src.models.trainer.build_ensemble_estimator", side_effect=fake_ensemble),
        patch.object(trainer, "_roc_auc_from_proba", return_value=0.55),
    ):
        trainer._nested_ensemble_cv(
            "soft_voting",
            top_models,
            {},
            X_full,
            y,
            groups,
            [f"f{i}" for i in range(X_full.shape[1])],
        )

    # 2 outer folds × 2 base models
    assert optuna_calls == ["m_a", "m_b", "m_a", "m_b"]
