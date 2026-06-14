"""ML-042: DL LOSO hyperparameter protocol — fixed config vs per-fold Optuna LR."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEP_TRAINER_PATH = REPO_ROOT / "fall_risk_pipeline" / "src" / "models" / "deep_trainer.py"
METHODS_PATH = REPO_ROOT / "docs" / "paper" / "methods.md"


def _loso_evaluate_source() -> str:
    tree = ast.parse(DEEP_TRAINER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_loso_evaluate":
            return ast.get_source_segment(
                DEEP_TRAINER_PATH.read_text(encoding="utf-8"), node
            ) or ""
    raise AssertionError("_loso_evaluate not found")


def test_loso_evaluate_exports_hyperparameter_protocol():
    source = _loso_evaluate_source()
    assert "hyperparameter_protocol" in source
    assert "fixed_global_config" in source
    assert "loso_inner_participant_optuna_lr" in source


def test_loso_evaluate_invokes_tune_when_enabled():
    from src.models.deep_trainer import DeepLearningPipeline

    config = {
        "paths": {
            "processed_data": str(REPO_ROOT / "fall_risk_pipeline" / "data" / "processed"),
            "metrics": str(REPO_ROOT / "fall_risk_pipeline" / "metrics"),
            "checkpoints": str(REPO_ROOT / "fall_risk_pipeline" / "checkpoints"),
            "figures_models": str(REPO_ROOT / "fall_risk_pipeline" / "figures"),
        },
        "dataset": {"sensor_positions": ["head", "sternum", "l_foot", "r_foot"]},
        "deep_learning": {
            "sequence_length": 8,
            "overlap": 0.5,
            "batch_size": 4,
            "max_epochs": 2,
            "learning_rate": 0.001,
            "early_stopping_patience": 2,
            "device": "cpu",
            "mixed_precision": False,
            "loso_hyperparameter_tuning": {"enabled": True, "n_trials": 1, "search_epochs": 1},
            "models": ["cnn1d"],
        },
    }
    pipeline = DeepLearningPipeline(config)

    labels = {f"p{i}": i % 3 for i in range(6)}
    participants = {}
    for pid, label in labels.items():
        participants[pid] = {
            "windows": np.random.randn(5, 4, 8).astype(np.float32),
            "label": label,
            "trial_ids": [f"{pid}_t0"],
        }

    tune_calls: list[float] = []

    def fake_tune(*_args, **_kwargs):
        tune_calls.append(1.0)
        return 0.002

    mock_model = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = mock_model
    mock_trainer.predict_proba.return_value = np.full((5, 3), 1.0 / 3.0, dtype=np.float32)

    bar = MagicMock()

    with (
        patch.object(pipeline, "_loso_tune_enabled", return_value=True),
        patch.object(pipeline, "_tune_loso_fold_learning_rate", side_effect=fake_tune),
        patch.object(pipeline, "_make_trainer", return_value=mock_trainer),
        patch("src.models.deep_trainer.build_deep_model", return_value=mock_model),
        patch(
            "src.models.deep_trainer.build_multiclass_metric_payload",
            return_value={
                "auc": 0.5,
                "auc_ci_low": 0.4,
                "auc_ci_high": 0.6,
                "f1": 0.4,
                "accuracy": 0.4,
            },
        ),
    ):
        result = pipeline._loso_evaluate("cnn1d", participants, n_channels=4, bar=bar)

    assert len(tune_calls) == len(participants)
    assert result["hyperparameter_protocol"] == "loso_inner_participant_optuna_lr"
    assert result["learning_rate_tuned_median"] == 0.002


def test_loso_evaluate_skips_tune_when_disabled():
    from src.models.deep_trainer import DeepLearningPipeline

    config = {
        "paths": {
            "processed_data": str(REPO_ROOT / "fall_risk_pipeline" / "data" / "processed"),
            "metrics": str(REPO_ROOT / "fall_risk_pipeline" / "metrics"),
            "checkpoints": str(REPO_ROOT / "fall_risk_pipeline" / "checkpoints"),
            "figures_models": str(REPO_ROOT / "fall_risk_pipeline" / "figures"),
        },
        "dataset": {"sensor_positions": ["head", "sternum", "l_foot", "r_foot"]},
        "deep_learning": {
            "sequence_length": 8,
            "overlap": 0.5,
            "batch_size": 4,
            "max_epochs": 1,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
            "device": "cpu",
            "mixed_precision": False,
            "loso_hyperparameter_tuning": {"enabled": False},
            "models": ["cnn1d"],
        },
    }
    pipeline = DeepLearningPipeline(config)

    labels = {f"p{i}": i % 3 for i in range(4)}
    participants = {}
    for pid, label in labels.items():
        participants[pid] = {
            "windows": np.random.randn(5, 4, 8).astype(np.float32),
            "label": label,
            "trial_ids": [f"{pid}_t0"],
        }

    mock_model = MagicMock()
    mock_trainer = MagicMock()
    mock_trainer.train.return_value = mock_model
    mock_trainer.predict_proba.return_value = np.full((5, 3), 1.0 / 3.0, dtype=np.float32)
    bar = MagicMock()

    with (
        patch.object(pipeline, "_tune_loso_fold_learning_rate") as tune_mock,
        patch.object(pipeline, "_make_trainer", return_value=mock_trainer) as make_trainer,
        patch("src.models.deep_trainer.build_deep_model", return_value=mock_model),
        patch(
            "src.models.deep_trainer.build_multiclass_metric_payload",
            return_value={
                "auc": 0.5,
                "auc_ci_low": 0.4,
                "auc_ci_high": 0.6,
                "f1": 0.4,
                "accuracy": 0.4,
            },
        ),
    ):
        result = pipeline._loso_evaluate("cnn1d", participants, n_channels=4, bar=bar)

    tune_mock.assert_not_called()
    assert make_trainer.call_args.kwargs["learning_rate"] == 0.001
    assert result["hyperparameter_protocol"] == "fixed_global_config"
    assert result["learning_rate_tuned_median"] == 0.001


def test_pipeline_config_enables_loso_dl_tuning():
    import yaml

    cfg = yaml.safe_load(
        (REPO_ROOT / "fall_risk_pipeline" / "configs" / "pipeline_config.yaml").read_text(
            encoding="utf-8"
        )
    )
    tune = cfg["deep_learning"]["loso_hyperparameter_tuning"]
    assert tune["enabled"] is True
    assert int(tune["n_trials"]) >= 5
    assert int(tune["search_epochs"]) >= 12


def test_methods_document_dl_hyperparameter_protocol():
    text = METHODS_PATH.read_text(encoding="utf-8")
    assert "ML-042" in text or "hyperparameter_protocol" in text
    assert "loso_hyperparameter_tuning" in text or "loso_inner_participant_optuna_lr" in text
