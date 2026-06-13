"""ML-045: predictions must load checkpoints via manifest-verified load_checkpoint."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
PREDICTIONS_PATH = PIPELINE_ROOT / "src" / "evaluation" / "predictions.py"


@pytest.fixture
def simple_model():
    return LogisticRegression().fit([[0.0], [1.0], [2.0]], [0, 0, 1])


def test_predictions_source_uses_load_checkpoint_not_pickle():
    source = PREDICTIONS_PATH.read_text(encoding="utf-8")
    assert "load_checkpoint" in source
    assert "pickle.load" not in source
    assert "require_manifest=True" in source


def test_load_fitted_model_verifies_manifest(tmp_path, simple_model, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.predictions import PredictionGenerator
    from src.utils.checkpoint_io import save_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    path = ckpt_dir / "ensemble.pkl"
    save_checkpoint(path, simple_model, manifest_dir=ckpt_dir)

    gen = PredictionGenerator(
        {
            "paths": {
                "features": str(tmp_path / "features"),
                "checkpoints": str(ckpt_dir),
                "metrics": str(tmp_path / "metrics"),
            }
        }
    )
    loaded = gen._load_fitted_model(path)
    assert loaded is not None
    assert hasattr(loaded, "predict")


def test_load_fitted_model_rejects_unmanifested_pkl(tmp_path, simple_model, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.predictions import PredictionGenerator

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    path = ckpt_dir / "rogue.pkl"
    joblib.dump(simple_model, path)

    gen = PredictionGenerator(
        {
            "paths": {
                "features": str(tmp_path / "features"),
                "checkpoints": str(ckpt_dir),
                "metrics": str(tmp_path / "metrics"),
            }
        }
    )
    assert gen._load_fitted_model(path) is None


def test_load_fitted_model_rejects_tampered_checkpoint(tmp_path, simple_model, monkeypatch):
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.predictions import PredictionGenerator
    from src.utils.checkpoint_io import save_checkpoint

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    path = ckpt_dir / "ensemble.pkl"
    save_checkpoint(path, simple_model, manifest_dir=ckpt_dir)
    path.write_bytes(path.read_bytes() + b"tamper")

    gen = PredictionGenerator(
        {
            "paths": {
                "features": str(tmp_path / "features"),
                "checkpoints": str(ckpt_dir),
                "metrics": str(tmp_path / "metrics"),
            }
        }
    )
    assert gen._load_fitted_model(path) is None
