"""Regression checks for low-severity audit fixes (ML-030, ML-031, DEP-004)."""

from __future__ import annotations

import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_deep_trainer_docstring_describes_soft_vote_not_majority():
    from src.models import deep_trainer

    doc = inspect.getmodule(deep_trainer).__doc__ or ""
    assert "mean class probability" in doc.lower() or "mean window" in doc.lower()
    assert "majority-vote" not in doc.lower()


def test_data_loader_warns_on_missing_sensors():
    source = (REPO_ROOT / "fall_risk_pipeline" / "src" / "ingestion" / "data_loader.py").read_text(
        encoding="utf-8"
    )
    assert "missing sensors" in source
    assert "missing_sensors" in source


def test_download_models_includes_mlp():
    source = (REPO_ROOT / "scripts" / "download_models.py").read_text(encoding="utf-8")
    assert "checkpoints/mlp.pkl" in source


def test_model_card_documents_mlp():
    model_card = (REPO_ROOT / "docs" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "mlp.pkl" in model_card


def test_api_upload_does_not_trust_upload_file_size_precheck():
    source = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    assert "do not trust UploadFile.size" in source or "_bounded_read" in source
    # Removed misleading aggregate pre-check on spoofable f.size
    assert "total_size = sum(f.size" not in source


def test_frontend_three_modulepreload_sri():
    html = (REPO_ROOT / "api" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'integrity="sha384-' in html
    assert "three.module.js" in html


def test_frontend_api_port_candidates_include_8000():
    js = (REPO_ROOT / "api" / "static" / "main.js").read_text(encoding="utf-8")
    assert "localhost:8000" in js
    assert "window.location.origin" in js
