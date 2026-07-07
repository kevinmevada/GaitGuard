"""SEC-013: load_dotenv must not run in production deployments."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

API_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = API_ROOT / "main.py"
sys.path.insert(0, str(API_ROOT))

import main  # noqa: E402


def test_main_source_gates_dotenv_on_production():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "_load_local_dotenv" in source
    assert "_is_production_deployment()" in source
    assert source.index("_load_local_dotenv") < source.index("FastAPI(")


def test_load_local_dotenv_skips_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    mock_load = MagicMock()
    monkeypatch.setitem(sys.modules, "dotenv", MagicMock(load_dotenv=mock_load))
    main._load_local_dotenv()
    mock_load.assert_not_called()


def test_load_local_dotenv_runs_in_development(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    mock_load = MagicMock()
    monkeypatch.setitem(sys.modules, "dotenv", MagicMock(load_dotenv=mock_load))
    main._load_local_dotenv()
    mock_load.assert_called_once()


def test_load_local_dotenv_skips_when_env_is_prod(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "prod")
    mock_load = MagicMock()
    monkeypatch.setitem(sys.modules, "dotenv", MagicMock(load_dotenv=mock_load))
    main._load_local_dotenv()
    mock_load.assert_not_called()
