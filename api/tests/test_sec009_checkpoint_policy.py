"""SEC-009: API startup enforces checkpoint HMAC in production."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

import main
from main import load_resources  # noqa: E402


def test_load_resources_requires_hmac_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setattr(main, "API_KEY", "test-api-key")
    monkeypatch.delenv("CHECKPOINT_HMAC_KEY", raising=False)
    with pytest.raises(RuntimeError, match="SEC-009"):
        load_resources()
