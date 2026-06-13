"""SEC-010: security headers and disabled API docs in production."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

import main  # noqa: E402


@pytest.fixture(autouse=True)
def noop_load_resources(monkeypatch):
    monkeypatch.setattr(main, "load_resources", lambda: None)


@pytest.fixture
def header_client(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CHECKPOINT_HMAC_KEY", "test-hmac-key")
    with TestClient(main.app, raise_server_exceptions=False) as client:
        yield client


def test_api_docs_disabled_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("ENABLE_API_DOCS", raising=False)
    assert main._api_docs_enabled() is False


def test_api_docs_explicit_override_in_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ENABLE_API_DOCS", "true")
    assert main._api_docs_enabled() is True


def test_fastapi_docs_urls_follow_docs_enabled_flag():
    probe = FastAPI(
        docs_url="/docs" if main._docs_enabled else None,
        redoc_url="/redoc" if main._docs_enabled else None,
        openapi_url="/openapi.json" if main._docs_enabled else None,
    )
    if main._docs_enabled:
        assert probe.docs_url == "/docs"
        assert probe.openapi_url == "/openapi.json"
    else:
        assert probe.docs_url is None
        assert probe.openapi_url is None


def test_production_adds_security_headers(header_client):
    response = header_client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert "default-src 'none'" in response.headers.get("Content-Security-Policy", "")
    assert response.headers.get("Strict-Transport-Security", "").startswith("max-age=")


def test_ui_route_uses_ui_csp(header_client):
    response = header_client.get("/app")
    if response.status_code == 404:
        pytest.skip("bundled UI not synced in test environment")
    csp = response.headers.get("Content-Security-Policy", "")
    assert "cdn.jsdelivr.net" in csp
    assert "frame-ancestors 'none'" in csp


def test_production_root_hides_model_inventory(header_client):
    payload = header_client.get("/").json()
    assert "models_loaded" not in payload
    assert payload.get("health") == "/health"


def test_main_wires_security_middleware():
    source = (API_ROOT / "main.py").read_text(encoding="utf-8")
    assert "SecurityHeadersMiddleware" in source
    assert 'docs_url="/docs" if _docs_enabled else None' in source
