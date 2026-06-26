"""SEC-006: bundled UI uses same-origin /app/predict without client-side API keys."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from main import (  # noqa: E402
    API_KEY,
    _is_same_origin_ui_request,
    app,
)


class _FakeHeaders:
    def __init__(self, mapping: dict[str, str]):
        self._mapping = {k.lower(): v for k, v in mapping.items()}

    def get(self, key: str, default: str = "") -> str:
        return self._mapping.get(key.lower(), default)


class _FakeRequest:
    def __init__(self, headers: dict[str, str]):
        self.headers = _FakeHeaders(headers)


def test_same_origin_ui_request_accepts_matching_origin():
    req = _FakeRequest({"host": "gaitguard.example.com", "origin": "https://gaitguard.example.com"})
    assert _is_same_origin_ui_request(req) is True


def test_same_origin_ui_request_accepts_app_referer():
    req = _FakeRequest({
        "host": "gaitguard.example.com",
        "referer": "https://gaitguard.example.com/app",
    })
    assert _is_same_origin_ui_request(req) is True


def test_same_origin_ui_request_rejects_cross_origin():
    req = _FakeRequest({
        "host": "gaitguard.example.com",
        "origin": "https://evil.example.com",
    })
    assert _is_same_origin_ui_request(req) is False


def test_same_origin_ui_request_rejects_missing_origin_and_referer():
    req = _FakeRequest({"host": "gaitguard.example.com"})
    assert _is_same_origin_ui_request(req) is False


def test_app_predict_route_registered():
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/app/predict" in paths
    assert "/predict" in paths


def test_frontend_uses_app_predict_without_client_api_key():
    js = (Path(__file__).resolve().parents[1] / "static" / "main.js").read_text(
        encoding="utf-8"
    )
    assert "predictPathForBase" in js
    assert "/app/predict" in js
    fetch_block = js[js.find("async function postPredictWithFallback") : js.find("function upsertApiDebugLine")]
    assert "GAITGUARD_API_KEY" not in fetch_block
    assert "X-API-Key" not in fetch_block
    assert "Authorization" not in fetch_block


@pytest.mark.skipif(not API_KEY, reason="GAITGUARD_API_KEY unset in test env")
def test_predict_still_requires_api_key_when_set():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/predict", files=[])
    assert response.status_code == 401
