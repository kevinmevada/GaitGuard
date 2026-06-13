"""SEC-008: rate-limit keys and inference concurrency behind reverse proxies."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from main import (  # noqa: E402
    PREDICT_MAX_CONCURRENT,
    _predict_semaphore,
    _trust_proxy_headers,
    rate_limit_client_key,
)


class _FakeHeaders:
    def __init__(self, mapping: dict[str, str]):
        self._mapping = {k.lower(): v for k, v in mapping.items()}

    def get(self, key: str, default: str = "") -> str:
        return self._mapping.get(key.lower(), default)


class _FakeClient:
    def __init__(self, host: str):
        self.host = host


class _FakeRequest:
    def __init__(self, headers: dict[str, str], client_host: str = "10.0.0.1"):
        self.headers = _FakeHeaders(headers)
        self.client = _FakeClient(client_host)


def test_trust_proxy_headers_explicit(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    assert _trust_proxy_headers() is True
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "false")
    assert _trust_proxy_headers() is False


def test_trust_proxy_headers_auto_in_production(monkeypatch):
    monkeypatch.delenv("TRUST_PROXY_HEADERS", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert _trust_proxy_headers() is True


def test_rate_limit_client_key_uses_forwarded_for_when_trusted(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "true")
    req = _FakeRequest(
        {"X-Forwarded-For": "203.0.113.10, 10.0.0.1"},
        client_host="10.0.0.1",
    )
    assert rate_limit_client_key(req) == "203.0.113.10"


def test_rate_limit_client_key_ignores_forwarded_for_when_untrusted(monkeypatch):
    monkeypatch.setenv("TRUST_PROXY_HEADERS", "false")
    monkeypatch.setenv("ENVIRONMENT", "development")
    req = _FakeRequest(
        {"X-Forwarded-For": "203.0.113.10, 10.0.0.1"},
        client_host="10.0.0.1",
    )
    assert rate_limit_client_key(req) == "10.0.0.1"


def test_predict_routes_rate_limited():
    source = (API_ROOT / "main.py").read_text(encoding="utf-8")
    assert "rate_limit_client_key" in source
    assert 'path in ("/predict", "/app/predict")' in source
    assert "PREDICT_RATE_LIMIT" in source
    assert "break" not in source.split("if RATE_LIMITING_AVAILABLE and limiter:")[-1].split("Entrypoint")[0]


def test_predict_concurrency_slot_releases_semaphore():
    from main import _predict_concurrency_slot

    async def _run() -> None:
        while _predict_semaphore._value < PREDICT_MAX_CONCURRENT:
            _predict_semaphore.release()
        before = _predict_semaphore._value
        async with _predict_concurrency_slot():
            assert _predict_semaphore._value == before - 1
        assert _predict_semaphore._value == before

    asyncio.run(_run())
