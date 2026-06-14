"""MED-007: CORS startup warning when using localhost defaults."""

from __future__ import annotations

import logging

from main import _DEFAULT_LOCAL_CORS_ORIGINS, resolve_cors_origins


def test_unset_cors_uses_localhost_only(caplog):
    with caplog.at_level(logging.WARNING):
        origins, allow_all = resolve_cors_origins(None)
    assert allow_all is False
    assert origins == list(_DEFAULT_LOCAL_CORS_ORIGINS)
    assert any("CORS_ORIGINS" in rec.message for rec in caplog.records)


def test_wildcard_cors_rejected():
    origins, allow_all = resolve_cors_origins("*")
    assert allow_all is False
    assert origins == list(_DEFAULT_LOCAL_CORS_ORIGINS)


def test_explicit_production_origins():
    origins, allow_all = resolve_cors_origins(
        "https://app.example.com,https://staging.example.com"
    )
    assert allow_all is False
    assert "https://app.example.com" in origins
    assert "https://staging.example.com" in origins
