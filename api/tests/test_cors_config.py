"""Tests for API CORS origin resolution."""

from __future__ import annotations

from main import _DEFAULT_LOCAL_CORS_ORIGINS, resolve_cors_origins


def test_unset_cors_uses_localhost_only():
    origins, allow_all = resolve_cors_origins(None)
    assert allow_all is False
    assert origins == list(_DEFAULT_LOCAL_CORS_ORIGINS)


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
