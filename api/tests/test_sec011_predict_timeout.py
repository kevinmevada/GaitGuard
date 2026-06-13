"""SEC-011: bounded /predict concurrency and inference timeout."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from main import (  # noqa: E402
    PREDICT_MAX_CONCURRENT,
    PREDICT_REQUEST_TIMEOUT_SEC,
    _execute_predict,
)


def test_predict_concurrency_clamped_to_2_4():
    assert 2 <= PREDICT_MAX_CONCURRENT <= 4


def test_predict_request_timeout_minimum():
    assert PREDICT_REQUEST_TIMEOUT_SEC >= 10


def test_execute_predict_uses_wait_for_timeout():
    source = (API_ROOT / "main.py").read_text(encoding="utf-8")
    assert "asyncio.wait_for" in source
    assert "PREDICT_REQUEST_TIMEOUT_SEC" in source
    assert "asyncio.TimeoutError" in source
    assert "status_code=504" in source
    assert "PREDICT_MAX_CONCURRENT = min(4, max(2" in source


def test_execute_predict_timeout_returns_504(monkeypatch):
    import main as api_main

    def slow_pipeline(*_args, **_kwargs):
        import time

        time.sleep(0.2)
        return {"success": True}

    monkeypatch.setattr(api_main, "PREDICT_REQUEST_TIMEOUT_SEC", 0)
    monkeypatch.setattr(api_main, "parse_uploaded_files", lambda _files: ({}, {}))
    monkeypatch.setattr(api_main, "_run_prediction_pipeline", slow_pipeline)

    async def _run() -> None:
        with pytest.raises(HTTPException) as exc:
            await _execute_predict([])
        assert exc.value.status_code == 504

    asyncio.run(_run())
