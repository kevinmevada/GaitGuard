"""SEC-016/017: predict path and production rate-limit guards."""

from __future__ import annotations

from pathlib import Path

API_MAIN = Path(__file__).resolve().parents[1] / "main.py"


def test_execute_predict_parses_in_thread_before_semaphore():
    source = API_MAIN.read_text(encoding="utf-8")
    predict_block = source.split("async def _execute_predict")[1].split("\nasync def ")[0]
    parse_idx = predict_block.find("asyncio.to_thread(parse_uploaded_files")
    sem_idx = predict_block.find("async with _predict_concurrency_slot()")
    assert parse_idx != -1 and sem_idx != -1
    assert parse_idx < sem_idx


def test_production_requires_slowapi():
    source = API_MAIN.read_text(encoding="utf-8")
    assert "SEC-017" in source
    assert "RATE_LIMITING_AVAILABLE" in source
    assert "slowapi must be installed" in source
