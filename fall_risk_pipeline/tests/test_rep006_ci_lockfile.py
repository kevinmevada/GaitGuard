"""REP-006: CI must install pinned lockfiles, not floating requirements.txt."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"
PIPELINE_LOCK = REPO_ROOT / "fall_risk_pipeline" / "requirements-lock.txt"
DEV_LOCK = REPO_ROOT / "fall_risk_pipeline" / "requirements-dev-lock.txt"
API_LOCK = REPO_ROOT / "api" / "requirements-lock.txt"


def test_ci_lockfiles_exist():
    assert PIPELINE_LOCK.is_file()
    assert DEV_LOCK.is_file()
    assert API_LOCK.is_file()


def test_dev_lock_pins_pytest():
    text = DEV_LOCK.read_text(encoding="utf-8")
    assert "pytest==" in text
    assert "pytest-cov==" in text


def test_api_lock_pins_fastapi_stack():
    text = API_LOCK.read_text(encoding="utf-8").lower()
    for pkg in ("fastapi==", "uvicorn", "python-multipart==", "slowapi=="):
        assert pkg in text


def test_ci_uses_extra_index_for_cpu_torch():
    assert "extra-index-url" in CI.read_text(encoding="utf-8")
