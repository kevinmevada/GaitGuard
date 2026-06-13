"""REP-010: local/Docker installs use lockfiles and PYTHONHASHSEED."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_makefile_uses_lockfiles_and_hash_seed():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "requirements-lock.txt" in makefile
    assert "PYTHONHASHSEED" in makefile
    assert "fall_risk_pipeline/requirements.txt" not in makefile


def test_dockerfile_uses_lockfiles():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "requirements-lock.txt" in dockerfile
    assert "fall_risk_pipeline/requirements.txt" not in dockerfile


def test_api_build_sh_uses_lockfile():
    build_sh = (REPO_ROOT / "api" / "build.sh").read_text(encoding="utf-8")
    assert "requirements-lock.txt" in build_sh
    assert "requirements.txt" not in build_sh


def test_dockerfile_api_production_guards():
    dockerfile = (REPO_ROOT / "Dockerfile.api").read_text(encoding="utf-8")
    assert "requirements-lock.txt" in dockerfile
    assert "CHECKPOINT_HMAC_KEY" in dockerfile
    assert "DEP-004" in dockerfile or "immutable Hub tag" in dockerfile
