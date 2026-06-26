"""COD-002: prod requirements exclude unused and dev-only packages."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAIN = REPO_ROOT / "fall_risk_pipeline" / "requirements.txt"
DEV = REPO_ROOT / "fall_risk_pipeline" / "requirements-dev.txt"
LOCK = REPO_ROOT / "fall_risk_pipeline" / "requirements-lock.txt"
CI = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_prod_requirements_exclude_unused_and_test_deps():
    main = MAIN.read_text(encoding="utf-8").lower()
    for pkg in ("torchvision", "pytest", "pytest-cov", "plotly", "tabulate", "jinja2", "optuna-dashboard"):
        assert pkg not in main


def test_dev_requirements_carry_pytest_only():
    dev = DEV.read_text(encoding="utf-8").lower()
    assert "pytest>=" in dev
    assert "-r requirements.txt" in dev
    for pkg in ("plotly", "tabulate", "jinja2", "optuna-dashboard"):
        assert pkg not in dev


def test_lockfile_excludes_torchvision_and_pytest():
    lock = LOCK.read_text(encoding="utf-8").lower()
    assert "torchvision==" not in lock
    assert "pytest==" not in lock


def test_ci_installs_pinned_lockfiles():
    ci = CI.read_text(encoding="utf-8")
    assert "requirements-lock.txt" in ci
    assert "requirements-dev-lock.txt" in ci
    assert "api/requirements-lock.txt" not in ci
    assert "sync_front_end" not in ci
    assert "requirements-dev.txt" not in ci
    assert "download.pytorch.org/whl/cpu" in ci
