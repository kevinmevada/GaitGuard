"""STR-002: Front_end / api/static drift must be committed after sync."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_front_end.py"
FRONT_END = REPO_ROOT / "Front_end"
API_STATIC = REPO_ROOT / "api" / "static"
ASSETS = ("index.html", "main.js", "style.css")


def test_sync_front_end_script_exists():
    assert SYNC_SCRIPT.is_file()


def test_ci_enforces_static_diff_after_sync():
    ci = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "sync_front_end.py" in ci
    assert "git diff --exit-code -- api/static/" in ci


def test_api_static_matches_front_end_after_sync():
    """Run sync, then fail if committed api/static/ still differs (STR-002)."""
    if not (REPO_ROOT / ".git").is_dir():
        pytest.skip("not a git checkout")

    subprocess.run(
        [sys.executable, str(SYNC_SCRIPT)],
        cwd=REPO_ROOT,
        check=True,
    )

    for name in ASSETS:
        assert (FRONT_END / name).read_bytes() == (API_STATIC / name).read_bytes(), name

    proc = subprocess.run(
        ["git", "diff", "--exit-code", "--", "api/static/"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.fail(
            "api/static/ is out of sync with Front_end in git. "
            "Run `python scripts/sync_front_end.py` and commit api/static/.\n"
            f"{proc.stdout}{proc.stderr}"
        )
