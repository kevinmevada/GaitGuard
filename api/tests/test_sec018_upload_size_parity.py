"""SEC-018: ZIP and multi-file upload size caps aligned."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_zip_uncompressed_default_matches_total_upload_cap():
    source = (REPO_ROOT / "api" / "main.py").read_text(encoding="utf-8")
    assert "MAX_UNCOMPRESSED_ZIP_MB = float(" in source
    assert 'os.getenv("MAX_UNCOMPRESSED_ZIP_MB"' in source
    assert "str(int(MAX_TOTAL_SIZE_MB))" in source
