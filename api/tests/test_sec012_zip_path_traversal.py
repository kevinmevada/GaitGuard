"""SEC-012: ZIP uploads must reject Zip Slip path traversal entries."""

from __future__ import annotations

import ast
import io
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

API_ROOT = Path(__file__).resolve().parents[1]
MAIN_PATH = API_ROOT / "main.py"
sys.path.insert(0, str(API_ROOT))

from main import (  # noqa: E402
    METADATA_FILE,
    REQUIRED_FILES,
    _is_safe_zip_member,
    parse_uploaded_files,
)


def _csv_line() -> str:
    return "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n0,1,2,3,4,5,6\n"


def _valid_zip_bytes(extra_entries: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in REQUIRED_FILES:
            zf.writestr(name, _csv_line())
        zf.writestr(METADATA_FILE, '{"participant_id":"p1","trial_id":"t1"}')
        for name, content in (extra_entries or {}).items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_is_safe_zip_member_rejects_traversal():
    assert not _is_safe_zip_member("../../head_raw.csv")
    assert not _is_safe_zip_member("foo/../../../head_raw.csv")
    assert not _is_safe_zip_member("/head_raw.csv")
    assert not _is_safe_zip_member("../metadata.json")


def test_is_safe_zip_member_accepts_nested_allowed_paths():
    assert _is_safe_zip_member("trial/head_raw.csv")
    assert _is_safe_zip_member("head_raw.csv")


def test_parse_uploaded_files_rejects_zip_slip_entry():
    upload = UploadFile(
        filename="trial.zip",
        file=io.BytesIO(_valid_zip_bytes({"../../head_raw.csv": _csv_line()})),
    )
    with pytest.raises(HTTPException) as exc:
        parse_uploaded_files([upload])
    assert exc.value.status_code == 400
    assert "path traversal" in exc.value.detail.lower()


def test_parse_uploaded_files_accepts_nested_zip_layout():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in REQUIRED_FILES:
            zf.writestr(f"trial/{name}", _csv_line())
        zf.writestr(f"trial/{METADATA_FILE}", '{"participant_id":"p1","trial_id":"t1"}')
    upload = UploadFile(filename="trial.zip", file=io.BytesIO(buf.getvalue()))
    sensor_frames, metadata = parse_uploaded_files([upload])
    assert set(sensor_frames.keys()) == set(REQUIRED_FILES.values())
    assert metadata["participant_id"] == "p1"


def test_zip_loop_does_not_use_dead_name_parts_check():
    source = MAIN_PATH.read_text(encoding="utf-8")
    assert "_is_safe_zip_member" in source
    assert "Path(entry).name != Path(entry).parts[-1]" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "parse_uploaded_files":
            fn_source = ast.get_source_segment(source, node) or ""
            assert "_is_safe_zip_member" in fn_source
            return
    raise AssertionError("parse_uploaded_files not found")
