"""SEC-007: upload filenames must match REQUIRED_FILES exactly (no substring guessing)."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from main import (  # noqa: E402
    ALLOWED_UPLOAD_NAMES,
    METADATA_FILE,
    REQUIRED_FILES,
    parse_uploaded_files,
    require_allowed_upload_name,
    upload_basename,
)


def _csv_upload(name: str, content: str = "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n0,1,2,3,4,5,6\n") -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(content.encode("utf-8")))


def _meta_upload(content: str = '{"participant_id":"p1","trial_id":"t1"}') -> UploadFile:
    return UploadFile(filename=METADATA_FILE, file=io.BytesIO(content.encode("utf-8")))


def _required_uploads() -> list[UploadFile]:
    return [
        *[_csv_upload(name) for name in REQUIRED_FILES],
        _meta_upload(),
    ]


def test_upload_basename_strips_path_components():
    assert upload_basename("nested/head_raw.csv") == "head_raw.csv"


def test_require_allowed_upload_name_rejects_substring_poison():
    with pytest.raises(HTTPException) as exc:
        require_allowed_upload_name("evil_head_raw.csv")
    assert exc.value.status_code == 400
    assert "Unrecognized upload file" in exc.value.detail


def test_require_allowed_upload_name_accepts_exact_required_files():
    for name in ALLOWED_UPLOAD_NAMES:
        assert require_allowed_upload_name(name) == name


def test_parse_uploaded_files_rejects_extra_csv():
    files = _required_uploads() + [_csv_upload("evil_head_raw.csv")]
    with pytest.raises(HTTPException) as exc:
        parse_uploaded_files(files)
    assert exc.value.status_code == 400
    assert "evil_head_raw.csv" in exc.value.detail


def test_parse_uploaded_files_accepts_exact_required_set():
    sensor_frames, metadata = parse_uploaded_files(_required_uploads())
    assert set(sensor_frames.keys()) == set(REQUIRED_FILES.values())
    assert metadata["participant_id"] == "p1"


def test_zip_rejects_unrecognized_basename(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name in REQUIRED_FILES:
            zf.writestr(name, "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n0,1,2,3,4,5,6\n")
        zf.writestr(METADATA_FILE, '{"participant_id":"p1","trial_id":"t1"}')
        zf.writestr("evil_head_raw.csv", "time,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n0,9,9,9,9,9,9\n")

    upload = UploadFile(filename="trial.zip", file=io.BytesIO(buf.getvalue()))
    with pytest.raises(HTTPException) as exc:
        parse_uploaded_files([upload])
    assert exc.value.status_code == 400
    assert "evil_head_raw.csv" in exc.value.detail
