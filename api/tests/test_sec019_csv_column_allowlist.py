"""SEC-019 / LOW-05: CSV column names validated with allowlist (not HTML denylist)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_ROOT))

from main import validate_csv_content  # noqa: E402


def test_validate_csv_content_accepts_imu_headers():
    df = pd.DataFrame([[0, 1, 2, 3]], columns=["time", "acc_x", "Acc Y", "gyr_z"])
    validate_csv_content(df, "head")


@pytest.mark.parametrize(
    "bad_col",
    [
        "<script>",
        "javascript:alert(1)",
        "<svg onload=alert(1)",
        "acc_x;drop table",
        "acc\x00_x",
    ],
)
def test_validate_csv_content_rejects_non_allowlisted_headers(bad_col: str):
    df = pd.DataFrame(columns=[bad_col, "acc_x"])
    with pytest.raises(HTTPException) as exc:
        validate_csv_content(df, "lower_back")
    assert exc.value.status_code == 400
