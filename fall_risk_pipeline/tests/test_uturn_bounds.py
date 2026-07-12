"""U-turn boundary parsing for turning features."""

from __future__ import annotations

import pytest

from src.features.feature_extractor import _parse_uturn_bounds


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (None, 100),
        (50, None),
        (float("nan"), 100),
        (50, float("nan")),
        ("bad", 100),
        (100, 50),
        (100, 100),
    ],
)
def test_parse_uturn_bounds_rejects_invalid(start, end):
    assert _parse_uturn_bounds(start, end) is None


def test_parse_uturn_bounds_accepts_valid_integers():
    assert _parse_uturn_bounds(100, 200) == (100, 200)
    assert _parse_uturn_bounds(100.0, 200.0) == (100, 200)
