"""Progress bar helpers must still iterate when stderr is not a TTY."""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

from src.utils.progress import blocking_progress, progress_bar, stage_spinner

_NO_FORCE = patch.dict(os.environ, {"GAITGUARD_FORCE_PROGRESS": "0"}, clear=False)


def test_null_progress_iterates_when_not_tty():
    items = [10, 20, 30]
    with _NO_FORCE, patch.object(sys.stderr, "isatty", return_value=False):
        assert list(progress_bar(items)) == items


def test_null_progress_manual_update_mode():
    with _NO_FORCE, patch.object(sys.stderr, "isatty", return_value=False):
        bar = progress_bar(total=5)
        bar.update(1)
        bar.close()
        assert list(bar) == []
        assert bar.n == 1


def test_stage_spinner_closes_when_not_tty():
    with _NO_FORCE, patch.object(sys.stderr, "isatty", return_value=False):
        with stage_spinner("fast_stage") as bar:
            bar.update(0)


def test_blocking_progress_yields_when_not_tty():
    with _NO_FORCE, patch.object(sys.stderr, "isatty", return_value=False):
        with blocking_progress("rfecv", interval=0.01) as bar:
            bar.update(1)