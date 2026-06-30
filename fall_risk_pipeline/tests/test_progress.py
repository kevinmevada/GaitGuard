"""Progress bar helpers must still iterate when stderr is not a TTY."""

from __future__ import annotations

import sys
from unittest.mock import patch

from src.utils.progress import progress_bar


def test_null_progress_iterates_when_not_tty():
    items = [10, 20, 30]
    with patch.object(sys.stderr, "isatty", return_value=False):
        assert list(progress_bar(items)) == items


def test_null_progress_manual_update_mode():
    with patch.object(sys.stderr, "isatty", return_value=False):
        bar = progress_bar(total=5)
        bar.update(1)
        bar.close()
        assert list(bar) == []
