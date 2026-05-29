"""Shared tqdm helpers — one in-place bar when stderr is a TTY."""

from __future__ import annotations

import sys
from typing import Any

from tqdm import tqdm


def stderr_is_tty() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


class _NullProgress:
    """No-op stand-in when progress bars would spam lines (piped / Tee-Object)."""

    def __init__(self, *args: Any, **kwargs: Any):
        self.total = kwargs.get("total")

    def refresh(self) -> None:
        pass

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix(self, **kwargs: Any) -> None:
        pass

    def set_postfix_str(self, s: str) -> None:
        pass

    def close(self) -> None:
        pass

    def __iter__(self):
        return iter([])

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def progress_bar(*args: Any, **kwargs: Any):
    """
    Return a tqdm bar that updates in place on an interactive terminal.

    When stderr is redirected (PowerShell Tee-Object, log capture), returns a
    no-op bar — callers should log milestones via loguru instead.
    """
    if not stderr_is_tty():
        return _NullProgress(*args, **kwargs)

    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.25)
    kwargs.setdefault("smoothing", 0.05)
    return tqdm(*args, **kwargs)
