"""Shared tqdm helpers — one in-place bar when stderr is a TTY."""

from __future__ import annotations

import contextlib
import os
import sys
import threading
from typing import Any, Iterator

from tqdm import tqdm

# Match legacy pipeline styling (ingest / preprocess / features).
RED_BAR_FORMAT = "\033[31m{l_bar}{bar}{r_bar}\033[0m"


def stderr_is_tty() -> bool:
    return hasattr(sys.stderr, "isatty") and sys.stderr.isatty()


def progress_enabled() -> bool:
    """Whether sub-stage tqdm bars should render (override via env vars)."""
    force = os.environ.get("GAITGUARD_FORCE_PROGRESS", "").lower()
    if force in ("1", "true", "yes"):
        return True
    if force in ("0", "false", "no"):
        return False
    no_progress = os.environ.get("GAITGUARD_NO_PROGRESS", "").lower()
    if no_progress in ("1", "true", "yes"):
        return False
    return stderr_is_tty()


class _NullProgress:
    """No-op stand-in when progress bars would spam lines (piped / Tee-Object)."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._iterable = _coerce_iterable(args[0] if args else None)
        self.total = kwargs.get("total")
        self.n = 0

    def refresh(self) -> None:
        pass

    def update(self, n: int = 1) -> None:
        self.n += n

    def set_postfix(self, **kwargs: Any) -> None:
        pass

    def set_postfix_str(self, s: str) -> None:
        pass

    def close(self) -> None:
        pass

    def __iter__(self):
        if self._iterable is not None:
            return iter(self._iterable)
        return iter([])

    def __enter__(self):
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def _coerce_iterable(candidate: Any) -> Any | None:
    """Return ``candidate`` if it is a tqdm-style iterable (not a scalar total)."""
    if candidate is None:
        return None
    if isinstance(candidate, (str, bytes, int, float)):
        return None
    try:
        iter(candidate)
    except TypeError:
        return None
    return candidate


def progress_bar(*args: Any, **kwargs: Any):
    """
    Return a tqdm bar that updates in place on an interactive terminal.

    When stderr is redirected (PowerShell Tee-Object, log capture), returns a
    no-op bar — callers should log milestones via loguru instead.
    """
    if not progress_enabled():
        return _NullProgress(*args, **kwargs)

    kwargs.setdefault("file", sys.stderr)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 0.25)
    kwargs.setdefault("smoothing", 0.05)
    if not stderr_is_tty():
        # Line-based updates survive Tee-Object / log capture (no in-place \\r).
        kwargs.setdefault("ascii", True)
    return tqdm(*args, **kwargs)


@contextlib.contextmanager
def blocking_progress(
    desc: str,
    *,
    unit: str = "s",
    interval: float = 30.0,
) -> Iterator[Any]:
    """
    Indeterminate bar that ticks every ``interval`` during blocking sklearn fits.

    RFECV and similar estimators expose no per-iteration callbacks; this keeps
    tqdm/log output moving until ``fit()`` returns.
    """
    if not progress_enabled():
        yield _NullProgress()
        return

    bar = progress_bar(total=None, desc=desc, unit=unit)
    stop = threading.Event()

    def _tick() -> None:
        tick = max(int(interval), 1)
        while not stop.wait(interval):
            bar.update(tick)

    ticker = threading.Thread(target=_tick, daemon=True, name=f"progress:{desc}")
    ticker.start()
    try:
        yield bar
    finally:
        stop.set()
        ticker.join(timeout=1.0)
        bar.close()


@contextlib.contextmanager
def stage_spinner(desc: str) -> Iterator[Any]:
    """
    Indeterminate bar for stages with no item loop (fast aggregation, I/O).

    Closes automatically on exit.
    """
    bar = progress_bar(total=None, desc=desc, unit="it")
    try:
        yield bar
    finally:
        bar.close()
