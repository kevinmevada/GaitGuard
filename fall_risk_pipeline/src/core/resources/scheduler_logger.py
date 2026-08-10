"""Decision logging for the resource scheduler."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import IO, Protocol, TextIO


class SupportsWrite(Protocol):
    def write(self, s: str) -> int: ...

    def flush(self) -> None: ...


class SchedulerLogger:
    """Format and emit worker-transition records.

    Parameters
    ----------
    console : bool
        Write to ``console_stream`` (default stderr).
    log_path : path-like, optional
        Optional append-only log file.
    console_stream : text stream, optional
        Console destination (injected for tests).
    """

    def __init__(
        self,
        *,
        console: bool = True,
        log_path: str | Path | None = None,
        console_stream: TextIO[str] | SupportsWrite | None = None,
    ) -> None:
        self._console = bool(console)
        self._console_stream: SupportsWrite = console_stream or sys.stderr
        self._log_path = Path(log_path) if log_path is not None else None
        self._file_handle: IO[str] | None = None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = self._log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        """Close any open file handle."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def log_decision(
        self,
        *,
        when: datetime,
        old_workers: int,
        new_workers: int,
        reason: str,
        ram_percent: float,
        idle_seconds: float,
    ) -> str:
        """Emit a formatted decision (always when workers change).

        Returns
        -------
        str
            The formatted record (useful for tests).
        """
        message = self.format_decision(
            when=when,
            old_workers=old_workers,
            new_workers=new_workers,
            reason=reason,
            ram_percent=ram_percent,
            idle_seconds=idle_seconds,
        )
        self._emit(message)
        return message

    def log_hold(
        self,
        *,
        when: datetime,
        workers: int,
        reason: str,
        ram_percent: float,
        idle_seconds: float,
    ) -> str:
        """Optional verbose hold/noop record (file only if console quiet)."""
        stamp = when.strftime("%H:%M:%S")
        message = (
            f"[{stamp}] Workers {workers} (hold)\n"
            f"Reason:\n{reason}\n\n"
            f"RAM:\n{ram_percent:.0f}%\n\n"
            f"Idle:\n{idle_seconds:.0f} sec\n"
        )
        # Holds stay out of console noise; file log keeps the audit trail.
        if self._file_handle is not None:
            self._file_handle.write(message + "\n")
            self._file_handle.flush()
        return message

    @staticmethod
    def format_decision(
        *,
        when: datetime,
        old_workers: int,
        new_workers: int,
        reason: str,
        ram_percent: float,
        idle_seconds: float,
    ) -> str:
        """Return the canonical multi-line decision string."""
        stamp = when.strftime("%H:%M:%S")
        return (
            f"[{stamp}]\n"
            f"Workers {old_workers} -> {new_workers}\n\n"
            f"Reason:\n{reason}\n\n"
            f"RAM:\n{ram_percent:.0f}%\n\n"
            f"Idle:\n{idle_seconds:.0f} sec\n"
        )

    def _emit(self, message: str) -> None:
        if self._console:
            self._console_stream.write(message + "\n")
            self._console_stream.flush()
        if self._file_handle is not None:
            self._file_handle.write(message + "\n")
            self._file_handle.flush()
