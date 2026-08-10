"""Cross-platform resource sampling via psutil."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

import psutil


class Clock(Protocol):
    """Injectable clock for deterministic tests."""

    def now(self) -> datetime:
        """Return timezone-aware UTC timestamp."""


class SystemClock:
    """Wall-clock using timezone-aware UTC."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """Point-in-time host resource sample.

    Attributes
    ----------
    ram_percent : float
        Used physical memory, 0–100.
    ram_available_gb : float
        Available physical memory in gibibytes.
    cpu_percent : float
        Non-blocking CPU utilization sample, 0–100.
    timestamp : datetime
        Sample time (UTC).
    """

    ram_percent: float
    ram_available_gb: float
    cpu_percent: float
    timestamp: datetime


class ResourceMonitor:
    """Collect host RAM / CPU statistics without platform-specific APIs.

    Parameters
    ----------
    clock : Clock, optional
        Timestamp provider (defaults to :class:`SystemClock`).
    cpu_sample_interval : float, optional
        Passed to ``psutil.cpu_percent``. Use ``0.0`` for non-blocking
        samples so the monitor never sleeps on the caller's thread.
    """

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        cpu_sample_interval: float = 0.0,
    ) -> None:
        self._clock = clock or SystemClock()
        self._cpu_sample_interval = float(cpu_sample_interval)
        # Prime psutil's CPU counter so the first non-blocking read is defined.
        psutil.cpu_percent(interval=None)

    def ram_percent(self) -> float:
        """Return used physical RAM as a percentage."""
        return float(psutil.virtual_memory().percent)

    def ram_available_gb(self) -> float:
        """Return available physical RAM in GiB."""
        return float(psutil.virtual_memory().available) / (1024.0**3)

    def cpu_percent(self) -> float:
        """Return a non-blocking CPU utilization percentage."""
        return float(psutil.cpu_percent(interval=self._cpu_sample_interval))

    def timestamp(self) -> datetime:
        """Return the current monitor timestamp."""
        return self._clock.now()

    def snapshot(self) -> ResourceSnapshot:
        """Return a coherent RAM + CPU sample."""
        vm = psutil.virtual_memory()
        return ResourceSnapshot(
            ram_percent=float(vm.percent),
            ram_available_gb=float(vm.available) / (1024.0**3),
            cpu_percent=float(psutil.cpu_percent(interval=self._cpu_sample_interval)),
            timestamp=self._clock.now(),
        )
