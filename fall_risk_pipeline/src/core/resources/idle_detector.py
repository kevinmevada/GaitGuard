"""User-input idle detection with platform adapters."""

from __future__ import annotations

import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Protocol


class MonotonicClock(Protocol):
    """Injectable monotonic clock (seconds)."""

    def monotonic(self) -> float:
        """Return monotonic seconds."""


class SystemMonotonicClock:
    """Wall monotonic clock via ``time.monotonic``."""

    def monotonic(self) -> float:
        return time.monotonic()


class IdleDetector(ABC):
    """Abstract idle detector.

    Implementations must avoid aggressive polling; caching is required.
    """

    @abstractmethod
    def idle_seconds(self) -> float:
        """Return seconds since the last keyboard/mouse input."""


class WindowsIdleDetector(IdleDetector):
    """Windows idle time via ``GetLastInputInfo`` (usered).

    Parameters
    ----------
    cache_ttl_s : float
        Minimum seconds between Win32 queries (default 5.0).
    clock : MonotonicClock, optional
        Injectable clock for tests.
    """

    def __init__(
        self,
        cache_ttl_s: float = 5.0,
        clock: MonotonicClock | None = None,
    ) -> None:
        if cache_ttl_s <= 0:
            raise ValueError("cache_ttl_s must be positive")
        self._cache_ttl_s = float(cache_ttl_s)
        self._clock = clock or SystemMonotonicClock()
        self._lock = threading.Lock()
        self._cached_idle_s = 0.0
        self._cached_at: float | None = None

    def idle_seconds(self) -> float:
        now = self._clock.monotonic()
        with self._lock:
            if (
                self._cached_at is not None
                and (now - self._cached_at) < self._cache_ttl_s
            ):
                return self._cached_idle_s
            idle = self._query_idle_seconds()
            self._cached_idle_s = idle
            self._cached_at = now
            return idle

    def _query_idle_seconds(self) -> float:
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            return 0.0
        tick = ctypes.windll.kernel32.GetTickCount()
        # DWORD wrap-around safe difference.
        idle_ms = (tick - info.dwTime) & 0xFFFFFFFF
        return float(idle_ms) / 1000.0


class NullIdleDetector(IdleDetector):
    """Fallback detector: reports always-active (0s idle).

    Used on non-Windows platforms until native adapters exist, and in tests.
    """

    def __init__(self, idle_seconds_value: float = 0.0) -> None:
        self._idle_seconds_value = float(idle_seconds_value)

    def idle_seconds(self) -> float:
        return self._idle_seconds_value

    def set_idle_seconds(self, value: float) -> None:
        """Test helper to mutate reported idle time."""
        self._idle_seconds_value = float(value)


def create_idle_detector(
    *,
    platform: str | None = None,
    cache_ttl_s: float = 5.0,
    clock: MonotonicClock | None = None,
) -> IdleDetector:
    """Construct a platform-appropriate idle detector.

    Parameters
    ----------
    platform : str, optional
        Override ``sys.platform`` (``"win32"``, ``"linux"``, …).
    cache_ttl_s : float
        Cache TTL for Windows queries.
    clock : MonotonicClock, optional
        Injectable monotonic clock.
    """
    plat = platform if platform is not None else sys.platform
    if plat.startswith("win"):
        return WindowsIdleDetector(cache_ttl_s=cache_ttl_s, clock=clock)
    # Future: Linux (X11/Wayland) and macOS Quartz idle adapters.
    return NullIdleDetector(idle_seconds_value=0.0)
