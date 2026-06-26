"""Wall-clock timing helpers for compute overhead reporting."""

from __future__ import annotations

import platform
import time
from collections.abc import Callable
from typing import Any, TypeVar

import torch

T = TypeVar("T")


def device_label() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return "cpu"


def host_info() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "device": device_label(),
    }


def median_seconds(samples: list[float]) -> float:
    if not samples:
        return float("nan")
    return float(sorted(samples)[len(samples) // 2])


def time_callable(
    fn: Callable[[], T],
    *,
    n_warmup: int = 1,
    n_repeat: int = 5,
) -> tuple[T, float, list[float]]:
    """
    Run *fn* with warmup; return (last_result, median_elapsed_s, all_elapsed_s).
    """
    result: T | None = None
    for _ in range(max(0, n_warmup)):
        result = fn()
    timings: list[float] = []
    for _ in range(max(1, n_repeat)):
        t0 = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - t0)
    return result, median_seconds(timings), timings


def time_training(fn: Callable[[], T]) -> tuple[T, float]:
    """Single-shot training wall time (seconds)."""
    t0 = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - t0
