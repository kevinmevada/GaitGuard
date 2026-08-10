"""Public façade for adaptive Parallel worker selection."""

from __future__ import annotations

import os
import threading
import time
from typing import Protocol

from src.core.resources.idle_detector import IdleDetector, create_idle_detector
from src.core.resources.resource_monitor import ResourceMonitor
from src.core.resources.scheduler_logger import SchedulerLogger
from src.core.resources.worker_policy import SchedulerPolicyConfig, WorkerPolicy


class MonotonicClock(Protocol):
    def monotonic(self) -> float: ...


class SystemMonotonicClock:
    def monotonic(self) -> float:
        return time.monotonic()


class ResourceScheduler:
    """Adaptive worker count for major joblib Parallel stages.

    Parameters
    ----------
    monitor : ResourceMonitor, optional
        Host RAM/CPU sampler.
    idle_detector : IdleDetector, optional
        User-activity idle detector.
    policy : WorkerPolicy, optional
        Decision / hysteresis engine.
    logger : SchedulerLogger, optional
        Decision logger.
    clock : MonotonicClock, optional
        Monotonic clock for intervals.
    logical_cpu_cap : int, optional
        Hard cap from ``os.cpu_count()-1`` (injected for tests).
    """

    def __init__(
        self,
        monitor: ResourceMonitor | None = None,
        idle_detector: IdleDetector | None = None,
        policy: WorkerPolicy | None = None,
        logger: SchedulerLogger | None = None,
        clock: MonotonicClock | None = None,
        *,
        logical_cpu_cap: int | None = None,
    ) -> None:
        self._monitor = monitor or ResourceMonitor()
        self._idle = idle_detector or create_idle_detector()
        self._policy = policy or WorkerPolicy()
        self._logger = logger or SchedulerLogger(console=True)
        self._clock = clock or SystemMonotonicClock()
        n_cpu = os.cpu_count() or 4
        self._logical_cpu_cap = (
            int(logical_cpu_cap)
            if logical_cpu_cap is not None
            else max(1, n_cpu - 1)
        )
        self._lock = threading.RLock()
        self._workers = self._policy.clamp(self._policy.config.baseline_workers)
        self._last_eval_mono: float | None = None

    @property
    def policy(self) -> WorkerPolicy:
        """Expose policy for tests / introspection."""
        return self._policy

    @property
    def config(self) -> SchedulerPolicyConfig:
        """Expose policy config."""
        return self._policy.config

    def current_workers(self, *, force: bool = False) -> int:
        """Return the worker count to use for the next Parallel call.

        Parameters
        ----------
        force : bool
            When True, bypass the 30s evaluation interval (tests).
        """
        with self._lock:
            now = self._clock.monotonic()
            snap = self._monitor.snapshot()
            idle_s = float(self._idle.idle_seconds())

            allow = force
            if not allow:
                if self._last_eval_mono is None:
                    allow = True
                else:
                    allow = (now - self._last_eval_mono) >= self._policy.config.eval_interval_s

            # Always allow emergency path even inside the interval.
            emergency = snap.ram_percent > self._policy.config.ram_critical_percent
            decision = self._policy.evaluate(
                current_workers=self._workers,
                ram_percent=snap.ram_percent,
                idle_seconds=idle_s,
                now_mono=now,
                allow_non_emergency_eval=allow or emergency,
            )

            if allow or decision.emergency:
                self._last_eval_mono = now

            new_workers = self._apply_cpu_cap(decision.workers)
            if new_workers != self._workers and (decision.changed or decision.emergency):
                self._logger.log_decision(
                    when=snap.timestamp,
                    old_workers=self._workers,
                    new_workers=new_workers,
                    reason=decision.reason,
                    ram_percent=snap.ram_percent,
                    idle_seconds=idle_s,
                )
                self._workers = new_workers
            return self._workers

    def _apply_cpu_cap(self, workers: int) -> int:
        capped = min(int(workers), self._logical_cpu_cap)
        return self._policy.clamp(capped)


class SchedulerRegistry:
    """Process-scoped scheduler so hysteresis spans Parallel calls.

    Not import-time global mutable state beyond a single guarded slot;
    tests call :func:`reset_resource_scheduler`.
    """

    _lock = threading.Lock()
    _instance: ResourceScheduler | None = None

    @classmethod
    def get(cls) -> ResourceScheduler:
        with cls._lock:
            if cls._instance is None:
                cls._instance = ResourceScheduler()
            return cls._instance

    @classmethod
    def set(cls, scheduler: ResourceScheduler | None) -> None:
        with cls._lock:
            cls._instance = scheduler

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            if cls._instance is not None:
                logger = getattr(cls._instance, "_logger", None)
                if logger is not None and hasattr(logger, "close"):
                    logger.close()
            cls._instance = None


def get_resource_scheduler() -> ResourceScheduler:
    """Return the process-scoped :class:`ResourceScheduler`."""
    return SchedulerRegistry.get()


def reset_resource_scheduler() -> None:
    """Drop the process-scoped scheduler (unit tests)."""
    SchedulerRegistry.reset()


def parallel_n_jobs(scheduler: ResourceScheduler | None = None) -> int:
    """Worker count for the next ``joblib.Parallel`` invocation."""
    sched = scheduler if scheduler is not None else get_resource_scheduler()
    return int(sched.current_workers())
