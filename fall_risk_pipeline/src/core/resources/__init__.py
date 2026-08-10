"""Resource-aware process scheduling for joblib Parallel stages.

Public API
----------
ResourceScheduler
    Primary façade: ``current_workers()``.
get_resource_scheduler
    Process-scoped registry so hysteresis persists across Parallel calls.
parallel_n_jobs
    Convenience wrapper used at call sites.
"""

from __future__ import annotations

from src.core.resources.idle_detector import IdleDetector, create_idle_detector
from src.core.resources.resource_monitor import ResourceMonitor
from src.core.resources.resource_scheduler import (
    ResourceScheduler,
    SchedulerRegistry,
    get_resource_scheduler,
    parallel_n_jobs,
    reset_resource_scheduler,
)
from src.core.resources.scheduler_logger import SchedulerLogger
from src.core.resources.worker_policy import SchedulerPolicyConfig, WorkerPolicy

__all__ = [
    "IdleDetector",
    "ResourceMonitor",
    "ResourceScheduler",
    "SchedulerLogger",
    "SchedulerPolicyConfig",
    "SchedulerRegistry",
    "WorkerPolicy",
    "create_idle_detector",
    "get_resource_scheduler",
    "parallel_n_jobs",
    "reset_resource_scheduler",
]
