"""Scheduling intelligence: idle / RAM worker policy with hysteresis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple


@dataclass(frozen=True, slots=True)
class SchedulerPolicyConfig:
    """Tunable policy constants (no magic numbers at call sites).

    Attributes
    ----------
    baseline_workers : int
        Startup / user-active worker count.
    idle_5min_workers, idle_15min_workers : int
        Elevated fan-out after sustained idle.
    ram_high_workers, ram_critical_workers : int
        Throttle targets under memory pressure.
    min_workers, max_workers : int
        Hard clamps applied after every decision.
    idle_5min_s, idle_15min_s : float
        Idle thresholds in seconds.
    ram_high_percent, ram_critical_percent : float
        RAM pressure thresholds (used %).
    eval_interval_s : float
        Minimum seconds between non-emergency evaluations.
    increase_hold_s : float
        Sustained-condition time before allowing an increase.
    increase_cooldown_s : float
        Silence window after an applied increase.
    """

    baseline_workers: int = 14
    idle_5min_workers: int = 16
    idle_15min_workers: int = 18
    ram_high_workers: int = 12
    ram_critical_workers: int = 10
    min_workers: int = 10
    max_workers: int = 18
    idle_5min_s: float = 5.0 * 60.0
    idle_15min_s: float = 15.0 * 60.0
    ram_high_percent: float = 85.0
    ram_critical_percent: float = 92.0
    eval_interval_s: float = 30.0
    increase_hold_s: float = 60.0
    increase_cooldown_s: float = 120.0

    def __post_init__(self) -> None:
        if self.min_workers > self.max_workers:
            raise ValueError("min_workers cannot exceed max_workers")
        if not (self.min_workers <= self.baseline_workers <= self.max_workers):
            raise ValueError("baseline_workers must lie within [min_workers, max_workers]")


class PolicyDecision(NamedTuple):
    """Outcome of a single policy evaluation.

    Attributes
    ----------
    workers : int
        Worker count after clamps.
    reason : str
        Human-readable rationale for logging.
    changed : bool
        True when the returned count differs from ``current_workers``.
    emergency : bool
        True when RAM-critical path short-circuited hysteresis.
    """

    workers: int
    reason: str
    changed: bool
    emergency: bool


class WorkerPolicy:
    """Compute worker targets from idle time and RAM with hysteresis.

    Parameters
    ----------
    config : SchedulerPolicyConfig, optional
        Policy constants.
    """

    def __init__(self, config: SchedulerPolicyConfig | None = None) -> None:
        self.config = config or SchedulerPolicyConfig()
        self._pending_target: int | None = None
        self._pending_since_mono: float | None = None
        self._last_increase_mono: float | None = None

    def reset(self) -> None:
        """Clear hysteresis / cooldown state (tests / reinjection)."""
        self._pending_target = None
        self._pending_since_mono = None
        self._last_increase_mono = None

    def desired_workers(
        self, *, ram_percent: float, idle_seconds: float
    ) -> tuple[int, str]:
        """Map sensors to the unconstrained policy target and reason.

        Priority (highest first): RAM >92%, RAM >85%, idle ≥15m, idle ≥5m,
        else baseline (covers user-active resume → 14).
        """
        cfg = self.config
        if ram_percent > cfg.ram_critical_percent:
            return cfg.ram_critical_workers, (
                f"RAM usage >{cfg.ram_critical_percent:.0f}% "
                f"({ram_percent:.1f}%)"
            )
        if ram_percent > cfg.ram_high_percent:
            return cfg.ram_high_workers, (
                f"RAM usage >{cfg.ram_high_percent:.0f}% ({ram_percent:.1f}%)"
            )
        if idle_seconds >= cfg.idle_15min_s:
            return cfg.idle_15min_workers, (
                f"Laptop idle for 15 minutes ({idle_seconds:.0f} sec)"
            )
        if idle_seconds >= cfg.idle_5min_s:
            return cfg.idle_5min_workers, (
                f"Laptop idle for 5 minutes ({idle_seconds:.0f} sec)"
            )
        return cfg.baseline_workers, (
            f"User activity / baseline (idle {idle_seconds:.0f} sec)"
        )

    def clamp(self, workers: int) -> int:
        """Enforce hard [min_workers, max_workers] bounds."""
        cfg = self.config
        return max(cfg.min_workers, min(int(workers), cfg.max_workers))

    def evaluate(
        self,
        *,
        current_workers: int,
        ram_percent: float,
        idle_seconds: float,
        now_mono: float,
        allow_non_emergency_eval: bool,
    ) -> PolicyDecision:
        """Apply priority rules + hysteresis; return the next worker count.

        Parameters
        ----------
        current_workers : int
            Last committed worker count.
        ram_percent, idle_seconds : float
            Sensor inputs.
        now_mono : float
            Monotonic seconds.
        allow_non_emergency_eval : bool
            False when called inside the 30s evaluation pause; only a
            RAM-critical emergency may still change workers.
        """
        desired, reason = self.desired_workers(
            ram_percent=ram_percent, idle_seconds=idle_seconds
        )
        desired = self.clamp(desired)
        current = self.clamp(current_workers)
        emergency = ram_percent > self.config.ram_critical_percent

        if emergency:
            self._clear_pending()
            changed = desired != current
            if changed and desired > current:
                self._last_increase_mono = now_mono
            return PolicyDecision(desired, reason, changed, True)

        if not allow_non_emergency_eval:
            return PolicyDecision(
                current,
                "Evaluation interval not elapsed (holding current workers)",
                False,
                False,
            )

        if desired < current:
            self._clear_pending()
            return PolicyDecision(desired, reason, True, False)

        if desired == current:
            self._clear_pending()
            return PolicyDecision(current, reason, False, False)

        # Increase path — hysteresis + cooldown.
        if self._pending_target != desired:
            self._pending_target = desired
            self._pending_since_mono = now_mono
            return PolicyDecision(
                current,
                f"Increase to {desired} pending ({self.config.increase_hold_s:.0f}s hold)",
                False,
                False,
            )

        assert self._pending_since_mono is not None
        held = now_mono - self._pending_since_mono
        if held < self.config.increase_hold_s:
            return PolicyDecision(
                current,
                f"Increase to {desired} pending "
                f"({held:.0f}/{self.config.increase_hold_s:.0f}s)",
                False,
                False,
            )

        if self._last_increase_mono is not None:
            since_inc = now_mono - self._last_increase_mono
            if since_inc < self.config.increase_cooldown_s:
                return PolicyDecision(
                    current,
                    f"Increase cooldown "
                    f"({since_inc:.0f}/{self.config.increase_cooldown_s:.0f}s)",
                    False,
                    False,
                )

        self._last_increase_mono = now_mono
        self._clear_pending()
        return PolicyDecision(desired, reason, True, False)

    def _clear_pending(self) -> None:
        self._pending_target = None
        self._pending_since_mono = None
