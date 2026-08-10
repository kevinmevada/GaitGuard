"""Unit tests for the adaptive ResourceScheduler subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pytest

from src.core.resources import (
    ResourceMonitor,
    ResourceScheduler,
    SchedulerLogger,
    SchedulerPolicyConfig,
    WorkerPolicy,
    create_idle_detector,
    get_resource_scheduler,
    parallel_n_jobs,
    reset_resource_scheduler,
)
from src.core.resources.idle_detector import NullIdleDetector, WindowsIdleDetector
from src.core.resources.resource_monitor import ResourceSnapshot
from src.core.resources.worker_policy import PolicyDecision


@dataclass
class FakeClock:
    t: float = 0.0

    def monotonic(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += float(seconds)


@dataclass
class FakeWallClock:
    when: datetime

    def now(self) -> datetime:
        return self.when


class FakeMonitor:
    def __init__(self, ram_percent: float = 50.0, cpu_percent: float = 10.0) -> None:
        self.ram_percent_value = float(ram_percent)
        self.cpu_percent_value = float(cpu_percent)
        self.ram_available_gb_value = 16.0
        self._wall = FakeWallClock(datetime(2026, 7, 14, 12, 5, 12, tzinfo=timezone.utc))

    def ram_percent(self) -> float:
        return self.ram_percent_value

    def ram_available_gb(self) -> float:
        return self.ram_available_gb_value

    def cpu_percent(self) -> float:
        return self.cpu_percent_value

    def timestamp(self) -> datetime:
        return self._wall.now()

    def snapshot(self) -> ResourceSnapshot:
        return ResourceSnapshot(
            ram_percent=self.ram_percent_value,
            ram_available_gb=self.ram_available_gb_value,
            cpu_percent=self.cpu_percent_value,
            timestamp=self._wall.now(),
        )


@pytest.fixture(autouse=True)
def _reset_registry() -> None:
    reset_resource_scheduler()
    yield
    reset_resource_scheduler()


def _scheduler(
    *,
    ram: float = 50.0,
    idle: float = 0.0,
    clock: FakeClock | None = None,
    logical_cpu_cap: int = 32,
) -> tuple[ResourceScheduler, FakeMonitor, NullIdleDetector, FakeClock, StringIO]:
    clock = clock or FakeClock()
    monitor = FakeMonitor(ram_percent=ram)
    idle_det = NullIdleDetector(idle_seconds_value=idle)
    buf = StringIO()
    logger = SchedulerLogger(console=True, console_stream=buf)
    policy = WorkerPolicy(SchedulerPolicyConfig())
    sched = ResourceScheduler(
        monitor=monitor,  # type: ignore[arg-type]
        idle_detector=idle_det,
        policy=policy,
        logger=logger,
        clock=clock,
        logical_cpu_cap=logical_cpu_cap,
    )
    return sched, monitor, idle_det, clock, buf


class TestStartupState:
    def test_baseline_workers_are_14(self) -> None:
        sched, *_ = _scheduler(ram=50.0, idle=0.0)
        assert sched.current_workers(force=True) == 14

    def test_parallel_n_jobs_matches_scheduler(self) -> None:
        sched, *_ = _scheduler()
        assert parallel_n_jobs(sched) == 14


class TestIdleTransitions:
    def test_idle_5_minutes_requires_hold_then_increases_to_16(self) -> None:
        sched, _m, idle, clock, buf = _scheduler(idle=300.0)
        assert sched.current_workers(force=True) == 14  # pending
        clock.advance(60.0)
        assert sched.current_workers(force=True) == 16
        assert "14 -> 16" in buf.getvalue()
        assert "5 minutes" in buf.getvalue()

    def test_idle_15_minutes_reaches_18_with_hold_and_cooldown(self) -> None:
        sched, _m, idle, clock, _buf = _scheduler(idle=900.0)
        assert sched.current_workers(force=True) == 14
        clock.advance(60.0)
        assert sched.current_workers(force=True) == 18  # jump desired is 18
        # After increase, cooldown blocks further increases (already at max).
        idle.set_idle_seconds(900.0)
        clock.advance(30.0)
        assert sched.current_workers(force=True) == 18

    def test_stepwise_5_then_15(self) -> None:
        sched, _m, idle, clock, _buf = _scheduler(idle=300.0)
        assert sched.current_workers(force=True) == 14
        clock.advance(60.0)
        assert sched.current_workers(force=True) == 16
        idle.set_idle_seconds(900.0)
        # Cooldown 120s after last increase; hold for 18 accumulates in parallel.
        clock.advance(30.0)
        assert sched.current_workers(force=True) == 16  # cooldown
        clock.advance(90.0)  # cooldown cleared and 60s hold already satisfied
        assert sched.current_workers(force=True) == 18


class TestActiveTransition:
    def test_activity_returns_immediately_to_14(self) -> None:
        sched, _m, idle, clock, buf = _scheduler(idle=300.0)
        sched.current_workers(force=True)
        clock.advance(60.0)
        assert sched.current_workers(force=True) == 16
        idle.set_idle_seconds(10.0)
        assert sched.current_workers(force=True) == 14
        assert "16 -> 14" in buf.getvalue()


class TestRamPressure:
    def test_ram_over_85_drops_to_12_immediately(self) -> None:
        sched, mon, _i, _c, buf = _scheduler(ram=86.0, idle=0.0)
        assert sched.current_workers(force=True) == 12
        assert "14 -> 12" in buf.getvalue()

    def test_ram_over_92_emergency_to_10(self) -> None:
        sched, mon, _i, clock, buf = _scheduler(ram=50.0, idle=0.0)
        assert sched.current_workers(force=True) == 14
        mon.ram_percent_value = 93.0
        # Even without force / inside interval — emergency path.
        assert sched.current_workers(force=False) == 10
        assert "14 -> 10" in buf.getvalue()
        assert "92%" in buf.getvalue()

    def test_ram_overrides_idle_boost(self) -> None:
        sched, mon, idle, clock, _buf = _scheduler(ram=86.0, idle=900.0)
        assert sched.current_workers(force=True) == 12


class TestHysteresisAndCooldown:
    def test_eval_interval_holds_non_emergency(self) -> None:
        sched, mon, idle, clock, _buf = _scheduler(idle=0.0)
        assert sched.current_workers(force=False) == 14
        idle.set_idle_seconds(300.0)
        # Immediate second call within 30s should not even start pending...
        # evaluate with allow=False returns current.
        assert sched.current_workers(force=False) == 14
        clock.advance(30.0)
        assert sched.current_workers(force=False) == 14  # pending
        clock.advance(60.0)
        assert sched.current_workers(force=False) == 16

    def test_increase_requires_60s_hold(self) -> None:
        policy = WorkerPolicy()
        d1 = policy.evaluate(
            current_workers=14,
            ram_percent=50.0,
            idle_seconds=300.0,
            now_mono=0.0,
            allow_non_emergency_eval=True,
        )
        assert d1 == PolicyDecision(14, d1.reason, False, False)
        d2 = policy.evaluate(
            current_workers=14,
            ram_percent=50.0,
            idle_seconds=300.0,
            now_mono=59.0,
            allow_non_emergency_eval=True,
        )
        assert d2.workers == 14
        assert d2.changed is False
        d3 = policy.evaluate(
            current_workers=14,
            ram_percent=50.0,
            idle_seconds=300.0,
            now_mono=60.0,
            allow_non_emergency_eval=True,
        )
        assert d3.workers == 16
        assert d3.changed is True

    def test_increase_cooldown_two_minutes(self) -> None:
        sched, _m, idle, clock, _buf = _scheduler(idle=300.0)
        sched.current_workers(force=True)
        clock.advance(60.0)
        assert sched.current_workers(force=True) == 16
        idle.set_idle_seconds(900.0)
        clock.advance(60.0)
        assert sched.current_workers(force=True) == 16  # cooldown; hold ticking
        clock.advance(60.0)  # 120s cooldown done + hold already >= 60s
        assert sched.current_workers(force=True) == 18


class TestClamps:
    def test_never_below_10_or_above_18(self) -> None:
        policy = WorkerPolicy()
        assert policy.clamp(9) == 10
        assert policy.clamp(19) == 18
        assert policy.clamp(14) == 14

    def test_logical_cpu_cap_respected(self) -> None:
        sched, _m, idle, clock, _buf = _scheduler(idle=900.0, logical_cpu_cap=11)
        sched.current_workers(force=True)
        clock.advance(60.0)
        # desired 18 but capped to 11, then clamp min 10 → 11
        assert sched.current_workers(force=True) == 11


class TestLogger:
    def test_format_decision_shape(self) -> None:
        text = SchedulerLogger.format_decision(
            when=datetime(2026, 7, 14, 12, 5, 12, tzinfo=timezone.utc),
            old_workers=14,
            new_workers=16,
            reason="Laptop idle for 5 minutes",
            ram_percent=61.0,
            idle_seconds=302.0,
        )
        assert text.startswith("[12:05:12]")
        assert "Workers 14 -> 16" in text
        assert "RAM:\n61%" in text
        assert "Idle:\n302 sec" in text

    def test_file_logging(self, tmp_path: Path) -> None:
        path = tmp_path / "sched.log"
        logger = SchedulerLogger(console=False, log_path=path)
        logger.log_decision(
            when=datetime(2026, 7, 14, 12, 5, 12, tzinfo=timezone.utc),
            old_workers=14,
            new_workers=12,
            reason="RAM usage >85%",
            ram_percent=86.0,
            idle_seconds=0.0,
        )
        logger.close()
        body = path.read_text(encoding="utf-8")
        assert "14 -> 12" in body


class TestSchedulerApiAndRegistry:
    def test_get_resource_scheduler_singleton(self) -> None:
        a = get_resource_scheduler()
        b = get_resource_scheduler()
        assert a is b

    def test_thread_safe_reads(self) -> None:
        import threading

        sched, *_ = _scheduler()
        results: list[int] = []

        def worker() -> None:
            results.append(sched.current_workers(force=True))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r == 14 for r in results)


class TestIdleDetectorFactory:
    def test_windows_factory(self) -> None:
        det = create_idle_detector(platform="win32")
        assert isinstance(det, WindowsIdleDetector)

    def test_non_windows_null(self) -> None:
        det = create_idle_detector(platform="linux")
        assert isinstance(det, NullIdleDetector)
        assert det.idle_seconds() == 0.0

    def test_windows_cache_ttl_validation(self) -> None:
        with pytest.raises(ValueError):
            WindowsIdleDetector(cache_ttl_s=0.0)


class TestResourceMonitorSmoke:
    def test_monitor_methods_run(self) -> None:
        mon = ResourceMonitor()
        assert 0.0 <= mon.ram_percent() <= 100.0
        assert mon.ram_available_gb() >= 0.0
        assert mon.cpu_percent() >= 0.0
        assert mon.timestamp().tzinfo is not None
        snap = mon.snapshot()
        assert snap.ram_percent == pytest.approx(mon.ram_percent(), abs=5.0)


class TestPolicyConfig:
    def test_invalid_bounds(self) -> None:
        with pytest.raises(ValueError):
            SchedulerPolicyConfig(min_workers=18, max_workers=10)

    def test_invalid_baseline(self) -> None:
        with pytest.raises(ValueError):
            SchedulerPolicyConfig(baseline_workers=9, min_workers=10, max_workers=18)

    def test_policy_reset_clears_pending(self) -> None:
        policy = WorkerPolicy()
        policy.evaluate(
            current_workers=14,
            ram_percent=50.0,
            idle_seconds=300.0,
            now_mono=0.0,
            allow_non_emergency_eval=True,
        )
        policy.reset()
        d = policy.evaluate(
            current_workers=14,
            ram_percent=50.0,
            idle_seconds=300.0,
            now_mono=60.0,
            allow_non_emergency_eval=True,
        )
        # After reset, 60s is not enough without a fresh pending start.
        assert d.workers == 14
        assert d.changed is False


class TestLoggerHoldAndRegistrySet:
    def test_log_hold_writes_file_only(self, tmp_path: Path) -> None:
        path = tmp_path / "hold.log"
        buf = StringIO()
        logger = SchedulerLogger(console=True, log_path=path, console_stream=buf)
        logger.log_hold(
            when=datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc),
            workers=14,
            reason="pending",
            ram_percent=50.0,
            idle_seconds=10.0,
        )
        logger.close()
        assert buf.getvalue() == ""
        assert "hold" in path.read_text(encoding="utf-8")

    def test_registry_set_override(self) -> None:
        from src.core.resources import SchedulerRegistry

        sched, *_ = _scheduler()
        SchedulerRegistry.set(sched)
        assert get_resource_scheduler() is sched


class TestWindowsIdleDetectorCached:
    def test_cache_avoids_repeat_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        clock = FakeClock()
        det = WindowsIdleDetector(cache_ttl_s=5.0, clock=clock)
        calls = {"n": 0}

        def fake_query(self: WindowsIdleDetector) -> float:
            calls["n"] += 1
            return 123.0

        monkeypatch.setattr(WindowsIdleDetector, "_query_idle_seconds", fake_query)
        assert det.idle_seconds() == 123.0
        clock.advance(1.0)
        assert det.idle_seconds() == 123.0
        assert calls["n"] == 1
        clock.advance(5.0)
        assert det.idle_seconds() == 123.0
        assert calls["n"] == 2

    def test_query_idle_seconds_win32(self) -> None:
        # Live Win32 path on the development host.
        det = WindowsIdleDetector(cache_ttl_s=0.01)
        assert det.idle_seconds() >= 0.0


class TestSchedulerProperties:
    def test_policy_and_config_properties(self) -> None:
        sched, *_ = _scheduler()
        assert sched.config.baseline_workers == 14
        assert sched.policy is not None
