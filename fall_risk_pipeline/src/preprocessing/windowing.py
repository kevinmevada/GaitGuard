"""
Trial-safe windowing — windows never span trial boundaries.

Policy
------
- 2 s windows @ 100 Hz → 200 samples (configurable via ``deep_learning.sequence_length``).
- 50% overlap within each trial only.
- ``create_windows`` must be called on a single trial's (C, T) array; never on
  concatenated multi-trial buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.deep_models import create_windows


@dataclass(frozen=True)
class WindowSpec:
    window_len: int
    overlap: float
    fs_hz: float

    @property
    def window_s(self) -> float:
        return self.window_len / self.fs_hz

    @property
    def step_samples(self) -> int:
        return max(1, int(self.window_len * (1.0 - self.overlap)))


def parse_window_spec(config: dict[str, Any]) -> WindowSpec:
    dl = config.get("deep_learning") or {}
    ds = config.get("dataset") or {}
    return WindowSpec(
        window_len=int(dl.get("sequence_length", 200)),
        overlap=float(dl.get("overlap", 0.5)),
        fs_hz=float(ds.get("sampling_rate", 100)),
    )


def window_single_trial(
    signal: np.ndarray,
    spec: WindowSpec,
    *,
    return_starts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Slide windows over one trial (C, T); boundaries reset per trial."""
    if signal.ndim != 2:
        raise ValueError(f"Expected (C, T) signal, got shape {signal.shape}")
    return create_windows(
        signal,
        spec.window_len,
        spec.overlap,
        return_starts=return_starts,
    )


def window_many_trials(
    trial_signals: dict[str, np.ndarray],
    spec: WindowSpec,
    *,
    return_starts: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Window each trial independently — no cross-trial windows."""
    windows_by_trial: dict[str, np.ndarray] = {}
    starts_by_trial: dict[str, np.ndarray] | None = {} if return_starts else None

    for tid, sig in trial_signals.items():
        if sig.shape[1] < spec.window_len:
            continue
        if return_starts:
            wins, starts = window_single_trial(sig, spec, return_starts=True)
            windows_by_trial[tid] = wins
            assert starts_by_trial is not None
            starts_by_trial[tid] = starts
        else:
            windows_by_trial[tid] = window_single_trial(sig, spec)

    if not return_starts:
        return windows_by_trial, None
    return windows_by_trial, starts_by_trial


def stack_trial_windows(windows_by_trial: dict[str, np.ndarray]) -> np.ndarray:
    if not windows_by_trial:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(list(windows_by_trial.values()), axis=0).astype(np.float32)
