"""
Delay-embedding parameter estimation for nonlinear time-series features.

- tau (lag): first local minimum of average mutual information (AMI)
  Fraser & Swinney, Phys. Rev. A 33, 1134 (1986).
- m (embedding dimension): false nearest neighbors (FNN) criterion
  Kennel et al., Phys. Rev. A 45, 3403 (1992).
"""

from __future__ import annotations

import numpy as np


def delay_embedding(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Build m-dimensional delay vectors [x_i, x_{i+tau}, ..., x_{i+(m-1)tau}]."""
    n = len(x) - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    return np.column_stack([x[i : i + n] for i in range(0, m * tau, tau)])


def average_mutual_information(x: np.ndarray, lag: int, n_bins: int | None = None) -> float:
    """Histogram-based AMI between x(t) and x(t+lag)."""
    if lag < 1 or len(x) <= lag + 20:
        return float("nan")
    a = np.asarray(x[:-lag], dtype=float)
    b = np.asarray(x[lag:], dtype=float)
    n_bins = n_bins or max(16, int(np.sqrt(len(a))))
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return float("nan")
    hist2, _, _ = np.histogram2d(a, b, bins=n_bins, range=[[lo, hi], [lo, hi]])
    pxy = hist2 / hist2.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    denom = px[:, None] * py[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((pxy > 0) & (denom > 0), pxy / denom, 1.0)
        terms = np.where(pxy > 0, pxy * np.log(ratio), 0.0)
    return float(np.sum(terms))


def estimate_tau_ami(x: np.ndarray, max_lag: int = 50) -> int:
    """First local minimum of AMI vs lag; global minimum if none found."""
    n = len(x)
    max_lag = min(max_lag, max(2, n // 10))
    mis = [average_mutual_information(x, tau) for tau in range(1, max_lag + 1)]
    valid = [i for i, v in enumerate(mis) if np.isfinite(v)]
    if not valid:
        return 1
    for i in range(1, len(mis) - 1):
        if np.isfinite(mis[i]) and mis[i] < mis[i - 1] and mis[i] < mis[i + 1]:
            return i + 1
    return int(valid[int(np.argmin([mis[i] for i in valid]))]) + 1


def false_nearest_neighbors_fraction(
    x: np.ndarray,
    tau: int,
    m: int,
    *,
    rtol: float = 10.0,
    atol: float = 2.0,
    theiler: int | None = None,
) -> float:
    """Fraction of false nearest neighbors when increasing dimension m -> m+1."""
    theiler = max(tau, 1) if theiler is None else max(theiler, 1)
    y_m = delay_embedding(x, m, tau)
    y_mp1 = delay_embedding(x, m + 1, tau)
    n = min(len(y_m), len(y_mp1))
    if n < 20:
        return 1.0
    y_m = y_m[:n]
    y_mp1 = y_mp1[:n]

    dists_all = np.linalg.norm(y_m - np.mean(y_m, axis=0), axis=1)
    a_m = float(np.mean(dists_all)) if len(dists_all) else 0.0

    false = 0
    total = 0
    for i in range(n):
        d = np.linalg.norm(y_m - y_m[i], axis=1)
        lo = max(0, i - theiler)
        hi = min(n, i + theiler + 1)
        d[lo:hi] = np.inf
        j = int(np.argmin(d))
        if not np.isfinite(d[j]) or d[j] == 0:
            continue
        total += 1
        dm1 = abs(y_mp1[i, m] - y_mp1[j, m])
        if dm1 / d[j] > rtol or (a_m > 0 and dm1 / a_m > atol):
            false += 1
    return float(false / total) if total else 1.0


def estimate_embedding_dimension_fnn(
    x: np.ndarray,
    tau: int,
    *,
    m_min: int = 2,
    m_max: int = 10,
    fnn_threshold: float = 0.10,
    rtol: float = 10.0,
    atol: float = 2.0,
) -> int:
    """Smallest m with FNN fraction below fnn_threshold."""
    for m in range(m_min, m_max + 1):
        if false_nearest_neighbors_fraction(x, tau, m, rtol=rtol, atol=atol) < fnn_threshold:
            return m
    return m_max
