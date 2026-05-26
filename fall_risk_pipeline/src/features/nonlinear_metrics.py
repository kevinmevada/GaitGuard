"""
Nonlinear time-series metrics for trunk accelerometry.

- Largest Lyapunov exponent: AMI + FNN embedding, Rosenstein via ``nolds.lyap_r``.
- Approximate entropy (ApEn): ``antropy.app_entropy`` (Pincus 1991).

Synthetic validation: ``validate_nonlinear_metrics()`` (see tests).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from src.features.delay_embedding import (
    estimate_embedding_dimension_fnn,
    estimate_tau_ami,
)


def _prepare_signal(signal: np.ndarray, min_length: int) -> np.ndarray | None:
    x = np.asarray(signal, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < min_length:
        return None
    return x


def largest_lyapunov_exponent(signal: np.ndarray, cfg: dict | None = None) -> float:
    """Largest Lyapunov exponent (Rosenstein et al. 1993) with AMI/FNN embedding."""
    cfg = cfg or {}
    x = _prepare_signal(signal, int(cfg.get("min_length", 200)))
    if x is None:
        return float("nan")

    fixed_m = cfg.get("validation_fixed_embedding_dim") or cfg.get("fixed_embedding_dim")
    fixed_tau = cfg.get("validation_fixed_lag") or cfg.get("fixed_lag")
    if fixed_m is not None and fixed_tau is not None:
        m, tau = int(fixed_m), int(fixed_tau)
    else:
        max_lag = int(cfg.get("max_lag", 50))
        m_max = int(cfg.get("max_embedding_dim", 10))
        tau = estimate_tau_ami(x, max_lag=max_lag)
        m = estimate_embedding_dimension_fnn(
            x,
            tau,
            m_min=int(cfg.get("min_embedding_dim", 2)),
            m_max=m_max,
            fnn_threshold=float(cfg.get("fnn_threshold", 0.10)),
            rtol=float(cfg.get("fnn_rtol", 10.0)),
            atol=float(cfg.get("fnn_atol", 2.0)),
        )

    if cfg.get("use_nolds", True):
        try:
            import nolds

            return float(nolds.lyap_r(x, emb_dim=m, lag=tau))
        except ImportError:
            logger.warning(
                "nolds not installed; using capped Rosenstein fallback for Lyapunov"
            )
        except Exception as exc:
            logger.debug(f"lyap_r failed (m={m}, tau={tau}): {exc}")

    return _lyapunov_rosenstein_fallback(x, m=m, tau=tau, cfg=cfg)


def _lyapunov_rosenstein_fallback(
    signal: np.ndarray, m: int, tau: int, cfg: dict
) -> float:
    N = len(signal)
    if N < m * tau + 50:
        return float("nan")

    n_pts = N - (m - 1) * tau
    emb = np.column_stack([signal[i : i + n_pts] for i in range(0, m * tau, tau)])

    divs = []
    n_anchors = min(n_pts, int(cfg.get("max_anchors", 200)))
    theiler = max(10, tau)
    for i in range(n_anchors):
        dists = np.linalg.norm(emb - emb[i], axis=1)
        lo = max(0, i - theiler)
        hi = min(n_pts, i + theiler + 1)
        dists[lo:hi] = np.inf
        j = int(np.argmin(dists))
        if not np.isfinite(dists[j]):
            continue
        steps = min(20, n_pts - max(i, j) - 1)
        if steps > 0:
            future_dists = [
                np.linalg.norm(emb[i + s] - emb[j + s]) + 1e-10
                for s in range(steps)
            ]
            divs.append(float(np.mean(np.log(future_dists))))

    return float(np.mean(divs)) if divs else float("nan")


def approximate_entropy(signal: np.ndarray, cfg: dict | None = None) -> float:
    """Approximate entropy via antropy (m=order, r=tolerance fraction of std)."""
    cfg = cfg or {}
    x = _prepare_signal(signal, int(cfg.get("min_length", 200)))
    if x is None:
        return float("nan")

    try:
        import antropy as ant
    except ImportError:
        logger.warning("antropy not installed; lb_apen will be NaN")
        return float("nan")

    order = int(cfg.get("order", 2))
    metric = str(cfg.get("metric", "chebyshev"))
    tolerance = cfg.get("tolerance")
    if tolerance is None:
        std = float(np.std(x))
        tolerance = 0.2 * std if std > 1e-12 else 1e-6

    try:
        return float(ant.app_entropy(x, order=order, metric=metric, tolerance=tolerance))
    except Exception as exc:
        logger.debug(f"app_entropy failed: {exc}")
        return float("nan")


def _synthetic_signals(n: int = 4000, fs: float = 100.0, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    sine = np.sin(2 * np.pi * 0.5 * t)
    noise = rng.standard_normal(n)

    logistic = np.zeros(n)
    logistic[0] = 0.2
    for i in range(1, n):
        logistic[i] = 3.9 * logistic[i - 1] * (1.0 - logistic[i - 1])

    return {"sine": sine, "white_noise": noise, "logistic_map": logistic}


def validate_nonlinear_metrics(
    lyap_cfg: dict | None = None,
    apen_cfg: dict | None = None,
    *,
    n_samples: int | None = None,
    fs: float = 100.0,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Sanity-check Lyapunov and ApEn on synthetic signals.

    Lyapunov (fixed m=3, tau=2 for reproducibility): periodic sine → |λ| ≈ 0;
    chaotic logistic map → λ clearly above sine.
    i.i.d. noise is not used as a positive-λ benchmark for Rosenstein estimators.
    ApEn: white noise > sine (higher complexity).
    """
    lyap_cfg = dict(lyap_cfg or {})
    apen_cfg = apen_cfg or {}
    lyap_cfg.setdefault("validation_fixed_embedding_dim", 3)
    lyap_cfg.setdefault("validation_fixed_lag", 2)
    n_samples = int(n_samples or lyap_cfg.get("validation_n_samples", 8000))
    signals = _synthetic_signals(n=n_samples, fs=fs, seed=seed)

    rows: list[dict[str, Any]] = []
    for name, sig in signals.items():
        rows.append({
            "signal": name,
            "lyapunov": largest_lyapunov_exponent(sig, lyap_cfg),
            "approximate_entropy": approximate_entropy(sig, apen_cfg),
            "n_samples": n_samples,
        })

    by_name = {r["signal"]: r for r in rows}
    lam_sine = abs(float(by_name["sine"]["lyapunov"]))
    lam_log = float(by_name["logistic_map"]["lyapunov"])
    apen_sine = float(by_name["sine"]["approximate_entropy"])
    apen_noise = float(by_name["white_noise"]["approximate_entropy"])

    sine_lyap_ok = lam_sine < float(lyap_cfg.get("validation_sine_lyap_max", 0.02))
    chaotic_lyap_ok = lam_log > float(lyap_cfg.get("validation_chaotic_lyap_min", 0.05))
    chaotic_vs_sine_ok = lam_log > max(
        float(lyap_cfg.get("validation_chaotic_vs_sine_factor", 3.0)) * lam_sine,
        float(lyap_cfg.get("validation_chaotic_lyap_min", 0.05)),
    )
    apen_ok = apen_noise > float(apen_cfg.get("validation_noise_vs_sine_factor", 1.5)) * apen_sine

    checks = {
        "sine_lyapunov_near_zero": sine_lyap_ok,
        "logistic_lyapunov_positive": chaotic_lyap_ok,
        "logistic_lyapunov_gt_sine": chaotic_vs_sine_ok,
        "apen_noise_gt_sine": apen_ok,
    }
    for key, passed in checks.items():
        rows.append({
            "signal": f"check:{key}",
            "lyapunov": float("nan"),
            "approximate_entropy": float("nan"),
            "n_samples": n_samples,
            "passed": passed,
        })

    all_passed = all(checks.values())
    return rows, all_passed
