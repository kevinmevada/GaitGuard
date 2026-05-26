"""Synthetic validation for Lyapunov exponent and approximate entropy."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.features.nonlinear_metrics import (
    approximate_entropy,
    largest_lyapunov_exponent,
    validate_nonlinear_metrics,
)


@pytest.fixture(scope="module")
def nonlinear_cfg() -> tuple[dict, dict]:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pipeline_config.yaml"
    with open(config_path, encoding="utf-8") as fh:
        features = yaml.safe_load(fh)["features"]
    return features["lyapunov"], features["approximate_entropy"]


def _validation_lyap_cfg(lyap_cfg: dict) -> dict:
    """Fixed embedding for reproducible Rosenstein checks (production uses AMI/FNN)."""
    cfg = dict(lyap_cfg)
    cfg["validation_fixed_embedding_dim"] = cfg.get("validation_fixed_embedding_dim", 3)
    cfg["validation_fixed_lag"] = cfg.get("validation_fixed_lag", 2)
    return cfg


def test_sine_lyapunov_near_zero(nonlinear_cfg):
    lyap_cfg, _ = nonlinear_cfg
    n = int(lyap_cfg.get("validation_n_samples", 8000))
    fs = 100.0
    t = np.arange(n) / fs
    sine = np.sin(2 * np.pi * 0.5 * t)
    lam = largest_lyapunov_exponent(sine, _validation_lyap_cfg(lyap_cfg))
    assert np.isfinite(lam)
    assert abs(lam) < lyap_cfg.get("validation_sine_lyap_max", 0.02)


def test_chaotic_logistic_lyapunov_positive(nonlinear_cfg):
    lyap_cfg, _ = nonlinear_cfg
    n = int(lyap_cfg.get("validation_n_samples", 8000))
    logistic = np.zeros(n)
    logistic[0] = 0.2
    for i in range(1, n):
        logistic[i] = 3.9 * logistic[i - 1] * (1.0 - logistic[i - 1])
    lam = largest_lyapunov_exponent(logistic, _validation_lyap_cfg(lyap_cfg))
    assert lam > lyap_cfg.get("validation_chaotic_lyap_min", 0.05)


def test_chaotic_lyapunov_exceeds_sine(nonlinear_cfg):
    lyap_cfg, _ = nonlinear_cfg
    n = int(lyap_cfg.get("validation_n_samples", 8000))
    fs = 100.0
    t = np.arange(n) / fs
    sine = np.sin(2 * np.pi * 0.5 * t)
    logistic = np.zeros(n)
    logistic[0] = 0.2
    for i in range(1, n):
        logistic[i] = 3.9 * logistic[i - 1] * (1.0 - logistic[i - 1])
    vcfg = _validation_lyap_cfg(lyap_cfg)
    lam_sine = abs(largest_lyapunov_exponent(sine, vcfg))
    lam_log = largest_lyapunov_exponent(logistic, vcfg)
    factor = lyap_cfg.get("validation_chaotic_vs_sine_factor", 3.0)
    assert lam_log > factor * lam_sine


def test_apen_noise_greater_than_sine(nonlinear_cfg):
    _, apen_cfg = nonlinear_cfg
    rng = np.random.default_rng(42)
    n, fs = 4000, 100.0
    t = np.arange(n) / fs
    sine = np.sin(2 * np.pi * 0.5 * t)
    noise = rng.standard_normal(n)
    apen_sine = approximate_entropy(sine, apen_cfg)
    apen_noise = approximate_entropy(noise, apen_cfg)
    assert np.isfinite(apen_sine) and np.isfinite(apen_noise)
    factor = apen_cfg.get("validation_noise_vs_sine_factor", 1.5)
    assert apen_noise > factor * apen_sine


def test_validate_nonlinear_metrics_bundle(nonlinear_cfg):
    lyap_cfg, apen_cfg = nonlinear_cfg
    rows, ok = validate_nonlinear_metrics(lyap_cfg, apen_cfg)
    assert ok
    checks = [r for r in rows if str(r["signal"]).startswith("check:")]
    assert len(checks) == 4
    assert all(r.get("passed") for r in checks)
