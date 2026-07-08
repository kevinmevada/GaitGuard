"""Tests for src/evaluation/uncertainty.py — post-hoc calibration and
split-conformal prediction sets, fit purely on existing OOF predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"
sys.path.insert(0, str(PIPELINE_ROOT))

from src.evaluation.uncertainty import (  # noqa: E402
    CalibrationArtifact,
    ConformalArtifact,
    apply_calibrator,
    conformal_prediction_set,
    coverage_report,
    fit_conformal_threshold,
    fit_isotonic_calibrator,
)


def _miscalibrated_binary(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    true_prob = np.clip(rng.beta(2, 2, size=n) * 0.6 + y_true * 0.2, 0.01, 0.99)
    raw_prob = np.clip(true_prob**0.3, 0.01, 0.99)  # overconfident distortion
    return y_true, raw_prob


def _multiclass_probs(n=2000, k=3, seed=1, signal=2.0, noise=0.5):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k)) * noise
    for i in range(n):
        logits[i, y_true[i]] += rng.normal(signal, 0.8)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    return y_true, probs


def test_binary_calibration_reduces_miscalibration_gap():
    y_true, raw_prob = _miscalibrated_binary()
    art = fit_isotonic_calibrator(y_true, raw_prob, label_mode="binary")
    calibrated = apply_calibrator(art, raw_prob)

    bins = np.linspace(0, 1, 11)
    bin_idx = np.clip(np.digitize(calibrated, bins) - 1, 0, 9)
    gap_before, gap_after, total = 0.0, 0.0, 0
    for b in range(10):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        emp = y_true[mask].mean()
        gap_before += abs(raw_prob[mask].mean() - emp) * mask.sum()
        gap_after += abs(calibrated[mask].mean() - emp) * mask.sum()
        total += mask.sum()
    assert gap_after / total < gap_before / total


def test_calibration_artifact_json_roundtrip(tmp_path):
    y_true, raw_prob = _miscalibrated_binary()
    art = fit_isotonic_calibrator(y_true, raw_prob, label_mode="binary")
    path = tmp_path / "cal.json"
    art.to_json(path)
    art2 = CalibrationArtifact.from_json(path)
    assert np.allclose(apply_calibrator(art, raw_prob), apply_calibrator(art2, raw_prob))


def test_multiclass_calibration_preserves_shape_and_sums_to_one():
    y_true, probs = _multiclass_probs()
    art = fit_isotonic_calibrator(y_true, probs, label_mode="multiclass")
    calibrated = apply_calibrator(art, probs)
    assert calibrated.shape == probs.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)


def test_multiclass_conformal_achieves_approximate_target_coverage():
    y_true, probs = _multiclass_probs(n=3000)
    cal_idx = np.arange(0, 1500)
    test_idx = np.arange(1500, 3000)

    for alpha in (0.05, 0.1, 0.2):
        conf = fit_conformal_threshold(
            y_true[cal_idx], probs[cal_idx], alpha=alpha, label_mode="multiclass"
        )
        report = coverage_report(conf, y_true[test_idx], probs[test_idx])
        # Split-conformal guarantees coverage >= 1 - alpha in expectation;
        # allow a small finite-sample tolerance.
        assert report["empirical_coverage"] >= (1 - alpha) - 0.03


def test_conformal_prediction_sets_are_never_empty():
    y_true, probs = _multiclass_probs(n=1000, signal=0.3, noise=1.5)  # weak signal
    conf = fit_conformal_threshold(y_true[:500], probs[:500], alpha=0.1, label_mode="multiclass")
    sets = conformal_prediction_set(conf, probs[500:])
    assert all(len(s) >= 1 for s in sets)


def test_conformal_artifact_json_roundtrip(tmp_path):
    y_true, probs = _multiclass_probs()
    conf = fit_conformal_threshold(y_true[:1000], probs[:1000], alpha=0.1, label_mode="multiclass")
    path = tmp_path / "conformal.json"
    conf.to_json(path)
    conf2 = ConformalArtifact.from_json(path)
    assert conf2.q_hat == conf.q_hat
    assert conf2.alpha == conf.alpha


def test_binary_conformal_path():
    rng = np.random.default_rng(3)
    y_bin = rng.integers(0, 2, size=1000)
    p_bin = np.clip(rng.beta(2, 2, size=1000) * 0.5 + y_bin * 0.3, 0.02, 0.98)
    conf = fit_conformal_threshold(y_bin[:500], p_bin[:500], alpha=0.1, label_mode="binary")
    sets = conformal_prediction_set(conf, p_bin[500:510])
    assert all(len(s) >= 1 for s in sets)


def test_conformal_prediction_set_rejects_class_count_mismatch():
    """Regression test: applying a conformal artifact fit on K classes to a
    probability vector with a different number of classes must raise, not
    silently produce a meaningless prediction set."""
    y_true, probs = _multiclass_probs(n=1000, k=3)
    conf = fit_conformal_threshold(y_true[:500], probs[:500], alpha=0.1, label_mode="multiclass")
    assert conf.n_classes == 3

    wrong_shape = np.array([[0.4, 0.6]])  # only 2 classes
    try:
        conformal_prediction_set(conf, wrong_shape)
        raised = False
    except ValueError:
        raised = True
    assert raised, "Expected a ValueError on class-count mismatch"
