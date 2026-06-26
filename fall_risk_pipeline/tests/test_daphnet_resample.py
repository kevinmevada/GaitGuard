"""Tests for DAPHNET 64→100 Hz resample_poly + PSD verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.preprocessing.daphnet_resample import (
    DaphnetResampleError,
    RESAMPLE_DOWN,
    RESAMPLE_UP,
    band_peak_frequency,
    resample_axis,
    resample_daphnet_signals,
    run_psd_verification_batch,
    verify_trunk_z_resample,
)


def test_resample_poly_ratio_64_to_100():
    fs = 64.0
    duration_s = 20.0
    t = np.arange(0, duration_s, 1.0 / fs)
    sig = np.sin(2 * np.pi * 5.0 * t)
    out = resample_axis(sig, up=RESAMPLE_UP, down=RESAMPLE_DOWN)
    expected = int(round(len(sig) * RESAMPLE_UP / RESAMPLE_DOWN))
    assert abs(len(out) - expected) <= 1


def test_resample_per_axis_independent():
    fs = 64.0
    n = 640
    t = np.arange(n) / fs
    signals = {
        "ankle": __import__("pandas").DataFrame(
            {"time": t, "acc_x": t, "acc_y": t * 2, "acc_z": t * 3}
        ),
        "thigh": __import__("pandas").DataFrame(
            {"time": t, "acc_x": t, "acc_y": t, "acc_z": t}
        ),
        "trunk": __import__("pandas").DataFrame(
            {"time": t, "acc_x": t, "acc_y": t, "acc_z": np.sin(2 * np.pi * 5 * t)}
        ),
    }
    out = resample_daphnet_signals(signals)
    assert set(out) == {"ankle", "thigh", "trunk"}
    for sensor in out:
        assert len(out[sensor]) == len(out["trunk"])
        assert np.isclose(out[sensor]["time"].iloc[1] - out[sensor]["time"].iloc[0], 0.01)


def test_psd_peak_preserved_within_half_hz(tmp_path: Path):
    fs = 64.0
    duration_s = 30.0
    t = np.arange(0, duration_s, 1.0 / fs)
    trunk_z = np.sin(2 * np.pi * 5.0 * t) + 0.05 * np.random.default_rng(0).standard_normal(len(t))
    trunk_z_100 = resample_axis(trunk_z)

    result = verify_trunk_z_resample(
        trunk_z,
        trunk_z_100,
        subject_id="S01",
        figure_dir=tmp_path,
    )
    assert result.passed
    assert result.peak_shift_hz <= 0.5
    assert (tmp_path / "psd_check_S01.png").is_file()
    assert 4.5 <= result.peak_hz_before <= 5.5


def test_psd_failure_on_bad_resample(tmp_path: Path):
    fs = 64.0
    t = np.arange(0, 20, 1.0 / fs)
    before = np.sin(2 * np.pi * 5.0 * t)
    # Deliberately wrong target rate (80 Hz) — peak shifts out of tolerance
    t_bad = np.arange(0, 20, 1.0 / 80.0)
    after = np.sin(2 * np.pi * 6.5 * t_bad)
    with pytest.raises(DaphnetResampleError, match="PSD check failed"):
        verify_trunk_z_resample(
            before,
            after,
            subject_id="S99",
            figure_dir=tmp_path,
            write_figure=False,
        )


def test_batch_requires_two_subjects(tmp_path: Path):
    fs = 64.0
    t = np.arange(0, 20, 1.0 / fs)
    z = np.sin(2 * np.pi * 5.0 * t)
    z100 = resample_axis(z)
    with pytest.raises(DaphnetResampleError, match="requires ≥2 subjects"):
        run_psd_verification_batch([("S01", z, z100)], figure_dir=tmp_path)


def test_band_peak_frequency_finds_5hz():
    fs = 64.0
    t = np.arange(0, 30, 1.0 / fs)
    sig = np.sin(2 * np.pi * 5.0 * t)
    peak = band_peak_frequency(sig, fs)
    assert 4.5 <= peak <= 5.5
