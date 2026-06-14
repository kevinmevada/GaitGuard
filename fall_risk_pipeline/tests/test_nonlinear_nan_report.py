"""HIGH-01: nonlinear metric NaN telemetry and visible computation failures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import nonlinear_metrics as nm


@pytest.fixture(autouse=True)
def _reset_nonlinear_warnings():
    nm._NONLINEAR_FAILURE_WARNINGS.clear()
    yield
    nm._NONLINEAR_FAILURE_WARNINGS.clear()


def test_computation_failures_log_at_warning_not_debug():
    source = (Path(__file__).resolve().parents[1] / "src" / "features" / "nonlinear_metrics.py").read_text(
        encoding="utf-8"
    )
    assert "_warn_computation_failure" in source
    assert "logger.debug(f\"lyap_r failed" not in source
    assert "logger.debug(f\"sample_entropy failed" not in source
    assert "logger.debug(f\"dfa failed" not in source


def test_warn_computation_failure_deduplicates_by_exception_type():
    nm._warn_computation_failure("sample_entropy", ValueError("bad tolerance"))
    nm._warn_computation_failure("sample_entropy", ValueError("other message"))
    assert nm._NONLINEAR_FAILURE_WARNINGS == {("sample_entropy", "ValueError")}


def test_write_nonlinear_nan_report_overall_and_by_cohort(tmp_path):
    trial_df = pd.DataFrame({
        "cohort": ["Healthy", "Healthy", "MS", "MS"],
        "lb_sampen": [1.0, np.nan, 1.1, np.nan],
        "lb_dfa": [0.9, 0.8, np.nan, np.nan],
        "head_sampen": [1.0, 1.0, np.nan, 1.0],
        "head_dfa": [0.7, np.nan, 0.6, 0.6],
        "lb_lyapunov": [0.01, 0.02, np.nan, 0.03],
        "head_lyapunov": [0.02, np.nan, 0.03, 0.04],
    })

    nm.write_nonlinear_nan_report(trial_df, tmp_path)
    out = tmp_path / "nonlinear_nan_report.csv"
    assert out.is_file()

    report = pd.read_csv(out)
    overall = report.loc[report["scope"] == "overall"].set_index("feature")["nan_rate"]
    assert overall["lb_sampen"] == pytest.approx(0.5)
    assert overall["lb_dfa"] == pytest.approx(0.5)

    ms_sampen = report.loc[
        (report["scope"] == "cohort") & (report["cohort"] == "MS") & (report["feature"] == "lb_sampen"),
        "nan_rate",
    ].iloc[0]
    assert ms_sampen == pytest.approx(0.5)

    healthy_dfa = report.loc[
        (report["scope"] == "cohort") & (report["cohort"] == "Healthy") & (report["feature"] == "lb_dfa"),
        "nan_rate",
    ].iloc[0]
    assert healthy_dfa == pytest.approx(0.0)


def test_sample_entropy_failure_returns_nan_and_warns(monkeypatch):
    class BrokenAnt:
        @staticmethod
        def sample_entropy(*args, **kwargs):
            raise RuntimeError("internal antropy failure")

    monkeypatch.setitem(__import__("sys").modules, "antropy", BrokenAnt())

    val = nm.sample_entropy(np.linspace(0, 1, 300), {"min_length": 200})
    assert np.isnan(val)
    assert ("sample_entropy", "RuntimeError") in nm._NONLINEAR_FAILURE_WARNINGS
