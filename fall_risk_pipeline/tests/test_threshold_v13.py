"""v13 threshold validation — no held-out leakage in anomaly calibration."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_ROOT = REPO_ROOT / "fall_risk_pipeline"


def test_healthy_percentile_ignores_pathological_train_scores():
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.anomaly_threshold_policy import (
        THRESHOLD_SOURCE_HEALTHY_PERCENTILE,
        fit_anomaly_threshold,
    )

    train_scores = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 200.0], dtype=float)
    healthy_mask = np.array([True, True, True, True, False, False], dtype=bool)
    config = {
        "primary_model": {
            "bilstm_ae_ensemble": {
                "anomaly_threshold": {"policy": "healthy_train_percentile", "percentile": 95.0}
            }
        }
    }
    thr, src = fit_anomaly_threshold(
        train_scores, config, healthy_train_mask=healthy_mask, y_train=np.array([0, 0, 0, 0, 1, 1])
    )
    assert src == THRESHOLD_SOURCE_HEALTHY_PERCENTILE
    assert thr < 10.0


def test_bilstm_loso_source_never_fits_threshold_on_test_scores():
    source = (PIPELINE_ROOT / "src/evaluation/bilstm_ae_loso_evaluator.py").read_text(encoding="utf-8")
    assert "fit_anomaly_threshold" in source
    assert "youden_threshold(y_true," not in source
    assert "test_scores[method]" not in source.split("fit_anomaly_threshold")[0]


def test_run_threshold_validation_passes_default_config(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.threshold_validation import run_threshold_validation

    config = {
        "paths": {"metrics": str(tmp_path / "metrics")},
        "_pipeline_meta": {
            "config_path": str(PIPELINE_ROOT / "configs" / "pipeline_config.yaml"),
        },
        "primary_model": {
            "bilstm_ae_ensemble": {
                "anomaly_threshold": {"policy": "healthy_train_percentile", "percentile": 95.0}
            }
        },
    }
    report = run_threshold_validation(config)
    assert report["v13_status"] == "PASS"
    assert (tmp_path / "metrics" / "threshold_validation_report.json").is_file()


def test_bilstm_loso_applies_train_threshold_to_test_only(tmp_path):
    sys.path.insert(0, str(PIPELINE_ROOT))
    from src.evaluation.bilstm_ae_loso_evaluator import run_bilstm_ae_loso_evaluation
    from src.models.bilstm_ae_scoring import METHOD_ENSEMBLE

    config = {
        "paths": {
            "processed_data": str(tmp_path / "data" / "processed"),
            "metrics": str(tmp_path / "results" / "metrics"),
            "checkpoints": str(tmp_path / "checkpoints"),
        },
        "reproducibility": {"seed": 42},
        "primary_model": {
            "bilstm_ae_ensemble": {
                "enabled": True,
                "anomaly_threshold": {"policy": "healthy_train_percentile", "percentile": 95.0},
            }
        },
        "deep_learning": {"sequence_length": 20, "overlap": 0.5},
        "dataset": {"sensor_positions": ["head", "lower_back", "left_foot", "right_foot"]},
    }

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "test_bilstm_ae_loso_eval",
        PIPELINE_ROOT / "tests" / "test_bilstm_ae_loso_eval.py",
    )
    loso_test = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loso_test)
    bundle = loso_test._synthetic_bundle()

    from src.models.bilstm_ae_scoring import (
        METHOD_AE_RECON,
        METHOD_IF_LATENT,
        METHOD_OCSVM_LATENT,
    )

    def fake_fold_scores(bundle, train_tids, test_tids, healthy_train_tids, config, **kwargs):
        n_tr, n_te = len(train_tids), len(test_tids)
        healthy_idx = [i for i, t in enumerate(train_tids) if t in healthy_train_tids]
        tr = np.zeros(n_tr)
        tr[healthy_idx] = np.linspace(0.1, 0.4, len(healthy_idx))
        tr[[i for i in range(n_tr) if i not in healthy_idx]] = 0.9
        te = np.full(n_te, 0.95)
        train_methods = {
            METHOD_AE_RECON: tr,
            METHOD_IF_LATENT: tr,
            METHOD_OCSVM_LATENT: tr,
            METHOD_ENSEMBLE: tr,
        }
        test_methods = {
            METHOD_AE_RECON: te,
            METHOD_IF_LATENT: te,
            METHOD_OCSVM_LATENT: te,
            METHOD_ENSEMBLE: te,
        }
        return train_methods, test_methods

    from unittest.mock import patch

    with (
        patch(
            "src.evaluation.bilstm_ae_loso_evaluator.load_voisard_trial_windows",
            return_value=bundle,
        ),
        patch(
            "src.evaluation.bilstm_ae_loso_evaluator.build_fold_trial_scores",
            side_effect=fake_fold_scores,
        ),
    ):
        metrics_df = run_bilstm_ae_loso_evaluation(config)

    ens = metrics_df.loc[metrics_df["method"] == METHOD_ENSEMBLE].iloc[0]
    assert ens["threshold_source"] == "loso_healthy_train_percentile"
    validation = (Path(config["paths"]["metrics"]) / "threshold_validation_report.json").read_text()
    assert '"v13_status": "PASS"' in validation
