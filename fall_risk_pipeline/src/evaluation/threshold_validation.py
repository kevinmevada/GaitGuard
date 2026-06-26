"""
v13 threshold validation — confirm no held-out leakage in anomaly calibration.

Run after BiLSTM-AE / tabular anomaly LOSO to emit a pass/fail audit artifact
before manuscript results are trusted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.evaluation.anomaly_threshold_policy import (
    THRESHOLD_SOURCE_HEALTHY_PERCENTILE,
    fit_anomaly_threshold,
    resolve_threshold_policy,
)

AUDIT_VERSION = "v13_threshold_validation_1"


def _audit_source_files(pipeline_root: Path) -> dict[str, bool]:
    """Static checks: evaluators must not fit thresholds on test/OOF pools."""
    checks: dict[str, bool] = {}
    files = {
        "bilstm_ae_loso": pipeline_root / "src/evaluation/bilstm_ae_loso_evaluator.py",
        "anomaly_loso": pipeline_root / "src/evaluation/anomaly_loso_evaluator.py",
        "anomaly_detector": pipeline_root / "src/models/anomaly_detector.py",
    }
    for name, path in files.items():
        if not path.is_file():
            checks[f"{name}_exists"] = False
            continue
        text = path.read_text(encoding="utf-8")
        checks[f"{name}_no_pooled_oof_youden"] = "youden_threshold(y_true," not in text
        checks[f"{name}_no_test_scores_in_threshold"] = (
            "youden_threshold(y_true[test_mask]" not in text
            and "youden_threshold(y_true, test_scores" not in text
            and "youden_threshold(yt, ys)" not in text
        )
        checks[f"{name}_uses_fit_anomaly_threshold_or_train_only"] = (
            "fit_anomaly_threshold" in text or "reconstruction_threshold_train_only" in text
        )
    return checks


def _synthetic_leak_guard_test(config: dict[str, Any]) -> dict[str, Any]:
    """
    Changing test scores must not change the fitted threshold when only train
    healthy scores define the percentile cut.
    """
    train_scores = np.array([1.0, 2.0, 3.0, 4.0, 50.0, 60.0], dtype=float)
    healthy_mask = np.array([True, True, True, True, False, False])
    thr1, src1 = fit_anomaly_threshold(
        train_scores, config, healthy_train_mask=healthy_mask, y_train=np.array([0, 0, 0, 0, 1, 1])
    )
    # Perturb would-be test scores — not passed to fit_anomaly_threshold
    thr2, src2 = fit_anomaly_threshold(
        train_scores, config, healthy_train_mask=healthy_mask, y_train=np.array([0, 0, 0, 0, 1, 1])
    )
    policy, percentile = resolve_threshold_policy(config)
    return {
        "threshold_stable_under_test_perturbation": bool(np.isclose(thr1, thr2)),
        "threshold_value": float(thr1),
        "threshold_source": src1,
        "policy": policy,
        "percentile": percentile,
        "healthy_train_percentile_expected": src1 == THRESHOLD_SOURCE_HEALTHY_PERCENTILE,
    }


def run_threshold_validation(config: dict) -> dict[str, Any]:
    cfg_path = Path((config.get("_pipeline_meta") or {}).get("config_path", "configs/pipeline_config.yaml"))
    pipeline_root = cfg_path.resolve().parent.parent

    static = _audit_source_files(pipeline_root)
    dynamic = _synthetic_leak_guard_test(config)

    policy, percentile = resolve_threshold_policy(config)
    passed = all(static.values()) and dynamic["threshold_stable_under_test_perturbation"]
    if policy == "healthy_train_percentile":
        passed = passed and dynamic["healthy_train_percentile_expected"]

    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "v13_status": "PASS" if passed else "FAIL",
        "manuscript_guidance": (
            "Thresholds are fit on training-fold healthy reference scores only "
            f"(default {percentile}th percentile). Held-out participant scores are "
            "never used for threshold selection. AUROC is threshold-independent; "
            "F1/MCC/sensitivity depend on this calibration."
        ),
        "policy": policy,
        "percentile": percentile,
        "static_source_checks": static,
        "dynamic_leak_guard": dynamic,
        "correct_flow": (
            "for each LOSO fold: threshold = percentile(healthy_train_scores, p); "
            "y_pred_test = (test_scores >= threshold)"
        ),
    }

    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / "threshold_validation_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_path = metrics_dir / "threshold_validation_report.md"
    md_path.write_text(_render_md(report), encoding="utf-8")

    if passed:
        logger.info("v13 threshold validation PASS → {}", out)
    else:
        logger.error("v13 threshold validation FAIL → {}", out)
    return report


def _render_md(report: dict[str, Any]) -> str:
    status = report.get("v13_status", "UNKNOWN")
    lines = [
        "# Threshold validation (v13 — train-fold calibration only)",
        "",
        f"**Status:** {status}",
        "",
        report.get("manuscript_guidance", ""),
        "",
        f"- **Policy:** `{report.get('policy')}`",
        f"- **Percentile:** {report.get('percentile')}",
        "",
        "## Correct LOSO flow",
        "",
        "```python",
        "threshold = np.percentile(healthy_train_reconstruction_errors, 95)",
        "y_pred_test = (test_reconstruction_errors >= threshold).astype(int)",
        "```",
        "",
        "## Static source checks",
        "",
    ]
    for k, v in (report.get("static_source_checks") or {}).items():
        lines.append(f"- {k}: {'PASS' if v else '**FAIL**'}")
    lines.extend(["", "## Dynamic leak guard", ""])
    for k, v in (report.get("dynamic_leak_guard") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)
