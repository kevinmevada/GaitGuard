"""
``fit_uncertainty`` pipeline stage.

Fits post-hoc probability calibration and a split-conformal prediction-set
threshold using the LOSO out-of-fold predictions the ``evaluate`` stage
already writes to ``oof_predictions.parquet`` (or ``.csv`` fallback — see
:func:`_load_oof_frame`). This stage **does not retrain, refit, or rerun
any model** — it only fits a small post-hoc mapping on predictions that
already exist, and is safe to run (or re-run) independently of the rest
of the pipeline.

Config (``calibration:`` section, all optional — sensible defaults if absent):

.. code-block:: yaml

    calibration:
      enabled: true
      model: null              # null => use the model with the most OOF rows
      conformal_alpha: 0.1      # target miscoverage rate (90% coverage)

Outputs (written to ``paths.metrics``):

- ``calibration_artifact.json`` — isotonic calibration mapping
- ``conformal_artifact.json`` — split-conformal threshold
- ``uncertainty_coverage_report.json`` — empirical coverage sanity check
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.dataset.label_policy import get_dataset_label_config
from src.evaluation.uncertainty import (
    coverage_report,
    fit_conformal_threshold,
    fit_isotonic_calibrator,
)
from src.utils.progress import progress_bar


def _cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("calibration") or {}


def _load_oof_frame(metrics_dir: Path) -> pd.DataFrame:
    parquet_path = metrics_dir / "oof_predictions.parquet"
    csv_path = metrics_dir / "oof_predictions.csv"
    if parquet_path.is_file():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            logger.warning(
                f"{parquet_path} exists but no parquet engine (pyarrow/fastparquet) "
                "is installed; falling back to CSV if available."
            )
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Neither {parquet_path} nor {csv_path} found — run the 'evaluate' stage first "
        "(fit_uncertainty consumes its OOF predictions; it does not train anything itself)."
    )


def _select_model(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in set(df["model"]):
            raise ValueError(
                f"Requested model {requested!r} not found in OOF predictions; "
                f"available models: {sorted(set(df['model']))}"
            )
        return requested
    # Default: the model with the most OOF rows (a reasonable, deterministic
    # default when the user hasn't specified one).
    counts = df["model"].value_counts()
    return str(counts.idxmax())


def _extract_y_true_prob(df: pd.DataFrame, label_mode: str) -> tuple[np.ndarray, np.ndarray]:
    y_true = df["y_true"].to_numpy(dtype=int)
    if label_mode == "binary":
        if "y_prob" not in df.columns:
            raise ValueError(
                "label_mode is 'binary' but OOF frame has no 'y_prob' column for this model."
            )
        return y_true, df["y_prob"].to_numpy(dtype=float)

    class_cols = sorted(
        [c for c in df.columns if c.startswith("y_prob_class_")],
        key=lambda c: int(c.rsplit("_", 1)[-1]),
    )
    if not class_cols:
        raise ValueError(
            "label_mode is 'multiclass' but OOF frame has no 'y_prob_class_*' columns "
            "for this model."
        )
    y_prob = df[class_cols].to_numpy(dtype=float)
    return y_true, y_prob


def run_fit_uncertainty(config: dict) -> dict[str, Any]:
    cfg = _cfg(config)
    if not cfg.get("enabled", True):
        logger.info("fit_uncertainty stage disabled in config")
        return {}

    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    df = _load_oof_frame(metrics_dir)
    if "model" not in df.columns:
        raise ValueError("oof_predictions has no 'model' column — cannot select a model.")

    model_name = _select_model(df, cfg.get("model"))
    model_df = df[df["model"] == model_name].reset_index(drop=True)
    logger.info(f"fit_uncertainty: using model={model_name!r} ({len(model_df)} OOF rows)")

    label_mode = get_dataset_label_config(config)["label_mode"]
    y_true, y_prob = _extract_y_true_prob(model_df, label_mode)

    if len(y_true) < 30:
        logger.warning(
            f"fit_uncertainty: only {len(y_true)} OOF rows for model={model_name!r} — "
            "calibration/conformal fits on this few points are themselves unstable; "
            "treat the resulting artifacts as provisional."
        )

    # Split OOF rows in half: first half to fit, second half to sanity-check
    # coverage on unseen (to the calibration step) rows. This is a coverage
    # *diagnostic*, not a strict requirement of split-conformal fitting
    # (which is already valid when fit directly on OOF rows — see
    # uncertainty.py's fit_conformal_threshold docstring).
    n = len(y_true)
    rng = np.random.default_rng(config.get("reproducibility", {}).get("seed", 42))
    perm = rng.permutation(n)
    half = n // 2
    fit_idx, check_idx = perm[:half] if half > 0 else perm, perm[half:] if half > 0 else perm

    steps = progress_bar(total=3, desc="fit_uncertainty", unit="step")

    calibration_artifact = fit_isotonic_calibrator(
        y_true[fit_idx], y_prob[fit_idx] if y_prob.ndim == 1 else y_prob[fit_idx], label_mode=label_mode
    )
    calibration_path = metrics_dir / "calibration_artifact.json"
    calibration_artifact.to_json(calibration_path)
    logger.info(f"Calibration artifact saved → {calibration_path}")
    steps.update(1)

    alpha = float(cfg.get("conformal_alpha", 0.1))
    conformal_artifact = fit_conformal_threshold(
        y_true[fit_idx], y_prob[fit_idx] if y_prob.ndim == 1 else y_prob[fit_idx], alpha=alpha, label_mode=label_mode
    )
    conformal_path = metrics_dir / "conformal_artifact.json"
    conformal_artifact.to_json(conformal_path)
    logger.info(f"Conformal artifact saved → {conformal_path} (q_hat={conformal_artifact.q_hat:.4f})")
    steps.update(1)

    report: dict[str, Any] = {"model": model_name, "label_mode": label_mode, "n_oof_rows": n}
    if len(check_idx) >= 10:
        cov = coverage_report(
            conformal_artifact,
            y_true[check_idx],
            y_prob[check_idx] if y_prob.ndim == 1 else y_prob[check_idx],
        )
        report["coverage_check"] = cov
        logger.info(
            f"Held-out coverage check: target={cov['target_coverage']:.2f} "
            f"empirical={cov['empirical_coverage']:.3f} "
            f"mean_set_size={cov['mean_set_size']:.2f}"
        )
    else:
        logger.warning("Too few rows for a held-out coverage check; skipping diagnostic.")

    report_path = metrics_dir / "uncertainty_coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    steps.update(1)
    steps.close()
    return report
