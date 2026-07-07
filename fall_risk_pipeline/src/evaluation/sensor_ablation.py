"""
Sensor-position ablation study.

For each subset of the four IMU positions {head, lower_back, left_foot, right_foot},
train the reference model under LOSO cross-validation and report macro-OVR AUC.

Output
------
results/metrics/sensor_ablation.csv
    Columns: sensor_subset, n_features, auc_mean, auc_bootstrap_std, auc_ci_low, auc_ci_high,
    validation (loso_subject_grouped)
results/figures/models/sensor_ablation.{pdf,png}
    Bar chart comparing AUC across sensor subsets.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.base as skbase
from loguru import logger
from sklearn.metrics import roc_auc_score

from src.dataset.label_policy import is_binary_task
from src.evaluation.multiclass_metrics import predict_multiclass
from src.models.trainer import ModelTrainer
from src.features.feature_matrix import (
    intersect_nested_rfecv_columns,
    load_patient_feature_matrix,
)
from src.utils.checkpoint_io import CheckpointIntegrityError, load_checkpoint
from src.utils.reproducibility import get_pipeline_seed

SENSOR_POSITIONS = ["head", "lower_back", "left_foot", "right_foot"]

SENSOR_PREFIXES = {
    "head": re.compile(r"^head_"),
    "lower_back": re.compile(r"^lb_|^turn_"),
    "left_foot": re.compile(r"^left_"),
    "right_foot": re.compile(r"^right_"),
}

CROSS_SITE = re.compile(r"^head_lb_")
_PATIENT_AGG = r"(?:_(?:mean|std|range|trend))?$"
FOOT_BILATERAL = re.compile(
    rf"^(cadence_mean|stance_phase_ratio|swing_phase_ratio|double_support_ratio){_PATIENT_AGG}"
)
BILATERAL_ASYMMETRY = re.compile(
    rf"^(stride_time_mean_asymmetry|stride_time_std_asymmetry|asymmetry_rms_acc){_PATIENT_AGG}"
)


def _is_bilateral_foot_feature(name: str) -> bool:
    return bool(FOOT_BILATERAL.match(name) or BILATERAL_ASYMMETRY.match(name))


def _features_for_sensors(
    feat_names: list[str], sensors: tuple[str, ...]
) -> list[int]:
    has_left = "left_foot" in sensors
    has_right = "right_foot" in sensors
    indices = []
    for i, name in enumerate(feat_names):
        if CROSS_SITE.match(name):
            if "head" in sensors and "lower_back" in sensors:
                indices.append(i)
            continue
        if _is_bilateral_foot_feature(name):
            # Joint left/right foot metrics and bilateral asymmetry indices.
            if has_left and has_right:
                indices.append(i)
            continue
        matched = False
        for s in sensors:
            pat = SENSOR_PREFIXES.get(s)
            if pat and pat.match(name):
                matched = True
                break
        if matched:
            indices.append(i)
    return indices


def _bootstrap_binary_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int,
    random_state: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    idx_all = np.arange(len(y_true))
    samples: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        samples.append(float(roc_auc_score(yt, yp)))
    if not samples:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(samples)
    return (
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
        float(np.std(arr)),
    )


def _bootstrap_macro_auc_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bootstrap: int,
    random_state: int,
    labels: list[int],
) -> tuple[float, float, float]:
    rng = np.random.default_rng(random_state)
    idx_all = np.arange(len(y_true))
    samples: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        yt = y_true[idx]
        yp = y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            samples.append(
                float(
                    roc_auc_score(
                        yt,
                        yp,
                        multi_class="ovr",
                        average="macro",
                        labels=labels,
                    )
                )
            )
        except ValueError:
            continue
    if not samples:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(samples)
    return (
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
        float(np.std(arr)),
    )


def _loso_evaluate_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    checkpoint,
    config: dict,
    *,
    feat_cols: list[str],
    scenario_col_idx: list[int],
    n_bootstrap: int,
    random_state: int,
) -> dict[str, float]:
    """
    True leave-one-subject-out: one participant held out per fold, macro-OVR AUC
    on pooled out-of-fold predictions (same protocol as feature_ablation).
    """
    nan = {
        "auc": float("nan"),
        "auc_bootstrap_std": float("nan"),
        "auc_ci_low": float("nan"),
        "auc_ci_high": float("nan"),
    }
    if not scenario_col_idx:
        return nan

    binary = is_binary_task(y, config)
    all_true: list[int] = []
    all_probs: list[Any] = []
    trainer = ModelTrainer(config)
    model_name = str(config.get("ablation", {}).get("reference_model", "xgboost"))

    for subj in np.unique(groups):
        test_idx = np.where(groups == subj)[0]
        train_idx = np.where(groups != subj)[0]
        if len(np.unique(y[train_idx])) < 2:
            continue

        fold_cols = intersect_nested_rfecv_columns(
            config, X, y, groups, feat_cols, train_idx, scenario_col_idx
        )
        if not fold_cols:
            continue

        model = skbase.clone(checkpoint)
        trainer.fit_pipeline(model_name, model, X[train_idx][:, fold_cols], y[train_idx])

        if binary:
            proba = model.predict_proba(X[test_idx][:, fold_cols])
            score = proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
            all_probs.extend(score.tolist())
        else:
            proba, _ = predict_multiclass(model, X[test_idx][:, fold_cols])
            all_probs.append(proba)

        all_true.extend(y[test_idx].tolist())

    y_true = np.asarray(all_true, dtype=int)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return nan

    if binary:
        y_prob = np.asarray(all_probs, dtype=float)
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
        ci_low, ci_high, auc_bootstrap_std = _bootstrap_binary_auc_ci(
            y_true, y_prob, n_bootstrap, random_state
        )
    else:
        y_proba = np.vstack(all_probs) if all_probs else np.empty((0, 0))
        labels = sorted(np.unique(y_true).tolist())
        try:
            auc = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
        except ValueError:
            auc = float("nan")
        ci_low, ci_high, auc_bootstrap_std = _bootstrap_macro_auc_ci(
            y_true, y_proba, n_bootstrap, random_state, labels
        )

    return {
        "auc": auc,
        "auc_bootstrap_std": auc_bootstrap_std,
        "auc_ci_low": ci_low,
        "auc_ci_high": ci_high,
    }


class SensorAblationStudy:
    def __init__(self, config: dict):
        self.config = config
        self.seed = get_pipeline_seed(config)
        ab_cfg = config.get("ablation", {})
        self.reference_model = str(ab_cfg.get("reference_model", "xgboost"))
        self.n_bootstrap = int(ab_cfg.get("n_bootstrap", 1000))

        self.ckpt_dir = Path(config["paths"]["checkpoints"])
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.fig_dir = Path(config["paths"]["figures_models"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.dpi = int(config.get("reporting", {}).get("figure_dpi", 300))
        self.fmt = config.get("reporting", {}).get("figure_format", "pdf")

    def run(self) -> pd.DataFrame:
        X, y, groups, feat_names, df = load_patient_feature_matrix(
            self.config, use_selected=False
        )

        ckpt_path = self.ckpt_dir / f"{self.reference_model}.pkl"
        if not ckpt_path.exists():
            logger.error(f"Checkpoint {ckpt_path} not found — run train first.")
            return pd.DataFrame()
        try:
            checkpoint = load_checkpoint(
                ckpt_path,
                manifest_dir=self.ckpt_dir,
                require_manifest=False,
            )
        except CheckpointIntegrityError as exc:
            logger.error("Checkpoint verification failed for %s: %s", self.reference_model, exc)
            return pd.DataFrame()

        subsets = []
        for r in range(1, len(SENSOR_POSITIONS) + 1):
            for combo in combinations(SENSOR_POSITIONS, r):
                subsets.append(combo)

        rows = []
        logger.info(f"Sensor ablation: {len(subsets)} subsets × LOSO CV")
        for combo in subsets:
            col_idx = _features_for_sensors(feat_names, combo)
            if not col_idx:
                logger.warning(f"No features for {combo} — skipping")
                continue

            label = "+".join(s.replace("_", "") for s in combo)
            metrics = _loso_evaluate_auc(
                X,
                y,
                groups,
                checkpoint,
                self.config,
                feat_cols=feat_names,
                scenario_col_idx=col_idx,
                n_bootstrap=self.n_bootstrap,
                random_state=self.seed,
            )

            if np.isnan(metrics["auc"]):
                continue

            rows.append({
                "sensor_subset": label,
                "sensors": ", ".join(combo),
                "n_sensors": len(combo),
                "n_features": len(col_idx),
                "auc_mean": metrics["auc"],
                "auc_bootstrap_std": metrics["auc_bootstrap_std"],
                "auc_ci_low": metrics["auc_ci_low"],
                "auc_ci_high": metrics["auc_ci_high"],
                "validation": "loso_subject_grouped",
            })
            logger.info(
                f"  {label:40s}  feats={len(col_idx):3d}  "
                f"AUC={metrics['auc']:.4f} "
                f"[{metrics['auc_ci_low']:.3f}, {metrics['auc_ci_high']:.3f}]"
            )

        result_df = pd.DataFrame(rows).sort_values("auc_mean", ascending=False)
        out_path = self.metrics_dir / "sensor_ablation.csv"
        result_df.to_csv(out_path, index=False)
        logger.info(f"Sensor ablation saved -> {out_path}")

        self._plot(result_df)
        return result_df

    def _plot(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        df_sorted = df.sort_values("auc_mean", ascending=True)
        labels = df_sorted["sensor_subset"].values
        aucs = df_sorted["auc_mean"].values
        errs = df_sorted["auc_bootstrap_std"].values
        n_sensors = df_sorted["n_sensors"].values

        colors = {1: "#e74c3c", 2: "#f39c12", 3: "#2ecc71", 4: "#3498db"}
        bar_colors = [colors.get(n, "#95a5a6") for n in n_sensors]

        fig, ax = plt.subplots(figsize=(10, max(6, len(labels) * 0.4)))
        bars = ax.barh(
            range(len(labels)), aucs, xerr=errs,
            color=bar_colors, edgecolor="white", capsize=3
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Macro OvR AUC (leave-one-subject-out)")
        ax.set_title("Sensor Position Ablation Study")
        ax.axvline(x=aucs[-1], color="gray", linestyle="--", alpha=0.5, label="All sensors")
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in aucs], padding=3, fontsize=7)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors[1], label="1 sensor"),
            Patch(facecolor=colors[2], label="2 sensors"),
            Patch(facecolor=colors[3], label="3 sensors"),
            Patch(facecolor=colors[4], label="4 sensors"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        for ext in [self.fmt, "png"]:
            fig.savefig(
                self.fig_dir / f"sensor_ablation.{ext}",
                dpi=self.dpi, bbox_inches="tight",
            )
        plt.close(fig)


def run_sensor_ablation(config: dict) -> pd.DataFrame:
    return SensorAblationStudy(config).run()
