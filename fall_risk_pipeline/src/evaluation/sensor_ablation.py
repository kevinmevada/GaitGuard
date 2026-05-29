"""
Sensor-position ablation study.

For each subset of the four IMU positions {head, lower_back, left_foot, right_foot},
train the reference model under LOSO cross-validation and report macro-OVR AUC.

Output
------
results/metrics/sensor_ablation.csv
    Columns: sensor_subset, n_features, auc_mean, auc_std, auc_ci_low, auc_ci_high
results/figures/models/sensor_ablation.{pdf,png}
    Bar chart comparing AUC across sensor subsets.
"""

from __future__ import annotations

import pickle
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.base as skbase
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset.label_policy import is_binary_task
from src.features.feature_matrix import load_patient_feature_matrix
from src.utils.reproducibility import get_pipeline_seed

SENSOR_POSITIONS = ["head", "lower_back", "left_foot", "right_foot"]

SENSOR_PREFIXES = {
    "head": re.compile(r"^head_"),
    "lower_back": re.compile(r"^lb_|^turn_"),
    "left_foot": re.compile(
        r"^left_|^cadence_mean|^stance_phase_ratio|^swing_phase_ratio"
        r"|^double_support_ratio|^stride_time_mean_asymmetry"
        r"|^stride_time_std_asymmetry|^asymmetry_rms_acc"
    ),
    "right_foot": re.compile(r"^right_"),
}

CROSS_SITE = re.compile(r"^head_lb_")


def _features_for_sensors(
    feat_names: list[str], sensors: tuple[str, ...]
) -> list[int]:
    indices = []
    for i, name in enumerate(feat_names):
        if CROSS_SITE.match(name):
            if "head" in sensors and "lower_back" in sensors:
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
        elif not any(p.match(name) for p in SENSOR_PREFIXES.values()):
            if "left_foot" in sensors or "right_foot" in sensors:
                indices.append(i)
    return indices


def _loso_auc(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    checkpoint,
    config: dict,
    seed: int,
) -> list[float]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    binary = is_binary_task(y, config)
    fold_aucs = []

    for train_idx, test_idx in cv.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = skbase.clone(checkpoint)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])
        try:
            if binary:
                auc = roc_auc_score(y[test_idx], proba[:, 1])
            else:
                auc = roc_auc_score(
                    y[test_idx], proba, multi_class="ovr", average="macro"
                )
        except ValueError:
            continue
        fold_aucs.append(auc)
    return fold_aucs


class SensorAblationStudy:
    def __init__(self, config: dict):
        self.config = config
        self.seed = get_pipeline_seed(config)
        ab_cfg = config.get("ablation", {})
        self.reference_model = str(ab_cfg.get("reference_model", "xgboost"))

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
        with open(ckpt_path, "rb") as f:
            checkpoint = pickle.load(f)

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

            X_sub = X[:, col_idx]
            label = "+".join(s.replace("_", "") for s in combo)
            fold_aucs = _loso_auc(X_sub, y, groups, checkpoint, self.config, self.seed)

            if not fold_aucs:
                continue

            auc_arr = np.array(fold_aucs)
            rows.append({
                "sensor_subset": label,
                "sensors": ", ".join(combo),
                "n_sensors": len(combo),
                "n_features": len(col_idx),
                "auc_mean": float(np.mean(auc_arr)),
                "auc_std": float(np.std(auc_arr)),
                "auc_ci_low": float(np.percentile(auc_arr, 2.5)),
                "auc_ci_high": float(np.percentile(auc_arr, 97.5)),
            })
            logger.info(
                f"  {label:40s}  feats={len(col_idx):3d}  "
                f"AUC={np.mean(auc_arr):.4f} +/- {np.std(auc_arr):.4f}"
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
        errs = df_sorted["auc_std"].values
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
        ax.set_xlabel("Macro OvR AUC (LOSO)")
        ax.set_title("Sensor Position Ablation Study")
        ax.axvline(x=aucs[-1], color="gray", linestyle="--", alpha=0.5, label="All sensors")

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
