"""
Cross-cohort transfer analysis (Leave-One-Cohort-Out).

Trains on all subjects from N-1 cohorts and tests on the held-out cohort.
Produces an 8x8 transfer matrix showing how well each training set
generalises to unseen pathologies.

Output
------
results/metrics/cross_cohort_transfer.csv
    Columns: test_cohort, n_train, n_test, auc, accuracy, f1_macro, ...
results/metrics/cross_cohort_pairwise.csv
    Columns: train_cohort, test_cohort, auc, f1_macro, accuracy, ...
results/figures/models/cross_cohort_transfer.{pdf,png}
    LOCO bar chart (macro OvR AUC).
results/figures/models/cross_cohort_pairwise.{pdf,png}
    Pairwise heatmap (macro-F1 by default; accuracy is misleading under imbalance).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.base as skbase
import xgboost as xgb
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

from src.dataset.label_balance import balanced_scale_pos_weight
from src.dataset.label_policy import is_binary_task
from src.features.feature_matrix import (
    load_patient_feature_matrix,
    nested_rfecv_column_indices,
)
from src.models.trainer import ModelTrainer
from src.utils.checkpoint_io import load_checkpoint
from src.utils.progress import progress_bar
from src.utils.reproducibility import get_pipeline_seed


def _xgboost_kwargs_for_y(y: np.ndarray, config: dict) -> dict:
    y = np.asarray(y).astype(int)
    n_classes = len(np.unique(y))
    if n_classes <= 2 and is_binary_task(y, config):
        return {
            "scale_pos_weight": balanced_scale_pos_weight(y),
            "eval_metric": "logloss",
        }
    if n_classes <= 2:
        return {"objective": "binary:logistic", "eval_metric": "logloss"}
    return {
        "objective": "multi:softprob",
        "num_class": int(n_classes),
        "eval_metric": "mlogloss",
    }


def _multiclass_sample_weights(y: np.ndarray, config: dict) -> np.ndarray | None:
    return ModelTrainer._xgb_sample_weights(y, config)


def _rebuild_xgb_pipeline(pipeline, y_train: np.ndarray, config: dict):
    """Reconfigure XGBoost in a cloned pipeline for the labels seen in this fold."""
    y_train = np.asarray(y_train).astype(int)
    if not hasattr(pipeline, "named_steps") or "clf" not in pipeline.named_steps:
        return pipeline
    clf = pipeline.named_steps["clf"]
    if not isinstance(clf, xgb.XGBClassifier):
        return pipeline

    skip = {"objective", "num_class", "eval_metric", "scale_pos_weight", "n_jobs"}
    params = {
        k: v
        for k, v in clf.get_params().items()
        if k not in skip and v is not None
    }
    new_clf = xgb.XGBClassifier(
        **params,
        n_jobs=1,
        **_xgboost_kwargs_for_y(y_train, config),
    )
    pipeline.set_params(clf=new_clf)
    return pipeline


def _fit_transfer_model(checkpoint, X_train, y_train, config: dict):
    """Fit with contiguous class indices so XGBoost matches the training subset."""
    model = skbase.clone(checkpoint)
    y_train = np.asarray(y_train).astype(int)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    model = _rebuild_xgb_pipeline(model, y_enc, config)
    weights = _multiclass_sample_weights(y_enc, config)
    if weights is not None:
        model.fit(X_train, y_enc, clf__sample_weight=weights)
    else:
        model.fit(X_train, y_enc)
    return model, le


def _score_transfer(
    model,
    le: LabelEncoder,
    X_test,
    y_test,
    binary: bool,
) -> tuple[float, float, float, float, str]:
    y_test = np.asarray(y_test).astype(int)
    known = np.isin(y_test, le.classes_)
    if int(known.sum()) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), "insufficient_known_samples"

    y_enc = le.transform(y_test[known])
    proba = model.predict_proba(X_test[known])
    preds = model.predict(X_test[known])
    mean_true_class_proba = float(np.mean(proba[np.arange(len(y_enc)), y_enc]))
    auc_status = "ok"

    try:
        if len(np.unique(y_enc)) < 2:
            auc = float("nan")
            auc_status = "undefined_single_class_test"
        elif binary and proba.shape[1] == 2:
            auc = roc_auc_score(y_enc, proba[:, 1])
        else:
            auc = roc_auc_score(
                y_enc, proba, multi_class="ovr", average="macro"
            )
    except ValueError:
        auc = float("nan")
        auc_status = "error_auc_computation"

    acc = accuracy_score(y_enc, preds)
    f1 = f1_score(y_enc, preds, average="macro", zero_division=0)
    return auc, acc, f1, mean_true_class_proba, auc_status


class CrossCohortTransfer:
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
        self.cohort_auc_min_n = int(
            config.get("models", {}).get("evaluation", {}).get("cohort_auc_min_n", 25)
        )

    def run(self) -> pd.DataFrame:
        X, y, groups, feat_names, df = load_patient_feature_matrix(
            self.config, use_selected=False
        )
        cohorts = df["cohort"].astype(str).values
        binary = is_binary_task(y, self.config)

        ckpt_path = self.ckpt_dir / f"{self.reference_model}.pkl"
        if not ckpt_path.exists():
            logger.error(f"Checkpoint {ckpt_path} not found — run train first.")
            return pd.DataFrame()
        checkpoint = load_checkpoint(
            ckpt_path, manifest_dir=ckpt_path.parent, require_manifest=False
        )

        unique_cohorts = sorted(np.unique(cohorts))
        logger.info(
            f"Cross-cohort transfer: {len(unique_cohorts)} cohorts, "
            f"model={self.reference_model}, nested_rfecv_per_train_fold=True"
        )

        rows = []

        for held_out in progress_bar(
            unique_cohorts, desc="cross_cohort loco", unit="cohort"
        ):
            train_mask = cohorts != held_out
            test_mask = cohorts == held_out

            n_train = int(train_mask.sum())
            n_test = int(test_mask.sum())

            if n_test < 2 or n_train < 5:
                logger.warning(f"Skipping {held_out}: n_train={n_train}, n_test={n_test}")
                continue

            y_train, y_test = y[train_mask], y[test_mask]
            if len(np.unique(y_train)) < 2:
                logger.warning(f"Skipping {held_out}: only one class in training set")
                continue

            col_idx = nested_rfecv_column_indices(
                self.config, X, y, groups, feat_names, train_mask
            )
            X_train = X[train_mask][:, col_idx]
            X_test = X[test_mask][:, col_idx]

            model, le = _fit_transfer_model(
                checkpoint, X_train, y_train, self.config
            )
            auc, acc, f1, mean_true_class_proba, auc_status = _score_transfer(
                model, le, X_test, y_test, binary
            )
            if auc_status == "ok" and n_test < self.cohort_auc_min_n:
                auc_status = "unstable_small_n"

            train_cohorts = ", ".join(
                c for c in unique_cohorts if c != held_out
            )
            rows.append({
                "test_cohort": held_out,
                "train_cohorts": train_cohorts,
                "n_train": n_train,
                "n_test": n_test,
                "n_features": len(col_idx),
                "n_train_classes": len(np.unique(y_train)),
                "n_test_classes": len(np.unique(y_test)),
                "auc": auc,
                "auc_status": auc_status,
                "mean_true_class_proba": mean_true_class_proba,
                "accuracy": acc,
                "f1_macro": f1,
                "feature_selection_protocol": "nested_rfecv_per_train_fold",
            })
            logger.info(
                f"  Hold-out {held_out:10s}  n={n_test:3d}  "
                f"AUC={auc:.4f} ({auc_status})  "
                f"TrueP={mean_true_class_proba:.4f}  Acc={acc:.4f}  F1={f1:.4f}"
            )

        pairwise_rows = []
        pairwise_pairs = [
            (train_cohort, test_cohort)
            for train_cohort in unique_cohorts
            for test_cohort in unique_cohorts
            if train_cohort != test_cohort
        ]
        for train_cohort, test_cohort in progress_bar(
            pairwise_pairs, desc="cross_cohort pairwise", unit="pair"
        ):
                train_mask = cohorts == train_cohort
                test_mask = cohorts == test_cohort

                y_tr = y[train_mask]
                y_te = y[test_mask]
                n_train = int(train_mask.sum())
                n_test = int(test_mask.sum())

                empty_row = {
                    "train_cohort": train_cohort,
                    "test_cohort": test_cohort,
                    "n_train": n_train,
                    "n_test": n_test,
                    "auc": float("nan"),
                    "auc_status": "skipped",
                    "mean_true_class_proba": float("nan"),
                    "accuracy": float("nan"),
                    "f1_macro": float("nan"),
                }

                if len(np.unique(y_tr)) < 2 or n_test < 2:
                    pairwise_rows.append(empty_row)
                    continue

                col_idx = nested_rfecv_column_indices(
                    self.config, X, y, groups, feat_names, train_mask
                )
                X_tr = X[train_mask][:, col_idx]
                X_te = X[test_mask][:, col_idx]

                model, le = _fit_transfer_model(
                    checkpoint, X_tr, y_tr, self.config
                )
                auc, acc, f1, mean_true_class_proba, auc_status = _score_transfer(
                    model, le, X_te, y_te, binary
                )
                if auc_status == "ok" and n_test < self.cohort_auc_min_n:
                    auc_status = "unstable_small_n"
                if auc_status == "ok" and n_train < self.cohort_auc_min_n:
                    auc_status = "unstable_small_n"
                    auc = float("nan")

                pairwise_rows.append({
                    "train_cohort": train_cohort,
                    "test_cohort": test_cohort,
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_features": len(col_idx),
                    "auc": auc,
                    "auc_status": auc_status,
                    "mean_true_class_proba": mean_true_class_proba,
                    "accuracy": acc,
                    "f1_macro": f1,
                    "feature_selection_protocol": "nested_rfecv_per_train_fold",
                })

        loco_df = pd.DataFrame(rows)
        out_path = self.metrics_dir / "cross_cohort_transfer.csv"
        loco_df.to_csv(out_path, index=False)
        logger.info(f"Cross-cohort transfer saved -> {out_path}")

        pairwise_df = pd.DataFrame(pairwise_rows)
        pair_path = self.metrics_dir / "cross_cohort_pairwise.csv"
        pairwise_df.to_csv(pair_path, index=False)
        logger.info(f"Pairwise transfer matrix saved -> {pair_path}")

        self._plot_loco(loco_df)
        self._plot_pairwise_heatmap(pairwise_df, unique_cohorts, metric="f1_macro")
        self._plot_pairwise_heatmap(
            pairwise_df,
            unique_cohorts,
            metric="auc",
            outfile_stem="cross_cohort_pairwise_auc",
        )
        return loco_df

    def _plot_loco(self, df: pd.DataFrame) -> None:
        if df.empty:
            return

        df_sorted = df.sort_values("auc", ascending=True)
        cohorts = df_sorted["test_cohort"].values
        aucs = df_sorted["auc"].values

        tier_colors = {
            "Healthy": "#2ecc71",
            "HipOA": "#3498db", "KneeOA": "#3498db", "ACL": "#3498db",
            "PD": "#e74c3c", "CVA": "#e74c3c", "CIPN": "#e74c3c", "RIL": "#e74c3c",
        }
        colors = [tier_colors.get(c, "#95a5a6") for c in cohorts]

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(range(len(cohorts)), aucs, color=colors, edgecolor="white")
        ax.set_yticks(range(len(cohorts)))
        ax.set_yticklabels(cohorts, fontsize=10)
        ax.set_xlabel("Macro OvR AUC")
        ax.set_title("Leave-One-Cohort-Out Transfer (test on held-out cohort)")
        ax.set_xlim(0, 1.05)

        for i, v in enumerate(aucs):
            if np.isfinite(v):
                ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#2ecc71", label="Healthy"),
            Patch(facecolor="#3498db", label="Orthopedic"),
            Patch(facecolor="#e74c3c", label="Neurological"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

        plt.tight_layout()
        for ext in [self.fmt, "png"]:
            fig.savefig(
                self.fig_dir / f"cross_cohort_transfer.{ext}",
                dpi=self.dpi, bbox_inches="tight",
            )
        plt.close(fig)

    def _plot_pairwise_heatmap(
        self,
        df: pd.DataFrame,
        cohorts: list[str],
        *,
        metric: str = "f1_macro",
        outfile_stem: str | None = None,
    ) -> None:
        if df.empty or metric not in df.columns:
            if not df.empty:
                logger.warning(
                    "Pairwise heatmap skipped — metric %r not in dataframe columns",
                    metric,
                )
            return

        metric_labels = {
            "f1_macro": ("Macro-F1", "Pairwise Cross-Cohort Transfer (Macro-F1)"),
            "accuracy": ("Accuracy", "Pairwise Cross-Cohort Transfer (Accuracy)"),
            "auc": ("Macro OvR AUC", "Pairwise Cross-Cohort Transfer (Macro OvR AUC)"),
        }
        colorbar_label, title = metric_labels.get(
            metric, (metric, f"Pairwise Cross-Cohort Transfer ({metric})")
        )
        stem = outfile_stem or (
            "cross_cohort_pairwise"
            if metric == "f1_macro"
            else f"cross_cohort_pairwise_{metric}"
        )

        matrix = np.full((len(cohorts), len(cohorts)), np.nan)
        cohort_idx = {c: i for i, c in enumerate(cohorts)}

        for _, row in df.iterrows():
            i = cohort_idx.get(row["train_cohort"])
            j = cohort_idx.get(row["test_cohort"])
            if i is not None and j is not None:
                matrix[i, j] = row[metric]

        np.fill_diagonal(matrix, np.nan)

        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(len(cohorts)))
        ax.set_xticklabels(cohorts, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(len(cohorts)))
        ax.set_yticklabels(cohorts, fontsize=9)
        ax.set_xlabel("Test Cohort")
        ax.set_ylabel("Train Cohort")
        ax.set_title(title)

        for i in range(len(cohorts)):
            for j in range(len(cohorts)):
                val = matrix[i, j]
                if np.isfinite(val):
                    color = "white" if val < 0.4 or val > 0.8 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

        plt.colorbar(im, ax=ax, label=colorbar_label, shrink=0.8)
        plt.tight_layout()
        for ext in [self.fmt, "png"]:
            fig.savefig(
                self.fig_dir / f"{stem}.{ext}",
                dpi=self.dpi, bbox_inches="tight",
            )
        plt.close(fig)


def run_cross_cohort_transfer(config: dict) -> pd.DataFrame:
    return CrossCohortTransfer(config).run()
