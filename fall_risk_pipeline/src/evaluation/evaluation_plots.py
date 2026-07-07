"""
Evaluation plotting, extracted from the ``Evaluator`` god-class.

This is a deliberately *partial* first slice of decomposing
``src/evaluation/evaluator.py`` (2,500+ lines, 70+ methods before this
extraction). It covers the plotting methods that depend only on
matplotlib/seaborn/sklearn — genuinely stateless given already-computed
results, with no coupling to fold assembly, ensembling, or threshold
selection logic.

**What was deliberately left in ``evaluator.py`` and why:** the three
SHAP-plotting methods (``_plot_shap_per_class_bars``, ``_plot_shap_summary``,
``_plot_shap_mean_bar``) were not moved here. They depend on the optional
``shap`` package, which is not available in the environment this
refactor was authored and tested in — moving code I cannot execute and
verify is a correctness risk, not a maintainability improvement. They
remain natural candidates for a second extraction pass once someone can
verify the moved SHAP code against a real ``shap`` install.

Usage from ``Evaluator``:

.. code-block:: python

    self._plotter = EvaluationPlotter(
        fig_dir=self.fig_models, metrics_dir=self.metrics_dir,
        fmt=self.fmt, dpi=self.dpi,
    )
    ...
    self._plotter.plot_roc_all(results)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from src.dataset.label_policy import MULTICLASS_NAMES
from src.evaluation.multiclass_metrics import is_multiclass_metric_result


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (Naeini et al., AAAI 2015)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_acc = float(y_true[mask].mean())
        ece += (count / n) * abs(avg_acc - avg_conf)
    return float(ece)


def row_normalized_confusion_matrix(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, np.maximum(row_sums, 1.0))


def confusion_matrix_tick_labels(n_cls: int, *, multiclass: bool) -> list[str]:
    if multiclass:
        return [MULTICLASS_NAMES.get(i, str(i)) for i in range(n_cls)]
    if n_cls == 2:
        return ["Low risk", "High risk"]
    return [str(i) for i in range(n_cls)]


class EvaluationPlotter:
    """Stateless (modulo output paths) evaluation figure generation.

    Takes already-computed per-model result dicts — never touches folds,
    training, or ensembling. Every method is independently callable and
    independently testable without constructing an ``Evaluator``.
    """

    OVR_COLORS = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"]

    def __init__(self, *, fig_dir: Path, metrics_dir: Path, fmt: str = "png", dpi: int = 150):
        self.fig_dir = Path(fig_dir)
        self.metrics_dir = Path(metrics_dir)
        self.fmt = fmt
        self.dpi = dpi
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, fig, name: str) -> None:
        for ext in {self.fmt, "png"}:
            fig.savefig(self.fig_dir / f"{name}.{ext}", dpi=self.dpi)
        plt.close(fig)

    def plot_roc_all(self, results: dict) -> None:
        fig, ax = plt.subplots()
        for name, res in results.items():
            ax.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.legend()
        ax.set_title("Nested Grouped ROC Curves")
        self._save(fig, "roc")

    def plot_pr_curves(self, results: dict) -> None:
        fig, ax = plt.subplots()
        for name, res in results.items():
            if is_multiclass_metric_result(res) or "y_prob" not in res:
                continue
            prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
            ax.plot(rec, prec, label=name)
        ax.legend()
        ax.set_title("Nested Grouped PR Curves")
        self._save(fig, "pr")

    def plot_calibration(self, results: dict) -> None:
        fig, ax = plt.subplots()
        for name, res in results.items():
            if is_multiclass_metric_result(res) or "y_prob" not in res:
                continue
            try:
                frac_pos, mean_pred = calibration_curve(res["y_true"], res["y_prob"], n_bins=10)
                ax.plot(mean_pred, frac_pos, label=name)
            except Exception as exc:
                logger.warning(f"Calibration curve failed for model {name}: {exc}")
                continue
        ax.legend()
        ax.set_title("Nested Grouped Calibration")
        self._save(fig, "calibration")

    def plot_multiclass_calibration(self, results: dict, *, top_features_plot: int = 20) -> None:
        """Per-class OvR reliability diagrams + Brier score for multiclass."""
        for name, res in results.items():
            y_true = np.asarray(res["y_true"]).astype(int)
            y_proba = res.get("y_proba_full")
            if y_proba is None:
                continue
            y_proba = np.asarray(y_proba, dtype=float)
            labels = sorted(set(np.unique(y_true)))
            n_classes = len(labels)

            fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 4.5), squeeze=False)
            axes = axes[0]

            for i, lbl in enumerate(labels):
                ax = axes[i]
                class_name = MULTICLASS_NAMES.get(lbl, str(lbl))
                y_bin = (y_true == lbl).astype(int)
                p_bin = y_proba[:, lbl] if lbl < y_proba.shape[1] else np.zeros(len(y_true))

                try:
                    frac_pos, mean_pred = calibration_curve(y_bin, p_bin, n_bins=10)
                    ax.plot(mean_pred, frac_pos, "s-", color=self.OVR_COLORS[i % len(self.OVR_COLORS)], lw=2)
                except (ValueError, IndexError):
                    pass

                ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Fraction of positives")
                ax.set_title(class_name, fontsize=10)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.02, 1.02)

                brier = float(np.mean((p_bin - y_bin) ** 2))
                ece = expected_calibration_error(y_bin, p_bin)
                ax.text(0.05, 0.9, f"Brier={brier:.3f}\nECE={ece:.3f}", transform=ax.transAxes, fontsize=9)

            fig.suptitle(f"{name} — per-class calibration (OvR)", fontsize=12)
            fig.tight_layout()
            self._save(fig, f"calibration_ovr_{name}")

        self._save_brier_table(results)

    def _save_brier_table(self, results: dict) -> None:
        best_name = max(results, key=lambda n: results[n]["auc"])
        best_res = results[best_name]
        y_true = np.asarray(best_res["y_true"]).astype(int)
        y_proba = best_res.get("y_proba_full")
        if y_proba is None:
            return
        y_proba = np.asarray(y_proba, dtype=float)
        labels = sorted(set(np.unique(y_true)))
        brier_rows = []
        for lbl in labels:
            class_name = MULTICLASS_NAMES.get(lbl, str(lbl))
            y_bin = (y_true == lbl).astype(int)
            p_bin = y_proba[:, lbl] if lbl < y_proba.shape[1] else np.zeros(len(y_true))
            brier_rows.append({
                "model": best_name,
                "class": class_name,
                "brier_score": float(np.mean((p_bin - y_bin) ** 2)),
                "ece": expected_calibration_error(y_bin, p_bin),
                "mean_predicted_prob": float(np.mean(p_bin)),
                "prevalence": float(np.mean(y_bin)),
            })
        if brier_rows:
            self.metrics_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(brier_rows).to_csv(
                self.metrics_dir / "calibration_brier_scores.csv", index=False
            )
            logger.info("Brier + ECE scores saved → calibration_brier_scores.csv")

    def plot_confusion_matrices(self, results: dict) -> None:
        for name, res in results.items():
            cm = np.asarray(res["confusion_matrix"])
            is_mc = is_multiclass_metric_result(res)
            class_names = confusion_matrix_tick_labels(cm.shape[0], multiclass=is_mc)
            cm_norm = row_normalized_confusion_matrix(cm)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm_norm,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax,
                vmin=0,
                vmax=1,
                cbar_kws={"label": "Row-normalized rate"},
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"{name} — normalized confusion matrix (LOSO OOF)")
            fig.tight_layout()
            self._save(fig, f"cm_{name}")

    def plot_multiclass_roc(self, results: dict) -> None:
        """Per-class OvR ROC curves for every model (multiclass)."""
        for name, res in results.items():
            per_class_roc = res.get("per_class_roc")
            if not per_class_roc:
                continue

            fig, ax = plt.subplots(figsize=(7, 6))
            for i, (lbl, roc_data) in enumerate(sorted(per_class_roc.items())):
                c = self.OVR_COLORS[i % len(self.OVR_COLORS)]
                ax.plot(
                    roc_data["fpr"], roc_data["tpr"],
                    color=c, lw=2,
                    label=f"{roc_data['name']} (AUC={roc_data['auc']:.3f})",
                )
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            auc_macro = res.get("auc", float("nan"))
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{name} — OvR ROC (macro AUC={auc_macro:.3f})")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            self._save(fig, f"roc_ovr_{name}")

        best_name = max(results, key=lambda n: results[n]["auc"])
        best_roc = results[best_name].get("per_class_roc")
        if best_roc:
            fig, ax = plt.subplots(figsize=(7, 6))
            for i, (lbl, roc_data) in enumerate(sorted(best_roc.items())):
                c = self.OVR_COLORS[i % len(self.OVR_COLORS)]
                ax.plot(
                    roc_data["fpr"], roc_data["tpr"],
                    color=c, lw=2,
                    label=f"{roc_data['name']} (AUC={roc_data['auc']:.3f})",
                )
            ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"Best model ({best_name}) — per-class OvR ROC")
            ax.legend(loc="lower right", fontsize=8)
            fig.tight_layout()
            self._save(fig, "roc_ovr_best")
