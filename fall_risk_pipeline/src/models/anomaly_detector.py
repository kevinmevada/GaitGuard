"""
src/models/anomaly_detector.py
Unsupervised anomaly detection for gait patterns.

FIXES APPLIED:
  1. StandardScaler is now fit only on "healthy" (normal) training samples and
     then applied to all data — mimicking a proper one-class setup. Previously
     scaler.fit_transform(X_all) leaked test statistics into the scaling step.
  2. Each anomaly score vector is independently min-max normalised to [0,1]
     BEFORE averaging in the ensemble, so no single method dominates by virtue
     of its raw score range.
  3. _analyze_by_cohort is called once inside _visualize_results instead of
     twice (redundant second call removed).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

console = Console()


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise an anomaly score vector to [0, 1]."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class GaitAnomalyDetector:
    """
    Unsupervised anomaly detection for gait patterns.
    Identifies unusual walking patterns that deviate from normal gait.
    """

    def __init__(self, config: dict):
        self.config = config
        self.feat_dir = Path(config["paths"]["features"])
        self.results_dir = Path(config["paths"]["results"]) / "anomaly_detection"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.anomaly_scores: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        logger.info("Starting anomaly detection analysis...")

        X, metadata = self._load_data()

        # FIX: fit scalers only on the "Healthy" (normal) cohort rows so that
        # test-sample statistics do not influence the scaling step.
        healthy_mask = (metadata["cohort"] == "Healthy").values
        if healthy_mask.sum() < 5:
            logger.warning(
                "Fewer than 5 Healthy samples — falling back to full-dataset scaling. "
                "Results may be optimistic."
            )
            healthy_mask = np.ones(len(X), dtype=bool)

        results: Dict[str, Any] = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} tasks"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Running anomaly detection...", total=6)

            progress.update(task_id, description="Anomaly method: isolation_forest")
            results["isolation_forest"] = self._isolation_forest_detection(X, metadata, healthy_mask)
            progress.advance(task_id)

            progress.update(task_id, description="Anomaly method: lof")
            results["lof"] = self._lof_detection(X, metadata, healthy_mask)
            progress.advance(task_id)

            progress.update(task_id, description="Anomaly method: one_class_svm")
            results["one_class_svm"] = self._one_class_svm_detection(X, metadata, healthy_mask)
            progress.advance(task_id)

            progress.update(task_id, description="Combining anomaly ensemble")
            results["ensemble"] = self._ensemble_detection(results, metadata)
            progress.advance(task_id)

            progress.update(task_id, description="Analyzing cohorts and visualizing")
            cohort_analysis = self._analyze_by_cohort(results, metadata)
            self._visualize_results(results, cohort_analysis, X, metadata)
            progress.advance(task_id)

            progress.update(task_id, description="Saving anomaly outputs")
            self._save_results(results, cohort_analysis, metadata)
            progress.advance(task_id)

            progress.update(task_id, description="Anomaly detection complete")

        logger.info("Anomaly detection completed!")
        return {
            "detection_results": results,
            "cohort_analysis": cohort_analysis,
            "summary": self._generate_summary(results, cohort_analysis),
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> Tuple[np.ndarray, pd.DataFrame]:
        df = pd.read_parquet(self.feat_dir / "trial_features.parquet")

        meta_cols = [
            "trial_id", "participant_id", "cohort", "risk_label",
            "fall_probability", "laterality_biased",
        ]

        feature_cols = [col for col in df.columns if col not in meta_cols]
        feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        X = X.values
        metadata = df[meta_cols].copy()

        logger.info(f"Loaded {X.shape[0]} trials with {X.shape[1]} features")
        return X, metadata

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    def _isolation_forest_detection(
        self, X: np.ndarray, metadata: pd.DataFrame, healthy_mask: np.ndarray
    ) -> Dict[str, Any]:
        # FIX: fit scaler on normal (healthy) samples only.
        scaler = StandardScaler()
        scaler.fit(X[healthy_mask])
        X_scaled = scaler.transform(X)

        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100,
        )
        iso_forest.fit(X_scaled[healthy_mask])          # train on normals only
        anomaly_labels = iso_forest.predict(X_scaled)
        anomaly_scores = iso_forest.decision_function(X_scaled)

        anomaly_binary = (anomaly_labels == -1).astype(int)

        self.models["isolation_forest"] = iso_forest
        self.scalers["isolation_forest"] = scaler
        self.anomaly_scores["isolation_forest"] = anomaly_scores

        return {
            "method": "Isolation Forest",
            "anomaly_labels": anomaly_labels,
            "anomaly_binary": anomaly_binary,
            "anomaly_scores": anomaly_scores,
            "anomaly_rate": float(np.mean(anomaly_binary)),
            "threshold": 0,
        }

    def _lof_detection(
        self, X: np.ndarray, metadata: pd.DataFrame, healthy_mask: np.ndarray
    ) -> Dict[str, Any]:
        scaler = StandardScaler()
        scaler.fit(X[healthy_mask])
        X_scaled = scaler.transform(X)

        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True,
        )
        lof.fit(X_scaled[healthy_mask])
        anomaly_labels = lof.predict(X_scaled)
        anomaly_scores = -lof.decision_function(X_scaled)

        anomaly_binary = (anomaly_labels == -1).astype(int)

        self.models["lof"] = lof
        self.scalers["lof"] = scaler
        self.anomaly_scores["lof"] = anomaly_scores

        return {
            "method": "Local Outlier Factor",
            "anomaly_labels": anomaly_labels,
            "anomaly_binary": anomaly_binary,
            "anomaly_scores": anomaly_scores,
            "anomaly_rate": float(np.mean(anomaly_binary)),
            "threshold": 0,
        }

    def _one_class_svm_detection(
        self, X: np.ndarray, metadata: pd.DataFrame, healthy_mask: np.ndarray
    ) -> Dict[str, Any]:
        scaler = StandardScaler()
        scaler.fit(X[healthy_mask])
        X_scaled = scaler.transform(X)

        svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
        svm.fit(X_scaled[healthy_mask])
        anomaly_labels = svm.predict(X_scaled)
        anomaly_scores = -svm.decision_function(X_scaled)

        anomaly_binary = (anomaly_labels == -1).astype(int)

        self.models["one_class_svm"] = svm
        self.scalers["one_class_svm"] = scaler
        self.anomaly_scores["one_class_svm"] = anomaly_scores

        return {
            "method": "One-Class SVM",
            "anomaly_labels": anomaly_labels,
            "anomaly_binary": anomaly_binary,
            "anomaly_scores": anomaly_scores,
            "anomaly_rate": float(np.mean(anomaly_binary)),
            "threshold": 0,
        }

    # ------------------------------------------------------------------
    # Ensemble
    # ------------------------------------------------------------------

    def _ensemble_detection(
        self, results: Dict[str, Any], metadata: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        FIX: normalise each method's scores to [0,1] independently before
        averaging, so a method with a larger raw score range cannot dominate.
        """
        normalised = []
        for method_name, method_result in results.items():
            if method_name == "ensemble":
                continue
            normalised.append(_normalise(method_result["anomaly_scores"]))

        ensemble_scores = np.mean(normalised, axis=0)

        threshold = np.percentile(ensemble_scores, 90)
        ensemble_binary = (ensemble_scores >= threshold).astype(int)

        return {
            "method": "Ensemble",
            "anomaly_binary": ensemble_binary,
            "anomaly_scores": ensemble_scores,
            "anomaly_rate": float(np.mean(ensemble_binary)),
            "threshold": float(threshold),
        }

    # ------------------------------------------------------------------
    # Analysis & visualisation
    # ------------------------------------------------------------------

    def _analyze_by_cohort(
        self, results: Dict[str, Any], metadata: pd.DataFrame
    ) -> Dict[str, Any]:
        cohort_analysis: Dict[str, Any] = {}

        for method_name, method_result in results.items():
            method_analysis: Dict[str, Any] = {}

            for cohort in metadata["cohort"].unique():
                cohort_mask = (metadata["cohort"] == cohort).values
                cohort_anomalies = method_result["anomaly_binary"][cohort_mask]

                method_analysis[cohort] = {
                    "anomaly_rate":  float(np.mean(cohort_anomalies)),
                    "total_trials":  int(len(cohort_anomalies)),
                    "anomaly_count": int(np.sum(cohort_anomalies)),
                }

            cohort_analysis[method_name] = method_analysis

        return cohort_analysis

    def _visualize_results(
        self,
        results: Dict[str, Any],
        cohort_analysis: Dict[str, Any],
        X: np.ndarray,
        metadata: pd.DataFrame,
    ):
        # 1. Anomaly score distributions by cohort
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for idx, (method_name, method_result) in enumerate(results.items()):
            if idx >= 4:
                break
            ax = axes[idx // 2, idx % 2]

            for cohort in ["Healthy", "PD", "CVA", "ACL"]:
                cohort_mask = (metadata["cohort"] == cohort).values
                if np.any(cohort_mask):
                    scores = method_result["anomaly_scores"][cohort_mask]
                    ax.hist(scores, alpha=0.6, label=cohort, bins=30)

            ax.set_title(f"{method_result['method']} - Anomaly Scores")
            ax.set_xlabel("Anomaly Score")
            ax.set_ylabel("Frequency")
            ax.legend()

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "anomaly_score_distributions.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        # 2. Anomaly rates by cohort (reuse already-computed cohort_analysis)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for idx, (method_name, method_analysis) in enumerate(cohort_analysis.items()):
            if idx >= 4:
                break
            ax = axes[idx // 2, idx % 2]

            cohorts = list(method_analysis.keys())
            rates   = [method_analysis[c]["anomaly_rate"] for c in cohorts]

            bars = ax.bar(cohorts, rates)
            ax.set_title(f"{method_name} - Anomaly Rate by Cohort")
            ax.set_ylabel("Anomaly Rate")
            ax.set_ylim(0, 1)

            for bar, rate in zip(bars, rates):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{rate:.3f}",
                    ha="center", va="bottom",
                )

            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "anomaly_rates_by_cohort.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        # 3. PCA visualisation
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        for idx, (method_name, method_result) in enumerate(results.items()):
            if idx >= 4:
                break
            ax = axes[idx // 2, idx % 2]

            normal_mask  = method_result["anomaly_binary"] == 0
            anomaly_mask = method_result["anomaly_binary"] == 1

            ax.scatter(X_pca[normal_mask,  0], X_pca[normal_mask,  1],
                       c="blue",  alpha=0.6, label="Normal",  s=20)
            ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1],
                       c="red",   alpha=0.8, label="Anomaly", s=30)

            ax.set_title(f"{method_result['method']} - PCA Visualization")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            ax.legend()

        plt.tight_layout()
        plt.savefig(
            self.results_dir / "pca_anomaly_visualization.png",
            dpi=300, bbox_inches="tight",
        )
        plt.close()

        logger.info("Visualizations saved to anomaly detection results directory")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results: Dict[str, Any],
        cohort_analysis: Dict[str, Any],
        metadata: pd.DataFrame,
    ):
        for method_name, method_result in results.items():
            results_df = metadata.copy()
            results_df[f"{method_name}_anomaly_score"] = method_result["anomaly_scores"]
            results_df[f"{method_name}_is_anomaly"]    = method_result["anomaly_binary"]
            results_df.to_csv(
                self.results_dir / f"{method_name}_results.csv", index=False
            )

        with open(self.results_dir / "cohort_analysis.json", "w") as f:
            json.dump(cohort_analysis, f, indent=2)

        for method_name, model in self.models.items():
            with open(self.results_dir / f"{method_name}_model.pkl", "wb") as f:
                pickle.dump(model, f)

        for method_name, scaler in self.scalers.items():
            with open(self.results_dir / f"{method_name}_scaler.pkl", "wb") as f:
                pickle.dump(scaler, f)

        logger.info("Results saved to anomaly detection directory")

    def _generate_summary(
        self, results: Dict[str, Any], cohort_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total_trials": int(len(next(iter(results.values()))["anomaly_binary"])),
            "methods": list(results.keys()),
            "method_performance": {},
        }

        for method_name, method_result in results.items():
            summary["method_performance"][method_name] = {
                "anomaly_rate":  float(method_result["anomaly_rate"]),
                "anomaly_count": int(np.sum(method_result["anomaly_binary"])),
            }

        ensemble_analysis = cohort_analysis.get("ensemble", {})
        if ensemble_analysis:
            sorted_cohorts = sorted(
                ensemble_analysis.items(),
                key=lambda x: x[1]["anomaly_rate"],
                reverse=True,
            )
            summary["highest_anomaly_cohorts"] = [
                {"cohort": cohort, "rate": analysis["anomaly_rate"]}
                for cohort, analysis in sorted_cohorts[:5]
            ]

        return summary


def detect_anomalies(config: dict) -> Dict[str, Any]:
    detector = GaitAnomalyDetector(config)
    return detector.run()