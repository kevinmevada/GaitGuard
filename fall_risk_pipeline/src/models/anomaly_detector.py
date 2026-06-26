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

from src.models.anomaly_feature_schema import save_trial_feature_schema
from src.models.anomaly_scoring import ANOMALY_METHODS, normalise_scores
from src.evaluation.anomaly_loso_evaluator import run_anomaly_loso_evaluation
from src.evaluation.primary_endpoint import (
    ENDPOINT_ANOMALY_ENSEMBLE,
    ENDPOINT_BILSTM_AE_ENSEMBLE,
    resolve_primary_endpoint,
)
from src.dataset.train_fit_mask import healthy_train_fit_mask
from src.preprocessing.fold_normalization import reconstruction_threshold_train_only
from src.utils.checkpoint_io import save_checkpoint
from src.utils.reproducibility import get_pipeline_seed

console = Console()

INSAMPLE_ARTIFACT_BANNER = (
    "# IN-SAMPLE: not suitable for reporting. "
    "Use results/metrics/anomaly_metrics.csv (LOSO OOF).\n"
)
INSAMPLE_JSON_DISCLAIMER = {
    "_disclaimer": (
        "IN-SAMPLE exploratory output from GaitAnomalyDetector.run(). "
        "Do not cite AUC/sensitivity/specificity from this file. "
        "Primary metrics: results/metrics/anomaly_metrics.csv."
    ),
    "_fit_policy": (
        "Scaler, one-class models, score normalisation, and ensemble threshold "
        "are fit on subject-split Healthy TRAIN trials only (v13 fix)."
    ),
}


def _normalise(scores: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Min-max to [0, 1] using *reference* scores (train-fold Healthy by default)."""
    return normalise_scores(scores, reference)


class GaitAnomalyDetector:
    """
    Unsupervised anomaly detection for gait patterns.
    Identifies unusual walking patterns that deviate from normal gait.
    """

    def __init__(self, config: dict):
        self.config = config
        self.random_state = get_pipeline_seed(config)
        self.feat_dir = Path(config["paths"]["features"])
        self.results_dir = Path(config["paths"]["results"]) / "anomaly_detection"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.anomaly_scores: Dict[str, np.ndarray] = {}
        self.trial_feature_columns: List[str] = []
        self._healthy_mask_size: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        logger.info(
            "Starting anomaly detection (exploratory in-sample bulk run + LOSO evaluation)..."
        )
        logger.warning(
            "Bulk anomaly outputs are IN-SAMPLE (Healthy train subjects rescored). "
            "Report only anomaly_metrics.csv from LOSO evaluation for manuscript metrics."
        )

        X, metadata, feature_cols = self._load_data()
        self.trial_feature_columns = feature_cols

        # v13: fit scaler / models / thresholds on subject-split Healthy TRAIN only.
        fit_mask = healthy_train_fit_mask(metadata, self.config)
        if fit_mask.sum() < 5:
            logger.warning(
                "Fewer than 5 Healthy TRAIN-fold samples — falling back to all Healthy. "
                "Re-run after subject_split manifest is generated."
            )
            fit_mask = (metadata["cohort"] == "Healthy").values
        self._fit_mask = fit_mask
        self._healthy_mask_size = int(fit_mask.sum())

        results: Dict[str, Any] = {}
        loso_metrics = None
        daphnet_fog_result = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold yellow]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} tasks"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Running anomaly detection...", total=8)

            progress.update(task_id, description="Anomaly method: isolation_forest")
            results["isolation_forest"] = self._isolation_forest_detection(X, metadata, fit_mask)
            progress.advance(task_id)

            progress.update(task_id, description="Anomaly method: lof")
            results["lof"] = self._lof_detection(X, metadata, fit_mask)
            progress.advance(task_id)

            progress.update(task_id, description="Anomaly method: one_class_svm")
            results["one_class_svm"] = self._one_class_svm_detection(X, metadata, fit_mask)
            progress.advance(task_id)

            progress.update(task_id, description="Combining anomaly ensemble")
            results["ensemble"] = self._ensemble_detection(results, metadata, fit_mask)
            progress.advance(task_id)

            progress.update(task_id, description="Analyzing cohorts and visualizing")
            cohort_analysis = self._analyze_by_cohort(results, metadata)
            self._visualize_results(results, cohort_analysis, X, metadata)
            progress.advance(task_id)

            progress.update(task_id, description="Saving anomaly outputs")
            self._save_results(results, cohort_analysis, metadata)
            progress.advance(task_id)

            progress.update(task_id, description="LOSO anomaly evaluation (ANOM-001)")
            primary_ep = resolve_primary_endpoint(self.config)
            if self.config.get("anomaly", {}).get("loso_evaluation", True):
                if primary_ep == ENDPOINT_BILSTM_AE_ENSEMBLE:
                    from src.evaluation.bilstm_ae_loso_evaluator import run_bilstm_ae_loso_evaluation

                    loso_metrics = run_bilstm_ae_loso_evaluation(self.config)
                else:
                    loso_metrics = run_anomaly_loso_evaluation(self.config)
            progress.advance(task_id)

            progress.update(task_id, description="DAPHNET sealed FOG eval (zero-shot)")
            try:
                if primary_ep == ENDPOINT_BILSTM_AE_ENSEMBLE:
                    from src.evaluation.daphnet_bilstm_ae_evaluator import run_daphnet_bilstm_ae_fog_eval

                    daphnet_fog_result = run_daphnet_bilstm_ae_fog_eval(self.config)
                else:
                    from src.evaluation.daphnet_fog_evaluator import run_daphnet_sealed_fog_eval

                    daphnet_fog_result = run_daphnet_sealed_fog_eval(self.config)
            except FileNotFoundError as exc:
                logger.info("Skipping DAPHNET sealed FOG eval: {}", exc)
            except Exception as exc:
                from src.evaluation.daphnet_fog_evaluator import DaphnetSealedEvalError
                from src.evaluation.daphnet_bilstm_ae_evaluator import DaphnetBilstmAeEvalError

                if isinstance(exc, (DaphnetSealedEvalError, DaphnetBilstmAeEvalError)):
                    logger.warning("DAPHNET sealed FOG eval skipped: {}", exc)
                else:
                    raise
            progress.advance(task_id)

            progress.update(task_id, description="Anomaly detection complete")

        logger.info("Anomaly detection completed!")
        summary = self._generate_summary(results, cohort_analysis)
        if loso_metrics is not None and not loso_metrics.empty:
            ens_method = (
                "bilstm_ae_ensemble"
                if resolve_primary_endpoint(self.config) == ENDPOINT_BILSTM_AE_ENSEMBLE
                else "ensemble"
            )
            ens = loso_metrics[loso_metrics["method"] == ens_method]
            if not ens.empty:
                summary["loso_ensemble_auc"] = float(ens.iloc[0]["auc"])
        if daphnet_fog_result:
            summary["daphnet_fog_auroc"] = daphnet_fog_result.get("auc")
        return {
            "detection_results": results,
            "cohort_analysis": cohort_analysis,
            "summary": summary,
            "loso_metrics": loso_metrics,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
        df = pd.read_parquet(self.feat_dir / "trial_features.parquet")
        if "trial_id" in df.columns:
            df = df[~df["trial_id"].astype(str).str.startswith("daphnet_")].reset_index(drop=True)

        from src.ingestion.daphnet_label_mapping import assert_labels_not_in_feature_columns

        meta_cols = [
            "trial_id", "participant_id", "cohort", "risk_label",
            "multiclass_label", "fall_probability", "laterality_biased",
        ]

        feature_cols = [col for col in df.columns if col not in meta_cols]
        feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        assert_labels_not_in_feature_columns(feature_cols, context="anomaly trial features")

        X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        X = X.values
        metadata = df[[c for c in meta_cols if c in df.columns]].copy()

        logger.info(f"Loaded {X.shape[0]} trials with {X.shape[1]} features")
        return X, metadata, feature_cols

    # ------------------------------------------------------------------
    # Detection methods
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_healthy_scaler(
        X: np.ndarray, fit_mask: np.ndarray
    ) -> tuple[StandardScaler, np.ndarray]:
        """Fit StandardScaler on train-fold Healthy rows; transform all rows."""
        scaler = StandardScaler()
        scaler.fit(X[fit_mask])
        return scaler, scaler.transform(X)

    def _isolation_forest_detection(
        self, X: np.ndarray, metadata: pd.DataFrame, fit_mask: np.ndarray
    ) -> Dict[str, Any]:
        scaler, X_scaled = self._fit_healthy_scaler(X, fit_mask)

        iso_forest = IsolationForest(
            contamination=0.1,
            random_state=self.random_state,
            n_estimators=100,
        )
        iso_forest.fit(X_scaled[fit_mask])          # train on normals only
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
        self, X: np.ndarray, metadata: pd.DataFrame, fit_mask: np.ndarray
    ) -> Dict[str, Any]:
        scaler, X_scaled = self._fit_healthy_scaler(X, fit_mask)

        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1,
            novelty=True,
        )
        lof.fit(X_scaled[fit_mask])
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
        self, X: np.ndarray, metadata: pd.DataFrame, fit_mask: np.ndarray
    ) -> Dict[str, Any]:
        scaler, X_scaled = self._fit_healthy_scaler(X, fit_mask)

        svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
        svm.fit(X_scaled[fit_mask])
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
        self,
        results: Dict[str, Any],
        metadata: pd.DataFrame,
        fit_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Normalise each method using train-fold Healthy scores only, then average.

        v13 fix: ensemble threshold (p90) computed on train-fold reconstruction /
        anomaly scores — never on held-out or pathological trials.
        """
        normalised = []
        for method_name in ANOMALY_METHODS:
            if method_name not in results:
                continue
            scores = np.asarray(results[method_name]["anomaly_scores"], dtype=float)
            normalised.append(_normalise(scores, scores[fit_mask]))

        ensemble_scores = np.mean(normalised, axis=0)
        train_ref = ensemble_scores[fit_mask]
        threshold = reconstruction_threshold_train_only(train_ref, percentile=90.0)
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

            for cohort in sorted(metadata["cohort"].unique()):
                cohort_mask = (metadata["cohort"] == cohort).values
                if np.any(cohort_mask):
                    scores = method_result["anomaly_scores"][cohort_mask]
                    ax.hist(scores, alpha=0.5, label=cohort, bins=30)

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

        # 3. PCA visualisation (fit on healthy reference, transform all)
        healthy_mask = (metadata["cohort"] == "Healthy").values
        pca = PCA(n_components=2)
        if healthy_mask.sum() >= 2:
            pca.fit(X[healthy_mask])
        else:
            pca.fit(X)
        X_pca = pca.transform(X)

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
            out_path = (
                self.results_dir / f"anomaly_exploratory_insample_{method_name}_results.csv"
            )
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(INSAMPLE_ARTIFACT_BANNER)
                results_df.to_csv(fh, index=False)

        cohort_payload = {**INSAMPLE_JSON_DISCLAIMER, "by_method": cohort_analysis}
        with open(
            self.results_dir / "anomaly_exploratory_insample_cohort_analysis.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cohort_payload, f, indent=2)

        for method_name, model in self.models.items():
            save_checkpoint(
                self.results_dir / f"{method_name}_model.pkl",
                model,
                manifest_dir=self.results_dir,
            )

        for method_name, scaler in self.scalers.items():
            save_checkpoint(
                self.results_dir / f"{method_name}_scaler.pkl",
                scaler,
                manifest_dir=self.results_dir,
            )

        if self.trial_feature_columns:
            schema_path = save_trial_feature_schema(
                self.results_dir,
                self.trial_feature_columns,
                healthy_n_samples=self._healthy_mask_size,
            )
            logger.info(
                f"Anomaly trial feature schema saved ({len(self.trial_feature_columns)} columns) "
                f"→ {schema_path}"
            )

        self._save_deploy_calibration(results)

        logger.info("Results saved to anomaly detection directory")

    def _save_deploy_calibration(self, results: Dict[str, Any]) -> None:
        """Persist deploy-time score ranges fit on train-fold Healthy trials only."""
        fit_mask = getattr(self, "_fit_mask", None)
        methods_calib: Dict[str, Dict[str, float]] = {}
        for method_name in ANOMALY_METHODS:
            if method_name not in results:
                continue
            scores = np.asarray(results[method_name]["anomaly_scores"], dtype=float)
            ref = scores[fit_mask] if fit_mask is not None else scores
            methods_calib[method_name] = {
                "min": float(np.nanmin(ref)),
                "max": float(np.nanmax(ref)),
            }
        payload = {
            "methods": methods_calib,
            "ensemble_threshold_p90": float(results["ensemble"]["threshold"]),
            "threshold_fit_policy": "healthy_train_fold_only",
            "primary_endpoint": resolve_primary_endpoint(self.config),
        }
        path = self.results_dir / "deploy_calibration.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        metrics_dir = Path(self.config["paths"]["metrics"])
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "deploy_calibration.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def _generate_summary(
        self, results: Dict[str, Any], cohort_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "total_trials": int(len(next(iter(results.values()))["anomaly_binary"])),
            "methods": list(results.keys()),
            "method_performance": {},
            "reporting_note": (
                "Bulk run metrics are in-sample only. "
                "Use loso_metrics / anomaly_metrics.csv for publication."
            ),
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