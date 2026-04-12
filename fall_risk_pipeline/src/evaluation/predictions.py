"""
Generate inference-style prediction exports without in-sample validation claims.

FIX: fall_probability and laterality_biased excluded from feature columns,
     matching the same NON_FEATURE_COLS guard used in trainer and evaluator.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.utils.validation import check_is_fitted

from src.models.trainer import NON_FEATURE_COLS


class PredictionGenerator:

    def __init__(self, config: dict):
        self.config = config
        self.feat_dir = Path(config["paths"]["features"])
        self.ckpt_dir = Path(config["paths"]["checkpoints"])
        self.results_dir = Path(config["paths"]["metrics"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        feat_path = self.feat_dir / "patient_features.parquet"
        if not feat_path.exists():
            raise FileNotFoundError(f"{feat_path} not found")

        df = pd.read_parquet(feat_path)

        # FIX: use the same exclusion set as trainer / evaluator so the feature
        # vector fed to the model at inference matches what it was trained on.
        feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        feat_cols = df[feat_cols].select_dtypes(include=np.number).columns.tolist()

        X = df[feat_cols].values.astype(np.float32)
        participant_ids = df["participant_id"].values
        cohorts = df["cohort"].values

        model = self._load_best_model()
        if model is None:
            logger.error("No model found")
            return

        y_prob = model.predict_proba(X)[:, 1]
        y_percent = y_prob * 100

        self._save_predictions(participant_ids, cohorts, y_percent, y_prob)
        self._save_summary(df, y_prob)

    def _load_best_model(self):
        # Prefer the evaluation metrics file (written by Evaluator).
        metrics_path = self.results_dir / "metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            if not df.empty:
                best_model_name = df.iloc[0]["model"]
                model = self._load_fitted_model(self.ckpt_dir / f"{best_model_name}.pkl")
                if model is not None:
                    return model

        # Fall back to trainer comparison file if evaluator hasn't run yet.
        comparison_path = self.results_dir / "model_comparison_cv.csv"
        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            if not df.empty:
                best_model_name = df.iloc[0]["model"]
                model = self._load_fitted_model(self.ckpt_dir / f"{best_model_name}.pkl")
                if model is not None:
                    logger.info(f"Loaded best model from trainer comparison: {best_model_name}")
                    return model

        # Last resort: try well-known names in order of typical performance.
        for model_name in ["ensemble", "lightgbm", "xgboost", "random_forest", "svm"]:
            model = self._load_fitted_model(self.ckpt_dir / f"{model_name}.pkl")
            if model is not None:
                logger.warning(f"Using fallback checkpoint: {model_name}")
                return model

        return None

    def _load_fitted_model(self, model_path: Path):
        if not model_path.exists():
            return None

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        try:
            check_is_fitted(model)
        except Exception as exc:
            logger.warning(f"Skipping unusable checkpoint {model_path.name}: {exc}")
            return None

        return model

    def _save_predictions(self, ids, cohorts, y_percent, y_prob):
        results = [
            {
                "participant_id": pid,
                "cohort": cohort,
                "risk_probability_percent": float(prob_pct),
                "risk_category": self._get_risk_category(float(prob_pct)),
                "model_confidence": self._confidence_label(float(prob)),
                "clinical_recommendation": self._get_recommendation(float(prob_pct)),
            }
            for pid, cohort, prob_pct, prob in zip(ids, cohorts, y_percent, y_prob)
        ]

        df = pd.DataFrame(results)
        path = self.results_dir / "predictions.csv"
        df.to_csv(path, index=False)
        logger.info(f"Saved predictions -> {path}")

    def _save_summary(self, df: pd.DataFrame, y_prob: np.ndarray):
        cohort_counts = df["cohort"].value_counts(dropna=False).to_dict() if "cohort" in df else {}
        summary = {
            "summary_type": "inference_export",
            "publication_note": (
                "Predictions are generated from the selected checkpoint and are not "
                "validation metrics."
            ),
            "n_samples": int(len(y_prob)),
            "n_participants": int(df["participant_id"].nunique()) if "participant_id" in df else int(len(y_prob)),
            "cohort_counts": {str(k): int(v) for k, v in cohort_counts.items()},
            "risk_distribution": {
                "low":       int(np.sum(y_prob < 0.2)),
                "moderate":  int(np.sum((y_prob >= 0.2) & (y_prob < 0.5))),
                "high":      int(np.sum((y_prob >= 0.5) & (y_prob < 0.8))),
                "very_high": int(np.sum(y_prob >= 0.8)),
            },
        }

        path = self.results_dir / "prediction_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved summary -> {path}")

    def _get_risk_category(self, p):
        if p < 20:  return "Low"
        if p < 40:  return "Moderate"
        if p < 60:  return "High"
        return "Very High"

    def _confidence_label(self, prob):
        if prob > 0.8 or prob < 0.2:  return "High"
        if prob > 0.6 or prob < 0.4:  return "Medium"
        return "Low"

    def _get_recommendation(self, p):
        if p < 20:  return "Routine monitoring"
        if p < 40:  return "Preventive exercises"
        if p < 60:  return "Clinical evaluation recommended"
        return "Immediate intervention required"