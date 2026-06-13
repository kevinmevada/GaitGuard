"""
Generate inference-style prediction exports without in-sample validation claims.

FIX: fall_probability and laterality_biased excluded from feature columns,
     matching the same NON_FEATURE_COLS guard used in trainer and evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.utils.validation import check_is_fitted

from src.evaluation.clinical_threshold import (
    ARTIFACT_FILENAME,
    assign_risk_level,
    default_clinical_threshold,
    elevated_risk_probability,
    load_clinical_threshold_artifact,
)
from src.evaluation.research_disclaimers import (
    RESEARCH_PROTOTYPE_DISCLAIMER,
    limitations_payload,
    screening_note,
)
from src.evaluation.primary_endpoint import resolve_inference_checkpoint_name
from src.features.feature_matrix import load_patient_feature_matrix
from src.utils.checkpoint_io import CheckpointIntegrityError, load_checkpoint


class PredictionGenerator:

    def __init__(self, config: dict):
        self.config = config
        self.feat_dir = Path(config["paths"]["features"])
        self.ckpt_dir = Path(config["paths"]["checkpoints"])
        self.results_dir = Path(config["paths"]["metrics"])
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        X, _, _, _, df = load_patient_feature_matrix(self.config)
        participant_ids = df["participant_id"].values
        cohorts = df["cohort"].values

        model = self._load_best_model()
        if model is None:
            logger.error("No model found")
            return

        proba = model.predict_proba(X)
        y_prob = elevated_risk_probability(proba, self.config)
        y_percent = y_prob * 100

        self._save_predictions(participant_ids, cohorts, y_percent, y_prob)
        self._save_summary(df, y_prob)

    def _load_best_model(self):
        deploy_name = resolve_inference_checkpoint_name(self.config, self.results_dir)
        model = self._load_fitted_model(self.ckpt_dir / f"{deploy_name}.pkl")
        if model is not None:
            logger.info(f"Loaded primary deploy checkpoint: {deploy_name}")
            return model

        metrics_path = self.results_dir / "metrics.csv"
        if metrics_path.exists():
            df = pd.read_csv(metrics_path)
            if not df.empty:
                best_model_name = df.iloc[0]["model"]
                model = self._load_fitted_model(self.ckpt_dir / f"{best_model_name}.pkl")
                if model is not None:
                    return model

        comparison_path = self.results_dir / "model_comparison_cv.csv"
        if comparison_path.exists():
            df = pd.read_csv(comparison_path)
            if not df.empty:
                best_model_name = df.iloc[0]["model"]
                model = self._load_fitted_model(self.ckpt_dir / f"{best_model_name}.pkl")
                if model is not None:
                    logger.info(f"Loaded best model from trainer comparison: {best_model_name}")
                    return model

        for model_name in [
            "ensemble",
            "ensemble_soft_voting",
            "ensemble_stacking",
            "lightgbm",
            "xgboost",
            "random_forest",
            "svm",
        ]:
            model = self._load_fitted_model(self.ckpt_dir / f"{model_name}.pkl")
            if model is not None:
                logger.warning(f"Using fallback checkpoint: {model_name}")
                return model

        return None

    def _load_fitted_model(self, model_path: Path):
        if not model_path.exists():
            return None

        try:
            model = load_checkpoint(
                model_path,
                manifest_dir=self.ckpt_dir,
                require_manifest=True,
            )
        except CheckpointIntegrityError as exc:
            logger.warning(
                f"Checkpoint verification failed for {model_path.name}: {exc}"
            )
            return None
        except Exception as exc:
            logger.warning(f"Failed to load checkpoint {model_path.name}: {exc}")
            return None

        try:
            check_is_fitted(model)
        except Exception as exc:
            logger.warning(f"Skipping unusable checkpoint {model_path.name}: {exc}")
            return None

        return model

    def _clinical_cutoff(self) -> dict:
        path = self.results_dir / ARTIFACT_FILENAME
        payload = load_clinical_threshold_artifact(path)
        return payload if payload else default_clinical_threshold()

    def _save_predictions(self, ids, cohorts, y_percent, y_prob):
        cutoff = self._clinical_cutoff().get("primary_cutoff", {})
        youden = float(cutoff.get("probability", 0.5))
        note = screening_note
        results = [
            {
                "participant_id": pid,
                "cohort": cohort,
                "risk_probability_percent": float(prob_pct),
                "risk_category": assign_risk_level(float(prob), youden),
                "above_youden_cutoff": float(prob) >= youden,
                "youden_cutoff_probability": youden,
                "model_confidence": self._confidence_label(float(prob)),
                "screening_note": note(float(prob), youden),
                "disclaimer": RESEARCH_PROTOTYPE_DISCLAIMER,
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
                "validation metrics. "
                + RESEARCH_PROTOTYPE_DISCLAIMER
            ),
            "n_samples": int(len(y_prob)),
            "n_participants": int(df["participant_id"].nunique()) if "participant_id" in df else int(len(y_prob)),
            "cohort_counts": {str(k): int(v) for k, v in cohort_counts.items()},
            "risk_distribution": self._risk_distribution(y_prob),
            "clinical_cutoff_youden": self._clinical_cutoff().get("primary_cutoff", {}),
            "limitations": limitations_payload(),
        }

        path = self.results_dir / "prediction_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        limits_path = self.results_dir / "limitations.md"
        limits_path.write_text(self._limitations_markdown(), encoding="utf-8")

        logger.info(f"Saved summary -> {path}")

    def _limitations_markdown(self) -> str:
        lim = limitations_payload()
        lines = [
            "# Prediction export — limitations",
            "",
            f"**{lim['disclaimer']}**",
            "",
        ]
        for item in lim["points"]:
            lines.append(f"- {item}")
        lines.append("")
        return "\n".join(lines)

    def _risk_distribution(self, y_prob: np.ndarray) -> dict[str, int]:
        cutoff = self._clinical_cutoff().get("primary_cutoff", {})
        youden = float(cutoff.get("probability", 0.5))
        low_edge = 0.5 * youden
        return {
            "low": int(np.sum(y_prob < low_edge)),
            "moderate_borderline": int(np.sum((y_prob >= low_edge) & (y_prob < youden))),
            "high_at_or_above_youden": int(np.sum(y_prob >= youden)),
        }

    def _confidence_label(self, prob):
        if prob > 0.8 or prob < 0.2:
            return "High (model score dispersion)"
        if prob > 0.6 or prob < 0.4:
            return "Medium (model score dispersion)"
        return "Low (model score dispersion)"
