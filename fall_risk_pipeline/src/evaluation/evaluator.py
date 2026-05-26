"""
Publication-oriented evaluation using nested subject-grouped validation.

FIXES APPLIED:
  1. Removed fall_probability / laterality_biased / n_trials from feature matrix.
  2. Fast mode now re-fits a checkpoint on train data per fold — no longer evaluates
     the full-data checkpoint in-sample (which inflated metrics).
  3. Replaced non-existent _tune_model_with_budget() call with _run_optuna() +
     _build_pipeline_from_params() — the correct public trainer API.
  4. SHAP aggregated across all LOSO held-out folds (mean |SHAP|), not a single patient.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from src.dataset.label_policy import is_binary_task, label_mode_description
from src.evaluation.auc_significance import pairwise_auc_significance
from src.evaluation.clinical_threshold import (
    build_clinical_threshold_artifact,
    save_clinical_threshold_artifact,
    youden_threshold,
)
from src.evaluation.classification_significance import pairwise_classification_significance
from src.evaluation.multiclass_metrics import build_multiclass_metric_payload
from src.features.feature_matrix import load_patient_feature_matrix
from src.models.ensemble_builder import (
    build_ensemble_estimator,
    ensemble_model_name,
    predict_ensemble_oof_proba,
    resolve_ensemble_methods,
)
from src.models.trainer import ModelTrainer


class Evaluator:
    """
    Nested grouped evaluation, calibration plots, SHAP, and metrics export.

    Parameters
    ----------
    config : dict
        Pipeline configuration dictionary.
    fast : bool
        If True (default), loads saved checkpoints and re-fits them per fold
        so held-out scores are unbiased.  Runs in a few minutes.
        Set to False for full per-fold Optuna retuning (publication mode).
    """

    def __init__(self, config: dict, fast: bool = False):
        self.config = config
        self.fast = fast

        self.feat_dir    = Path(config["paths"]["features"])
        self.ckpt_dir    = Path(config["paths"]["checkpoints"])
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.fig_models  = Path(config["paths"]["figures_models"])
        self.fig_shap    = Path(config["paths"]["figures_shap"])
        self.fmt = config["reporting"]["figure_format"]
        self.dpi = config["reporting"]["figure_dpi"]

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.fig_models.mkdir(parents=True, exist_ok=True)
        self.fig_shap.mkdir(parents=True, exist_ok=True)

        self.trainer = ModelTrainer(config)
        eval_cfg = config.get("models", {}).get("evaluation", {})
        self.validation_strategy = eval_cfg.get("strategy", "nested_group_cv")

        self.nested_trials  = int(eval_cfg.get("nested_n_trials",  min(3, self.trainer.n_trials)))
        self.nested_timeout = int(eval_cfg.get("nested_timeout_per_model", min(60, self.trainer.timeout)))

        mode = "FAST (per-fold refit)" if self.fast else "FULL nested CV"
        logger.info(f"Evaluator initialized — mode: {mode}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        X, y, groups, feat_names, cohorts = self._load_data()
        binary_task = is_binary_task(y, self.config)
        logger.info(label_mode_description(self.config))

        model_names = [
            name for name in self.config["models"]["run"]
            if name not in ("cnn_1d", "lstm")
        ]
        all_results: dict[str, dict] = {}
        cohort_rows: list[dict] = []

        for name in tqdm(
            model_names,
            desc="Evaluating models",
            colour="red",
            bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
        ):
            if self.fast:
                result = self._fast_evaluate_model(name, X, y, groups, cohorts)
            else:
                result = self._nested_group_evaluate_model(name, X, y, groups, cohorts)

            all_results[name] = result
            cohort_rows.extend(self._cohort_metric_rows(name, result))
            self._log_model_metrics(name, result, binary_task)

        ensemble_methods = resolve_ensemble_methods(self.config)
        if ensemble_methods:
            if self.fast:
                ensemble_results = self._fast_evaluate_ensembles(
                    X, y, groups, cohorts, model_names, ensemble_methods
                )
            else:
                ensemble_results = self._nested_group_evaluate_ensembles(
                    X, y, groups, cohorts, ensemble_methods
                )
            for ens_name, ensemble_result in ensemble_results.items():
                all_results[ens_name] = ensemble_result
                cohort_rows.extend(self._cohort_metric_rows(ens_name, ensemble_result))
                self._log_model_metrics(ens_name, ensemble_result, binary_task)
            self._save_ensemble_comparison(ensemble_results)

        if binary_task:
            self._plot_roc_all(all_results)
            self._plot_pr_curves(all_results)
            self._plot_calibration(all_results)
        else:
            self._plot_multiclass_roc(all_results)
            self._plot_multiclass_calibration(all_results)
            self._save_per_class_auc_table(all_results)
        self._plot_confusion_matrices(all_results)

        best_name = max(all_results, key=lambda n: all_results[n]["auc"])
        logger.info(f"Best model (AUC): {best_name}")

        pairwise_df, vs_ref_df = pd.DataFrame(), pd.DataFrame()
        mcnemar_pairwise_df, mcnemar_vs_ref_df, fold_disc_df = (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
        )
        if binary_task:
            pairwise_df, vs_ref_df = self._run_auc_pairwise_tests(all_results, best_name)
            mcnemar_pairwise_df, mcnemar_vs_ref_df, fold_disc_df = self._run_mcnemar_tests(
                all_results, best_name
            )
        else:
            logger.info(
                "Skipping DeLong/McNemar/threshold/ROC-PR (binary-only); "
                "multiclass uses macro-OVR AUC and argmax predictions."
            )

        self._shap_analysis(best_name, X, y, groups, feat_names, cohorts)

        self._save_oof_predictions(all_results)
        self._export_clinical_threshold(all_results, best_name)
        self._save_all_metrics(
            all_results, len(np.unique(groups)), vs_ref_df, mcnemar_vs_ref_df
        )
        if binary_task:
            self._save_threshold_comparison(all_results)
        self._save_cohort_metrics(cohort_rows)
        if not pairwise_df.empty:
            pairwise_df.to_csv(self.metrics_dir / "auc_pairwise_pvalues.csv", index=False)
        if not mcnemar_pairwise_df.empty:
            mcnemar_pairwise_df.to_csv(
                self.metrics_dir / "mcnemar_pairwise_pvalues.csv", index=False
            )
        if not fold_disc_df.empty:
            fold_disc_df.to_csv(
                self.metrics_dir / "mcnemar_fold_discordant.csv", index=False
            )

        self._leakage_comparison(X, y, groups, all_results)

        return all_results

    # ------------------------------------------------------------------
    # FAST MODE — per-fold refit of saved checkpoint architecture
    # ------------------------------------------------------------------

    def _fast_evaluate_model(self, name: str, X, y, groups, cohorts):
        """
        Load a saved checkpoint to recover its hyperparameters, then re-fit it
        on each LOSO training fold and evaluate on the held-out subject.
        This avoids in-sample evaluation while skipping Optuna retuning.
        """
        checkpoint = self._load_checkpoint(name)
        if checkpoint is None:
            logger.warning(f"Checkpoint not found for {name} — falling back to nested CV")
            return self._nested_group_evaluate_model(name, X, y, groups, cohorts)

        unique_subjects = np.unique(groups)
        binary_task = is_binary_task(y, self.config)
        all_probs, all_true, all_cohorts, all_pids = [], [], [], []
        all_pred_ty, all_pred_05, fold_thresholds = [], [], []
        all_proba_blocks: list[np.ndarray] = []

        for subj in unique_subjects:
            test_idx  = np.where(groups == subj)[0]
            train_idx = np.where(groups != subj)[0]

            if len(np.unique(y[train_idx])) < 2:
                continue

            import sklearn.base as skbase
            fold_model = skbase.clone(checkpoint)
            fold_model.fit(X[train_idx], y[train_idx])

            test_prob, pred_ty, pred_05, thresh = self._loso_fold_classifications(
                fold_model, X, y, train_idx, test_idx
            )
            if binary_task:
                all_probs.extend(test_prob.tolist())
            else:
                all_proba_blocks.append(np.asarray(test_prob, dtype=float))
            all_true.extend(y[test_idx].tolist())
            all_cohorts.extend(cohorts[test_idx].tolist())
            all_pids.extend(groups[test_idx].tolist())
            all_pred_ty.extend(pred_ty.tolist())
            all_pred_05.extend(pred_05.tolist())
            fold_thresholds.append(thresh)

        logger.info(f"{name} — fast LOSO done ({len(unique_subjects)} subjects)")
        if not binary_task:
            all_probs = np.vstack(all_proba_blocks).tolist() if all_proba_blocks else []
        return self._finalize_oof_payload(
            name, all_true, all_probs, all_cohorts, all_pids,
            all_pred_ty, all_pred_05, fold_thresholds,
        )

    def _select_top_base_models(
        self, tuned: list[tuple[str, dict]], top_k: int
    ) -> list[tuple[str, dict]]:
        rank_path = self.metrics_dir / "model_comparison_cv.csv"
        if rank_path.exists():
            rank_df = pd.read_csv(rank_path)
            order = rank_df.sort_values("cv_auc", ascending=False)["model"].tolist()
            tuned = sorted(
                tuned,
                key=lambda item: order.index(item[0]) if item[0] in order else len(order),
            )
        else:
            tuned = sorted(
                tuned, key=lambda item: item[1].get("cv_auc", 0.0), reverse=True
            )
        return tuned[:top_k]

    def _ensemble_cv_folds(self) -> int:
        return int(
            self.config.get("models", {}).get("ensemble", {}).get("stacking", {}).get(
                "cv_folds",
                self.config["models"]["tuning"]["cv_folds"],
            )
        )

    def _fast_evaluate_ensembles(
        self,
        X,
        y,
        groups,
        cohorts,
        model_names: list[str],
        methods: list[str],
    ) -> dict[str, dict]:
        import sklearn.base as skbase

        checkpoints = {}
        for name in model_names:
            m = self._load_checkpoint(name)
            if m is not None:
                checkpoints[name] = m

        if not checkpoints:
            logger.warning("No checkpoints found — falling back to nested ensemble CV")
            return self._nested_group_evaluate_ensembles(
                X, y, groups, cohorts, methods
            )

        ens_cfg = self.config["models"]["ensemble"]
        top_k = ens_cfg["top_k"]
        cv_folds = self._ensemble_cv_folds()
        rs = self.config["models"]["evaluation"]["random_state"]

        accum = {
            ensemble_model_name(m): {
                "probs": [], "true": [], "cohorts": [], "pids": [],
                "pred_ty": [], "pred_05": [], "thresh": [],
            }
            for m in methods
        }

        binary_task = is_binary_task(y, self.config)
        unique_subjects = np.unique(groups)
        for subj in unique_subjects:
            test_idx = np.where(groups == subj)[0]
            train_idx = np.where(groups != subj)[0]
            if len(np.unique(y[train_idx])) < 2:
                continue

            tuned = []
            for name, ckpt in checkpoints.items():
                fold_model = skbase.clone(ckpt)
                fold_model.fit(X[train_idx], y[train_idx])
                tuned.append((name, {"pipeline": fold_model, "cv_auc": 0.0}))
            top_models = self._select_top_base_models(tuned, top_k)

            for method in methods:
                key = ensemble_model_name(method)
                if method == "stacking":
                    test_prob = predict_ensemble_oof_proba(
                        method,
                        top_models,
                        X[train_idx],
                        y[train_idx],
                        groups[train_idx],
                        X[test_idx],
                        cv_folds=cv_folds,
                        random_state=rs,
                    )
                    train_prob = predict_ensemble_oof_proba(
                        method,
                        top_models,
                        X[train_idx],
                        y[train_idx],
                        groups[train_idx],
                        X[train_idx],
                        cv_folds=cv_folds,
                        random_state=rs,
                    )
                else:
                    train_prob_list = [
                        res["pipeline"].predict_proba(X[train_idx])
                        for _, res in top_models
                    ]
                    test_prob_list = [
                        res["pipeline"].predict_proba(X[test_idx])
                        for _, res in top_models
                    ]
                    train_prob = np.mean(train_prob_list, axis=0)
                    test_prob = np.mean(test_prob_list, axis=0)

                acc = accum[key]
                if binary_task:
                    train_sc = train_prob[:, 1] if np.ndim(train_prob) > 1 else train_prob
                    test_sc = test_prob[:, 1] if np.ndim(test_prob) > 1 else test_prob
                    thresh = self._youden_threshold(y[train_idx], train_sc)
                    acc["probs"].extend(test_sc.tolist())
                    acc["pred_ty"].extend((test_sc >= thresh).astype(int).tolist())
                    acc["pred_05"].extend((test_sc >= 0.5).astype(int).tolist())
                    acc["thresh"].append(thresh)
                else:
                    pred = np.argmax(test_prob, axis=1).astype(int)
                    if "blocks" not in acc:
                        acc["blocks"] = []
                    acc["blocks"].append(np.asarray(test_prob, dtype=float))
                    acc["pred_ty"].extend(pred.tolist())
                    acc["pred_05"].extend(pred.tolist())
                    acc["thresh"].append(float("nan"))
                acc["true"].extend(y[test_idx].tolist())
                acc["cohorts"].extend(cohorts[test_idx].tolist())
                acc["pids"].extend(groups[test_idx].tolist())

        out = {}
        for method in methods:
            key = ensemble_model_name(method)
            acc = accum[key]
            probs = acc["probs"]
            if not binary_task and acc.get("blocks"):
                probs = np.vstack(acc["blocks"]).tolist()
            out[key] = self._finalize_oof_payload(
                key, acc["true"], probs, acc["cohorts"], acc["pids"],
                acc["pred_ty"], acc["pred_05"], acc["thresh"],
            )
        logger.info(
            f"Ensemble fast LOSO done ({len(unique_subjects)} subjects, "
            f"{len(checkpoints)} bases, methods={methods})"
        )
        return out

    # ------------------------------------------------------------------
    # FULL NESTED CV MODE — parallelized LOSO with Optuna retuning
    # Use only for final publication-level reporting
    # ------------------------------------------------------------------

    def _evaluate_one_subject(self, subj, name, X, y, groups, cohorts):
        """Single LOSO fold — called in parallel."""
        test_idx  = np.where(groups == subj)[0]
        train_idx = np.where(groups != subj)[0]

        if len(np.unique(y[train_idx])) < 2:
            return None, None, None, None, None, None, None

        # FIX: use the correct trainer API (_run_optuna + _build_pipeline_from_params)
        # instead of the non-existent _tune_model_with_budget.
        best_params, _ = self.trainer._run_optuna(
            name,
            X[train_idx],
            y[train_idx],
            groups[train_idx],
            n_trials=self.nested_trials,
            timeout=self.nested_timeout,
        )
        best_pipeline = self.trainer._build_pipeline_from_params(
            name, best_params, y[train_idx]
        )
        best_pipeline.fit(X[train_idx], y[train_idx])

        test_prob, pred_ty, pred_05, thresh = self._loso_fold_classifications(
            best_pipeline, X, y, train_idx, test_idx
        )
        return (
            test_prob.tolist(),
            y[test_idx].tolist(),
            cohorts[test_idx].tolist(),
            groups[test_idx].tolist(),
            pred_ty.tolist(),
            pred_05.tolist(),
            thresh,
        )

    def _nested_group_evaluate_model(self, name, X, y, groups, cohorts):
        """Parallelized nested LOSO-CV with per-fold Optuna retuning."""
        unique_subjects = np.unique(groups)

        fold_results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._evaluate_one_subject)(subj, name, X, y, groups, cohorts)
            for subj in tqdm(
                unique_subjects,
                desc=f"  {name} folds",
                leave=False,
                colour="red",
                bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
            )
        )

        binary_task = is_binary_task(y, self.config)
        all_probs, all_true, all_cohorts, all_pids = [], [], [], []
        all_pred_ty, all_pred_05, fold_thresholds = [], [], []
        all_proba_blocks: list[np.ndarray] = []
        for fold in fold_results:
            if fold[0] is None:
                continue
            probs, trues, fold_cohorts, fold_pids, pred_ty, pred_05, thresh = fold
            if binary_task:
                all_probs.extend(probs)
            else:
                all_proba_blocks.append(np.asarray(probs, dtype=float))
            all_true.extend(trues)
            all_cohorts.extend(fold_cohorts)
            all_pids.extend(fold_pids)
            all_pred_ty.extend(pred_ty)
            all_pred_05.extend(pred_05)
            fold_thresholds.append(thresh)

        if not binary_task:
            all_probs = np.vstack(all_proba_blocks).tolist() if all_proba_blocks else []

        return self._finalize_oof_payload(
            name, all_true, all_probs, all_cohorts, all_pids,
            all_pred_ty, all_pred_05, fold_thresholds,
        )

    def _tune_top_k_base_models(
        self, model_names: list[str], X, y, groups, train_idx: np.ndarray, top_k: int
    ) -> list[tuple[str, dict]]:
        tuned_results = []
        for name in model_names:
            best_params, best_score = self.trainer._run_optuna(
                name,
                X[train_idx],
                y[train_idx],
                groups[train_idx],
                n_trials=self.nested_trials,
                timeout=self.nested_timeout,
            )
            best_pipeline = self.trainer._build_pipeline_from_params(
                name, best_params, y[train_idx]
            )
            best_pipeline.fit(X[train_idx], y[train_idx])
            tuned_results.append((name, {"pipeline": best_pipeline, "cv_auc": best_score}))
        return sorted(tuned_results, key=lambda item: item[1]["cv_auc"], reverse=True)[:top_k]

    def _evaluate_ensemble_one_subject(
        self, subj, model_names, X, y, groups, cohorts, methods: list[str]
    ):
        """Single LOSO fold: tune bases once, evaluate all ensemble methods."""
        test_idx = np.where(groups == subj)[0]
        train_idx = np.where(groups != subj)[0]

        if len(np.unique(y[train_idx])) < 2:
            return None

        top_k = self.config["models"]["ensemble"]["top_k"]
        top_models = self._tune_top_k_base_models(
            model_names, X, y, groups, train_idx, top_k
        )
        ens_cfg = self.config["models"]["ensemble"]
        cv_folds = self._ensemble_cv_folds()
        rs = self.config["models"]["evaluation"]["random_state"]

        binary_task = is_binary_task(y, self.config)
        fold_out = {}
        for method in methods:
            key = ensemble_model_name(method)
            if method == "stacking":
                test_prob = predict_ensemble_oof_proba(
                    method,
                    top_models,
                    X[train_idx],
                    y[train_idx],
                    groups[train_idx],
                    X[test_idx],
                    cv_folds=cv_folds,
                    random_state=rs,
                )
                train_prob = predict_ensemble_oof_proba(
                    method,
                    top_models,
                    X[train_idx],
                    y[train_idx],
                    groups[train_idx],
                    X[train_idx],
                    cv_folds=cv_folds,
                    random_state=rs,
                )
                if binary_task:
                    thresh = self._youden_threshold(y[train_idx], train_prob)
                    pred_ty = (test_prob >= thresh).astype(int)
                    pred_05 = (test_prob >= 0.5).astype(int)
                else:
                    pred_ty = np.argmax(test_prob, axis=1).astype(int)
                    pred_05 = pred_ty
                    thresh = float("nan")
            else:
                fitted = build_ensemble_estimator(
                    top_models, method, cv_folds=cv_folds, random_state=rs
                )
                fitted.fit(X[train_idx], y[train_idx])
                test_prob = fitted.predict_proba(X[test_idx])
                if binary_task:
                    train_prob = fitted.predict_proba(X[train_idx])[:, 1]
                    test_prob = test_prob[:, 1]
                    thresh = self._youden_threshold(y[train_idx], train_prob)
                    pred_ty = (test_prob >= thresh).astype(int)
                    pred_05 = (test_prob >= 0.5).astype(int)
                else:
                    pred_ty = np.argmax(test_prob, axis=1).astype(int)
                    pred_05 = pred_ty
                    thresh = float("nan")

            fold_out[key] = (
                test_prob.tolist(),
                y[test_idx].tolist(),
                cohorts[test_idx].tolist(),
                groups[test_idx].tolist(),
                pred_ty.tolist(),
                pred_05.tolist(),
                thresh,
            )
        return fold_out

    def _nested_group_evaluate_ensembles(
        self, X, y, groups, cohorts, methods: list[str]
    ) -> dict[str, dict]:
        """Parallelized nested LOSO-CV for each ensemble method (shared base tuning)."""
        model_names = [
            name for name in self.config["models"]["run"]
            if name not in ("cnn_1d", "lstm")
        ]
        unique_subjects = np.unique(groups)
        keys = [ensemble_model_name(m) for m in methods]
        accum = {
            k: {
                "probs": [], "true": [], "cohorts": [], "pids": [],
                "pred_ty": [], "pred_05": [], "thresh": [],
            }
            for k in keys
        }

        fold_results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._evaluate_ensemble_one_subject)(
                subj, model_names, X, y, groups, cohorts, methods
            )
            for subj in tqdm(
                unique_subjects,
                desc="  ensemble folds",
                leave=False,
                colour="red",
                bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
            )
        )

        binary_task = is_binary_task(y, self.config)
        for fold in fold_results:
            if fold is None:
                continue
            for key, pack in fold.items():
                probs, trues, fold_cohorts, fold_pids, pred_ty, pred_05, thresh = pack
                acc = accum[key]
                if binary_task:
                    acc["probs"].extend(probs)
                else:
                    if "blocks" not in acc:
                        acc["blocks"] = []
                    acc["blocks"].append(np.asarray(probs, dtype=float))
                acc["true"].extend(trues)
                acc["cohorts"].extend(fold_cohorts)
                acc["pids"].extend(fold_pids)
                acc["pred_ty"].extend(pred_ty)
                acc["pred_05"].extend(pred_05)
                acc["thresh"].append(thresh)

        out = {}
        for key in keys:
            acc = accum[key]
            probs = acc["probs"]
            if not binary_task and acc.get("blocks"):
                probs = np.vstack(acc["blocks"]).tolist()
            out[key] = self._finalize_oof_payload(
                key,
                acc["true"],
                probs,
                acc["cohorts"],
                acc["pids"],
                acc["pred_ty"],
                acc["pred_05"],
                acc["thresh"],
            )
        return out

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _loso_fold_classifications(
        self,
        fitted_model: Any,
        X: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Binary: Youden J on train-fold positive-class probability.
        Multiclass: full predict_proba matrix and argmax class.
        """
        if not is_binary_task(y, self.config):
            test_proba = fitted_model.predict_proba(X[test_idx])
            pred = np.argmax(test_proba, axis=1).astype(int)
            return test_proba, pred, pred, float("nan")

        train_prob = fitted_model.predict_proba(X[train_idx])[:, 1]
        test_prob = fitted_model.predict_proba(X[test_idx])[:, 1]
        thresh_train = self._youden_threshold(y[train_idx], train_prob)
        test_pred_train_youden = (test_prob >= thresh_train).astype(int)
        test_pred_fixed = (test_prob >= 0.5).astype(int)
        return test_prob, test_pred_train_youden, test_pred_fixed, thresh_train

    def _log_model_metrics(self, name: str, result: dict, binary_task: bool) -> None:
        if binary_task:
            logger.info(
                f"{name:20s}  "
                f"AUC={result['auc']:.4f}  "
                f"[{result['auc_ci_low']:.3f}, {result['auc_ci_high']:.3f}]  "
                f"Sens={result['sensitivity']:.3f}  "
                f"Spec={result['specificity']:.3f}  "
                f"F1={result['f1']:.4f}  "
                f"AUC-PR={result['auc_pr']:.4f}"
            )
            return
        per = result.get("per_class_metrics", {})
        per_str = ", ".join(
            f"{k}: F1={v['f1']:.3f}" for k, v in per.items()
        ) if per else "n/a"
        logger.info(
            f"{name:20s}  macro-OVR AUC={result['auc']:.4f}  "
            f"macro-F1={result['f1']:.4f}  acc={result['accuracy']:.4f}  "
            f"per-class [{per_str}]"
        )

    def _finalize_oof_payload(
        self,
        name: str,
        all_true: list,
        all_probs: list,
        all_cohorts: list,
        all_pids: list,
        all_pred_train_youden: list | None = None,
        all_pred_fixed: list | None = None,
        fold_thresholds: list | None = None,
    ) -> dict:
        y_true = np.array(all_true)
        cohorts_arr = np.array(all_cohorts)

        if not is_binary_task(y_true, self.config):
            y_proba = np.asarray(all_probs, dtype=float)
            if y_proba.ndim == 1 and y_proba.size and y_proba.size != len(y_true):
                y_proba = np.vstack(
                    [np.asarray(row, dtype=float) for row in all_probs]
                )
            y_pred = (
                np.array(all_pred_train_youden, dtype=int)
                if all_pred_train_youden is not None
                else np.argmax(y_proba, axis=1).astype(int)
            )
            payload = build_multiclass_metric_payload(
                name, y_true, y_proba, y_pred, cohorts=cohorts_arr
            )
            payload["participant_ids"] = np.array(all_pids)
            payload["fold_thresholds"] = np.array(fold_thresholds or [])
            return payload

        y_prob = np.array(all_probs)

        if all_pred_train_youden is not None:
            y_pred = np.array(all_pred_train_youden, dtype=int)
        else:
            y_pred = (y_prob >= 0.5).astype(int)

        y_pred_fixed = (
            np.array(all_pred_fixed, dtype=int)
            if all_pred_fixed is not None
            else (y_prob >= 0.5).astype(int)
        )

        thresh_train_mean = (
            float(np.mean(fold_thresholds))
            if fold_thresholds
            else float("nan")
        )
        thresh_eval_youden = self._youden_threshold(y_true, y_prob)

        payload = self._build_metric_payload(
            name,
            y_true,
            y_prob,
            cohorts_arr,
            y_pred=y_pred,
            decision_threshold=thresh_train_mean,
            threshold_strategy="train_fold_youden",
        )

        cmp = self._threshold_comparison_metrics(
            y_true, y_prob, y_pred, y_pred_fixed, thresh_train_mean, thresh_eval_youden
        )
        payload.update(cmp)
        payload["participant_ids"] = np.array(all_pids)
        payload["fold_thresholds"] = np.array(fold_thresholds or [])
        payload["y_pred_fixed"] = y_pred_fixed
        payload["y_pred_eval_youden"] = (y_prob >= thresh_eval_youden).astype(int)

        logger.info(
            f"{name} thresholds — train-fold Youden (mean)={thresh_train_mean:.3f}, "
            f"eval Youden (optimistic)={thresh_eval_youden:.3f}, "
            f"Δacc(eval−train)={cmp['delta_accuracy_eval_minus_train']:.3f}"
        )
        return payload

    def _load_data(self):
        X, y, groups, feat_cols, df = load_patient_feature_matrix(self.config)
        cohorts = df["cohort"].astype(str).values
        logger.info(f"Evaluation feature matrix: {X.shape[1]} columns")
        return X, y, groups, feat_cols, cohorts

    def _load_checkpoint(self, model_name: str):
        path = self.ckpt_dir / f"{model_name}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _youden_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Youden J threshold (max TPR − FPR) — must be fit on training data only at eval time."""
        return youden_threshold(y_true, y_prob)

    def _export_clinical_threshold(self, results: dict, best_name: str) -> None:
        """Export Youden cutoff + sens/spec for API and paper (binary / collapsed multiclass)."""
        res = results.get(best_name)
        if res is None:
            logger.warning("Clinical threshold export skipped — no best model result.")
            return

        y_true = np.asarray(res["y_true"])
        y_proba_full = res.get("y_proba_full")
        y_prob = np.asarray(res["y_prob"], dtype=float)
        if y_proba_full is not None:
            y_proba = np.asarray(y_proba_full, dtype=float)
        elif y_prob.ndim == 1:
            y_proba = np.column_stack([1.0 - y_prob, y_prob])
        else:
            y_proba = y_prob

        train_youden = res.get("threshold_train_youden_mean")
        if train_youden is None:
            fold_t = res.get("fold_thresholds")
            if fold_t is not None and len(fold_t):
                train_youden = float(np.mean(fold_t))

        train_metrics = {
            "sensitivity": res.get("sensitivity", float("nan")),
            "specificity": res.get("specificity", float("nan")),
            "accuracy": res.get("accuracy", float("nan")),
        }

        payload = build_clinical_threshold_artifact(
            self.config,
            reference_model=best_name,
            y_true=y_true,
            y_proba=y_proba,
            train_fold_youden_mean=float(train_youden) if train_youden is not None else None,
            train_fold_metrics=train_metrics,
            validation_strategy=self.validation_strategy,
        )
        path = save_clinical_threshold_artifact(payload, self.metrics_dir)
        logger.info(
            "Clinical threshold (Youden J) -> %s | primary prob=%.3f sens=%.3f spec=%.3f",
            path,
            payload["primary_cutoff"]["probability"],
            payload["primary_cutoff"].get("sensitivity", float("nan")),
            payload["primary_cutoff"].get("specificity", float("nan")),
        )

    def _classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "sensitivity": float(tp / (tp + fn + 1e-10)),
            "specificity": float(tn / (tn + fp + 1e-10)),
            "confusion_matrix": cm,
        }

    def _threshold_comparison_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred_train_youden: np.ndarray,
        y_pred_fixed: np.ndarray,
        thresh_train_mean: float,
        thresh_eval_youden: float,
    ) -> dict:
        """Compare discrete metrics under train-fold Youden, fixed 0.5, and eval Youden."""
        m_train = self._classification_metrics(y_true, y_pred_train_youden)
        m_fixed = self._classification_metrics(y_true, y_pred_fixed)
        m_eval = self._classification_metrics(
            y_true, (y_prob >= thresh_eval_youden).astype(int)
        )
        return {
            "threshold_train_youden_mean": thresh_train_mean,
            "threshold_eval_youden": thresh_eval_youden,
            "threshold_fixed": 0.5,
            "accuracy_at_0.5": m_fixed["accuracy"],
            "f1_at_0.5": m_fixed["f1"],
            "sensitivity_at_0.5": m_fixed["sensitivity"],
            "specificity_at_0.5": m_fixed["specificity"],
            "accuracy_eval_youden": m_eval["accuracy"],
            "f1_eval_youden": m_eval["f1"],
            "sensitivity_eval_youden": m_eval["sensitivity"],
            "specificity_eval_youden": m_eval["specificity"],
            "delta_accuracy_eval_minus_train": m_eval["accuracy"] - m_train["accuracy"],
            "delta_f1_eval_minus_train": m_eval["f1"] - m_train["f1"],
        }

    def _bootstrap_auc_ci(self, y_true: np.ndarray, y_prob: np.ndarray, n_bootstrap: int = 1000) -> tuple[float, float]:
        rng = np.random.default_rng(self.trainer.random_state)
        idx_all = np.arange(len(y_true))
        samples: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            yt = y_true[idx]
            yp = y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            samples.append(float(roc_auc_score(yt, yp)))
        if not samples:
            return float("nan"), float("nan")
        return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

    def _build_metric_payload(
        self,
        name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        cohorts: np.ndarray | None = None,
        *,
        y_pred: np.ndarray | None = None,
        decision_threshold: float = 0.5,
        threshold_strategy: str = "train_fold_youden",
    ):
        if len(y_true) == 0:
            raise ValueError(f"{name} evaluation failed: no valid grouped splits")

        if y_pred is None:
            y_pred = (y_prob >= decision_threshold).astype(int)

        if len(np.unique(y_true)) < 2:
            auc_roc = float("nan")
            auc_pr  = float("nan")
            auc_ci_low, auc_ci_high = float("nan"), float("nan")
            fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
        else:
            auc_roc     = roc_auc_score(y_true, y_prob)
            auc_pr      = average_precision_score(y_true, y_prob)
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_ci_low, auc_ci_high = self._bootstrap_auc_ci(y_true, y_prob)

        cls = self._classification_metrics(y_true, y_pred)

        return {
            "model":            name,
            "auc":              float(auc_roc),
            "auc_pr":           float(auc_pr),
            "auc_ci_low":       auc_ci_low,
            "auc_ci_high":      auc_ci_high,
            "decision_threshold": decision_threshold,
            "threshold_strategy": threshold_strategy,
            "f1":               cls["f1"],
            "accuracy":         cls["accuracy"],
            "sensitivity":      cls["sensitivity"],
            "specificity":      cls["specificity"],
            "fpr":              fpr,
            "tpr":              tpr,
            "y_true":           y_true,
            "y_prob":           y_prob,
            "y_pred":           y_pred,
            "confusion_matrix": cls["confusion_matrix"],
            "report":           classification_report(y_true, y_pred, output_dict=True),
            "cohorts":          cohorts,
        }

    def _cohort_metric_rows(self, model_name: str, result: dict) -> list[dict]:
        cohorts = result.get("cohorts")
        if cohorts is None:
            return []

        y_true = result["y_true"]
        y_prob = result["y_prob"]
        y_pred = result["y_pred"]
        thresh = float(result.get("decision_threshold", 0.5))
        rows: list[dict] = []
        for cohort in sorted(set(cohorts)):
            mask = cohorts == cohort
            if int(mask.sum()) < 2:
                continue
            sub_true = y_true[mask]
            sub_prob = y_prob[mask]
            sub_pred = y_pred[mask]
            try:
                sub = self._build_metric_payload(
                    f"{model_name}:{cohort}",
                    sub_true,
                    sub_prob,
                    cohorts[mask],
                    y_pred=sub_pred,
                    decision_threshold=thresh,
                )
            except ValueError:
                continue
            rows.append({
                "model":               model_name,
                "cohort":              cohort,
                "n":                   int(mask.sum()),
                "auc":                 sub["auc"],
                "auc_ci_low":          sub["auc_ci_low"],
                "auc_ci_high":         sub["auc_ci_high"],
                "auc_pr":              sub["auc_pr"],
                "f1":                  sub["f1"],
                "sensitivity":         sub["sensitivity"],
                "specificity":         sub["specificity"],
                "accuracy":            sub["accuracy"],
                "evaluation_mode":     "fast" if self.fast else "full_nested",
                "validation_strategy": self.validation_strategy,
            })
        return rows

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_roc_all(self, results):
        fig, ax = plt.subplots()
        for name, res in results.items():
            ax.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.legend()
        ax.set_title("Nested Grouped ROC Curves")
        self._save(fig, "roc")

    def _plot_pr_curves(self, results):
        fig, ax = plt.subplots()
        for name, res in results.items():
            prec, rec, _ = precision_recall_curve(res["y_true"], res["y_prob"])
            ax.plot(rec, prec, label=name)
        ax.legend()
        ax.set_title("Nested Grouped PR Curves")
        self._save(fig, "pr")

    def _plot_calibration(self, results):
        fig, ax = plt.subplots()
        for name, res in results.items():
            try:
                frac_pos, mean_pred = calibration_curve(
                    res["y_true"], res["y_prob"], n_bins=10
                )
                ax.plot(mean_pred, frac_pos, label=name)
            except Exception:
                continue
        ax.legend()
        ax.set_title("Nested Grouped Calibration")
        self._save(fig, "calibration")

    def _plot_multiclass_calibration(self, results: dict) -> None:
        """Per-class OvR reliability diagrams + Brier score for multiclass."""
        from src.dataset.label_policy import MULTICLASS_NAMES

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

            brier_scores = {}
            for i, lbl in enumerate(labels):
                ax = axes[i]
                class_name = MULTICLASS_NAMES.get(lbl, str(lbl))
                y_bin = (y_true == lbl).astype(int)
                p_bin = y_proba[:, lbl] if lbl < y_proba.shape[1] else np.zeros(len(y_true))

                try:
                    frac_pos, mean_pred = calibration_curve(y_bin, p_bin, n_bins=10)
                    ax.plot(mean_pred, frac_pos, "s-", color=self._OVR_COLORS[i % len(self._OVR_COLORS)], lw=2)
                except (ValueError, IndexError):
                    pass

                ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Fraction of positives")
                ax.set_title(class_name, fontsize=10)
                ax.set_xlim(-0.02, 1.02)
                ax.set_ylim(-0.02, 1.02)

                brier = float(np.mean((p_bin - y_bin) ** 2))
                brier_scores[class_name] = brier
                ax.text(0.05, 0.9, f"Brier={brier:.3f}", transform=ax.transAxes, fontsize=9)

            fig.suptitle(f"{name} — per-class calibration (OvR)", fontsize=12)
            fig.tight_layout()
            self._save(fig, f"calibration_ovr_{name}")

        best_name = max(results, key=lambda n: results[n]["auc"])
        best_res = results[best_name]
        y_true = np.asarray(best_res["y_true"]).astype(int)
        y_proba = best_res.get("y_proba_full")
        if y_proba is not None:
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
                    "mean_predicted_prob": float(np.mean(p_bin)),
                    "prevalence": float(np.mean(y_bin)),
                })
            if brier_rows:
                pd.DataFrame(brier_rows).to_csv(
                    self.metrics_dir / "calibration_brier_scores.csv", index=False
                )
                logger.info("Brier scores saved → calibration_brier_scores.csv")

    def _plot_confusion_matrices(self, results):
        for name, res in results.items():
            fig, ax = plt.subplots()
            cm = res["confusion_matrix"]
            ax.imshow(cm, cmap="Blues")
            ax.set_title(name)
            self._save(fig, f"cm_{name}")

    _OVR_COLORS = ["#2196F3", "#FF9800", "#F44336", "#4CAF50", "#9C27B0"]

    def _plot_multiclass_roc(self, results: dict) -> None:
        """Per-class OvR ROC curves for every model (multiclass)."""
        for name, res in results.items():
            per_class_roc = res.get("per_class_roc")
            if not per_class_roc:
                continue

            fig, ax = plt.subplots(figsize=(7, 6))
            for i, (lbl, roc_data) in enumerate(sorted(per_class_roc.items())):
                c = self._OVR_COLORS[i % len(self._OVR_COLORS)]
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
                c = self._OVR_COLORS[i % len(self._OVR_COLORS)]
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

    def _save_per_class_auc_table(self, results: dict) -> None:
        """Save per-class AUC / sensitivity / specificity table for all models."""
        rows = []
        for name, res in results.items():
            per_class = res.get("per_class_metrics", {})
            for class_name, metrics in per_class.items():
                rows.append({
                    "model": name,
                    "class": class_name,
                    "auc_ovr": metrics.get("auc_ovr", float("nan")),
                    "avg_precision": metrics.get("avg_precision", float("nan")),
                    "sensitivity": metrics.get("sensitivity", float("nan")),
                    "specificity": metrics.get("specificity", float("nan")),
                    "precision": metrics.get("precision", float("nan")),
                    "recall": metrics.get("recall", float("nan")),
                    "f1": metrics.get("f1", float("nan")),
                    "support": metrics.get("support", 0),
                })
        if rows:
            df = pd.DataFrame(rows)
            out = self.metrics_dir / "per_class_metrics.csv"
            df.to_csv(out, index=False)
            logger.info(f"Per-class AUC/metrics table saved → {out}")

    def _shap_analysis(
        self, name: str, X, y, groups, feat_names: list[str],
        cohorts: np.ndarray | None = None,
    ) -> None:
        if not self.config["explainability"]["shap_enabled"]:
            return

        strategy = self.config["explainability"].get("shap_strategy", "loso_aggregate")
        if strategy == "full_checkpoint":
            self._shap_full_checkpoint_background(name, X, y, feat_names)
        else:
            self._shap_loso_aggregate(name, X, y, groups, feat_names, cohorts=cohorts)

    def _unwrap_pipeline_model(self, model: Any) -> tuple[Any, Any]:
        """Return (classifier, preprocess_transformer or None)."""
        if hasattr(model, "named_steps"):
            if "clf" in model.named_steps:
                return model.named_steps["clf"], model[:-1]
            if len(model.named_steps) > 0:
                last_key = list(model.named_steps.keys())[-1]
                return model.named_steps[last_key], model[:-1]
        return model, None

    def _transform_for_shap(self, model: Any, X: np.ndarray) -> np.ndarray:
        _, preprocess = self._unwrap_pipeline_model(model)
        if preprocess is not None:
            return preprocess.transform(X)
        return X

    def _normalize_shap_values(self, shap_vals: Any) -> np.ndarray:
        """Binary classifier: return (n_samples, n_features) SHAP matrix."""
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        arr = np.asarray(shap_vals, dtype=float)
        if arr.ndim == 3:
            arr = arr[:, :, 1]
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def _compute_shap_matrix(
        self, name: str, model: Any, X_proc: np.ndarray
    ) -> np.ndarray | None:
        clf, _ = self._unwrap_pipeline_model(model)
        try:
            if name in ("xgboost", "lightgbm", "random_forest"):
                explainer = shap.TreeExplainer(clf)
                return self._normalize_shap_values(explainer.shap_values(X_proc))
            background = shap.sample(X_proc, min(50, max(len(X_proc), 1)))
            explainer = shap.KernelExplainer(
                lambda x: clf.predict_proba(x)[:, 1],
                background,
            )
            return self._normalize_shap_values(
                explainer.shap_values(X_proc, nsamples=min(50, len(X_proc)))
            )
        except Exception as exc:
            logger.warning(f"SHAP failed for {name}: {exc}")
            return None

    def _shap_loso_aggregate(
        self, name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
        feat_names: list[str], *, cohorts: np.ndarray | None = None,
    ) -> None:
        """
        Compute SHAP on every LOSO held-out fold (refit per fold), concatenate
        all OOF explanations, and aggregate mean |SHAP| for global importance.
        Then produce per-cohort SHAP breakdowns.
        """
        checkpoint = self._load_checkpoint(name)
        if checkpoint is None:
            logger.warning(f"Skipping SHAP — checkpoint not found for {name}")
            return

        import sklearn.base as skbase

        unique_subjects = np.unique(groups)
        exp_cfg = self.config["explainability"]
        max_oof = int(exp_cfg.get("n_shap_samples", 200))

        fold_shap_blocks: list[np.ndarray] = []
        fold_x_blocks: list[np.ndarray] = []
        fold_cohort_blocks: list[np.ndarray] = []
        n_explained = 0

        for subj in tqdm(
            unique_subjects,
            desc=f"  SHAP {name} LOSO",
            leave=False,
            colour="green",
        ):
            if n_explained >= max_oof:
                break

            test_idx = np.where(groups == subj)[0]
            train_idx = np.where(groups != subj)[0]
            if len(np.unique(y[train_idx])) < 2:
                continue

            fold_model = skbase.clone(checkpoint)
            fold_model.fit(X[train_idx], y[train_idx])

            remaining = max_oof - n_explained
            idx = test_idx[:remaining]
            X_test = X[idx]
            X_proc = self._transform_for_shap(fold_model, X_test)

            shap_block = self._compute_shap_matrix(name, fold_model, X_proc)
            if shap_block is None or shap_block.size == 0:
                continue

            fold_shap_blocks.append(shap_block)
            fold_x_blocks.append(X_proc)
            if cohorts is not None:
                fold_cohort_blocks.append(cohorts[idx])
            n_explained += len(X_proc)

        if not fold_shap_blocks:
            logger.warning(f"No SHAP values collected for {name}")
            return

        shap_all = np.vstack(fold_shap_blocks)
        x_all = np.vstack(fold_x_blocks)
        mean_abs = np.abs(shap_all).mean(axis=0)

        self._save_shap_importance_table(name, feat_names, mean_abs, shap_all)
        self._plot_shap_summary(name, shap_all, x_all, feat_names)
        self._plot_shap_mean_bar(name, feat_names, mean_abs)

        logger.info(
            f"SHAP {name}: LOSO aggregate over {shap_all.shape[0]} OOF rows "
            f"({len(fold_shap_blocks)} folds)"
        )

        if cohorts is not None and fold_cohort_blocks:
            cohort_labels = np.concatenate(fold_cohort_blocks)
            self._shap_per_cohort(name, shap_all, x_all, cohort_labels, feat_names)

    def _shap_full_checkpoint_background(
        self, name: str, X: np.ndarray, y: np.ndarray, feat_names: list[str]
    ) -> None:
        """TreeExplainer on the full-data checkpoint with a training-set background sample."""
        checkpoint = self._load_checkpoint(name)
        if checkpoint is None:
            logger.warning(f"Skipping SHAP — checkpoint not found for {name}")
            return

        if name not in ("xgboost", "lightgbm", "random_forest"):
            logger.warning(
                f"SHAP full_checkpoint strategy is slow for {name}; "
                "using LOSO aggregate instead"
            )
            groups = np.arange(len(y))
            self._shap_loso_aggregate(name, X, y, groups, feat_names)
            return

        checkpoint.fit(X, y)
        clf, _ = self._unwrap_pipeline_model(checkpoint)
        x_proc = self._transform_for_shap(checkpoint, X)

        n_bg = int(self.config["explainability"].get("shap_background_samples", 100))
        n_explain = min(int(self.config["explainability"]["n_shap_samples"]), len(x_proc))
        rng = np.random.default_rng(self.trainer.random_state)
        bg_idx = rng.choice(len(x_proc), size=min(n_bg, len(x_proc)), replace=False)
        ex_idx = rng.choice(len(x_proc), size=n_explain, replace=False)

        try:
            explainer = shap.TreeExplainer(clf, data=x_proc[bg_idx])
            shap_vals = self._normalize_shap_values(explainer.shap_values(x_proc[ex_idx]))
        except Exception as exc:
            logger.warning(f"SHAP full checkpoint failed for {name}: {exc}")
            return

        mean_abs = np.abs(shap_vals).mean(axis=0)
        self._save_shap_importance_table(name, feat_names, mean_abs, shap_vals)
        self._plot_shap_summary(name, shap_vals, x_proc[ex_idx], feat_names)
        self._plot_shap_mean_bar(name, feat_names, mean_abs)
        logger.info(f"SHAP {name}: full checkpoint + background (n={n_explain})")

    def _save_shap_importance_table(
        self,
        name: str,
        feat_names: list[str],
        mean_abs: np.ndarray,
        shap_all: np.ndarray,
    ) -> None:
        df = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": mean_abs,
            "std_abs_shap": np.abs(shap_all).std(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        df.to_csv(self.metrics_dir / f"shap_importance_{name}.csv", index=False)

    def _plot_shap_summary(
        self,
        name: str,
        shap_vals: np.ndarray,
        x_proc: np.ndarray,
        feat_names: list[str],
    ) -> None:
        top_k = int(self.config["explainability"].get("top_features_plot", 20))
        try:
            shap.summary_plot(
                shap_vals,
                x_proc,
                feature_names=feat_names,
                max_display=top_k,
                show=False,
            )
            fig = plt.gcf()
            fig.suptitle(f"{name} — SHAP (aggregated LOSO OOF)")
            self._save_shap(fig, f"shap_{name}")
        except Exception as exc:
            logger.warning(f"SHAP summary plot failed for {name}: {exc}")

    def _plot_shap_mean_bar(
        self, name: str, feat_names: list[str], mean_abs: np.ndarray
    ) -> None:
        top_k = int(self.config["explainability"].get("top_features_plot", 20))
        order = np.argsort(mean_abs)[::-1][:top_k]
        labels = [feat_names[i] for i in order][::-1]
        values = mean_abs[order][::-1]

        fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.25)))
        ax.barh(labels, values, color="steelblue")
        ax.set_xlabel("Mean |SHAP| (LOSO OOF aggregate)")
        ax.set_title(f"{name} — global feature importance")
        fig.tight_layout()
        self._save_shap(fig, f"shap_bar_{name}")

    def _shap_per_cohort(
        self,
        name: str,
        shap_all: np.ndarray,
        x_all: np.ndarray,
        cohort_labels: np.ndarray,
        feat_names: list[str],
    ) -> None:
        """Per-cohort SHAP importance tables and bar plots."""
        unique_cohorts = np.unique(cohort_labels)
        top_k = int(self.config["explainability"].get("top_features_plot", 20))
        all_rows = []

        for cohort in sorted(unique_cohorts):
            mask = cohort_labels == cohort
            if mask.sum() < 2:
                continue
            shap_c = shap_all[mask]
            mean_abs_c = np.abs(shap_c).mean(axis=0)

            for i, fname in enumerate(feat_names):
                all_rows.append({
                    "cohort": cohort,
                    "feature": fname,
                    "mean_abs_shap": float(mean_abs_c[i]),
                    "std_abs_shap": float(np.abs(shap_c[:, i]).std()),
                    "n_subjects": int(mask.sum()),
                })

            order = np.argsort(mean_abs_c)[::-1][:top_k]
            labels = [feat_names[i] for i in order][::-1]
            values = mean_abs_c[order][::-1]

            fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.25)))
            ax.barh(labels, values, color="steelblue")
            ax.set_xlabel("Mean |SHAP|")
            ax.set_title(f"{name} — {cohort} (n={mask.sum()})")
            fig.tight_layout()
            self._save_shap(fig, f"shap_bar_{name}_{cohort}")

        if all_rows:
            df = pd.DataFrame(all_rows)
            out = self.metrics_dir / f"shap_importance_{name}_per_cohort.csv"
            df.to_csv(out, index=False)
            logger.info(
                f"Per-cohort SHAP saved → {out} "
                f"({len(unique_cohorts)} cohorts)"
            )

    def _leakage_comparison(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
        grouped_results: dict[str, dict],
    ) -> None:
        """Run ungrouped StratifiedKFold CV to quantify subject-leakage inflation.

        Produces leakage_comparison.csv with grouped (LOSO) vs ungrouped AUC
        for each model, showing the optimistic bias from subject leakage.
        """
        eval_cfg = self.config["models"]["evaluation"]
        if not eval_cfg.get("leakage_comparison", True):
            logger.info("Leakage comparison disabled in config — skipping.")
            return

        from sklearn.model_selection import StratifiedKFold
        import sklearn.base as skbase

        n_folds = eval_cfg.get("leakage_kfold_splits", 10)
        rs = eval_cfg["random_state"]
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rs)
        binary_task = is_binary_task(y, self.config)
        rows = []

        model_names = [n for n in grouped_results if not n.startswith("ensemble_")]

        for name in model_names:
            checkpoint = self._load_checkpoint(name)
            if checkpoint is None:
                continue

            all_true, all_prob = [], []
            for train_idx, test_idx in skf.split(X, y):
                if len(np.unique(y[train_idx])) < 2:
                    continue
                fold_model = skbase.clone(checkpoint)
                fold_model.fit(X[train_idx], y[train_idx])

                if binary_task:
                    proba = fold_model.predict_proba(X[test_idx])[:, 1]
                    all_prob.extend(proba.tolist())
                else:
                    proba = fold_model.predict_proba(X[test_idx])
                    all_prob.append(proba)
                all_true.extend(y[test_idx].tolist())

            if not all_true:
                continue

            y_true = np.array(all_true)
            if binary_task:
                y_prob = np.array(all_prob)
                ungrouped_auc = float(roc_auc_score(y_true, y_prob))
            else:
                y_prob = np.vstack(all_prob)
                ungrouped_auc = float(
                    roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
                )

            grouped_auc = float(grouped_results[name]["auc"])
            inflation = ungrouped_auc - grouped_auc

            rows.append({
                "model": name,
                "auc_grouped_loso": grouped_auc,
                "auc_ungrouped_kfold": ungrouped_auc,
                "auc_inflation": inflation,
                "inflation_pct": float(inflation / (grouped_auc + 1e-10) * 100),
                "grouped_strategy": "LOSO",
                "ungrouped_strategy": f"StratifiedKFold(k={n_folds})",
                "n_participants": len(np.unique(groups)),
            })
            logger.info(
                f"  Leakage check {name}: "
                f"grouped={grouped_auc:.4f} ungrouped={ungrouped_auc:.4f} "
                f"inflation={inflation:+.4f} ({inflation / (grouped_auc + 1e-10) * 100:+.1f}%)"
            )

        if rows:
            df = pd.DataFrame(rows)
            out = self.metrics_dir / "leakage_comparison.csv"
            df.to_csv(out, index=False)
            logger.info(f"Subject-leakage comparison saved → {out}")

    def _save_ensemble_comparison(self, ensemble_results: dict[str, dict]) -> None:
        """Write ensemble-method comparison table and paired DeLong p-value."""
        if not ensemble_results:
            return

        rows = []
        for name, res in ensemble_results.items():
            method = name.replace("ensemble_", "", 1) if name.startswith("ensemble_") else name
            rows.append({
                "model": name,
                "ensemble_method": method,
                "auc": res["auc"],
                "auc_ci_low": res.get("auc_ci_low", float("nan")),
                "auc_ci_high": res.get("auc_ci_high", float("nan")),
                "auc_pr": res.get("auc_pr", float("nan")),
                "accuracy": res["accuracy"],
                "f1": res["f1"],
                "sensitivity": res.get("sensitivity", float("nan")),
                "specificity": res.get("specificity", float("nan")),
                "evaluation_mode": "fast" if self.fast else "full_nested",
            })
        comp_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
        comp_path = self.metrics_dir / "ensemble_comparison.csv"
        comp_df.to_csv(comp_path, index=False)
        logger.info(f"Ensemble comparison saved → {comp_path}")

        if len(ensemble_results) < 2:
            return

        first = next(iter(ensemble_results.values()))
        if not is_binary_task(np.asarray(first["y_true"]), self.config):
            logger.info(
                "Skipping ensemble pairwise AUC tests (DeLong/bootstrap are binary-only)."
            )
            return

        eval_cfg = self.config.get("models", {}).get("evaluation", {})
        n_boot = int(eval_cfg.get("delong_bootstrap_n", 1000))
        seed = int(eval_cfg.get("random_state", self.trainer.random_state))
        pairwise_df, _ = pairwise_auc_significance(
            ensemble_results,
            reference=max(ensemble_results, key=lambda n: ensemble_results[n]["auc"]),
            n_bootstrap=n_boot,
            random_state=seed,
        )
        if not pairwise_df.empty:
            pairwise_df.to_csv(
                self.metrics_dir / "ensemble_pairwise_pvalues.csv", index=False
            )

    def _run_auc_pairwise_tests(self, results: dict, reference: str):
        eval_cfg = self.config.get("models", {}).get("evaluation", {})
        if not eval_cfg.get("auc_pairwise_tests", True):
            return pd.DataFrame(), pd.DataFrame()

        ref = eval_cfg.get("auc_reference_model") or reference
        n_boot = int(eval_cfg.get("delong_bootstrap_n", 1000))
        seed = int(eval_cfg.get("random_state", self.trainer.random_state))

        return pairwise_auc_significance(
            results,
            reference=ref,
            n_bootstrap=n_boot,
            random_state=seed,
        )

    def _run_mcnemar_tests(self, results: dict, reference: str):
        eval_cfg = self.config.get("models", {}).get("evaluation", {})
        if not eval_cfg.get("mcnemar_tests", True):
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        ref = eval_cfg.get("mcnemar_reference_model") or eval_cfg.get("auc_reference_model")
        if not ref:
            ref = max(results, key=lambda n: float(results[n]["accuracy"]))

        exact = bool(eval_cfg.get("mcnemar_exact", False))
        return pairwise_classification_significance(
            results, reference=ref, exact_mcnemar=exact
        )

    def _save_oof_predictions(self, results: dict) -> None:
        rows = []
        for model_name, res in results.items():
            y_pred = res.get("y_pred")
            if y_pred is None:
                y_pred = (np.asarray(res["y_prob"]) >= float(res.get("decision_threshold", 0.5))).astype(int)
            pids = res.get("participant_ids", [])
            y_pred_fixed = res.get("y_pred_fixed", y_pred)
            for i, (yt, yp) in enumerate(zip(res["y_true"], res["y_prob"])):
                row = {
                    "model": model_name,
                    "y_true": int(yt),
                    "y_prob": float(yp),
                    "y_pred": int(y_pred[i]),
                    "y_pred_fixed": int(y_pred_fixed[i]),
                    "threshold_train_youden": float(res.get("decision_threshold", 0.5)),
                    "threshold_eval_youden": float(res.get("threshold_eval_youden", 0.5)),
                }
                if len(pids) > i:
                    row["participant_id"] = str(pids[i])
                rows.append(row)
        if rows:
            pd.DataFrame(rows).to_parquet(
                self.metrics_dir / "oof_predictions.parquet",
                index=False,
            )

    def _save_all_metrics(
        self,
        results: dict,
        n_participants: int,
        vs_ref_df: pd.DataFrame | None = None,
        mcnemar_vs_ref_df: pd.DataFrame | None = None,
    ):
        p_map = {}
        if vs_ref_df is not None and not vs_ref_df.empty:
            p_map = vs_ref_df.set_index("model").to_dict(orient="index")

        m_map = {}
        if mcnemar_vs_ref_df is not None and not mcnemar_vs_ref_df.empty:
            m_map = mcnemar_vs_ref_df.set_index("model").to_dict(orient="index")

        rows = []
        for name, res in results.items():
            pvals = p_map.get(name, {})
            mvals = m_map.get(name, {})
            ref_model = pvals.get("reference_model") or mvals.get("reference_model", "")
            per_class = res.get("per_class_metrics", {})
            rows.append({
                "model":               name,
                "label_mode":          res.get("label_mode", "binary"),
                "auc":                 res["auc"],
                "auc_weighted_ovr":    res.get("auc_weighted_ovr", float("nan")),
                "macro_f1":            res.get("macro_f1", res["f1"]),
                "f1_weighted":         res.get("f1_weighted", float("nan")),
                "auc_ci_low":          res.get("auc_ci_low", float("nan")),
                "auc_ci_high":         res.get("auc_ci_high", float("nan")),
                "auc_pr":              res.get("auc_pr", float("nan")),
                "accuracy":            res["accuracy"],
                "f1":                  res["f1"],
                "sensitivity":         res.get("sensitivity", 0.0),
                "specificity":         res.get("specificity", 0.0),
                "decision_threshold":  res.get("decision_threshold", 0.5),
                "threshold_strategy": res.get("threshold_strategy", "train_fold_youden"),
                "threshold_eval_youden": res.get("threshold_eval_youden", float("nan")),
                "accuracy_at_0.5":     res.get("accuracy_at_0.5", float("nan")),
                "f1_at_0.5":           res.get("f1_at_0.5", float("nan")),
                "accuracy_eval_youden": res.get("accuracy_eval_youden", float("nan")),
                "delta_accuracy_eval_minus_train": res.get(
                    "delta_accuracy_eval_minus_train", float("nan")
                ),
                "p_delong_vs_best":    pvals.get("p_delong_vs_reference", float("nan")),
                "p_bootstrap_mwu_vs_best": pvals.get("p_bootstrap_mwu_vs_reference", float("nan")),
                "p_mcnemar_vs_best":   mvals.get("p_mcnemar_vs_reference", float("nan")),
                "auc_reference_model": ref_model,
                "p_delong_fmt":        pvals.get("p_delong_fmt", ""),
                "p_mcnemar_fmt":       mvals.get("p_mcnemar_fmt", ""),
                "evaluation_mode":     "fast" if self.fast else "full_nested",
                "validation_strategy": self.validation_strategy,
                "participants":        n_participants,
                **{
                    f"per_class_{k}_f1": v.get("f1", float("nan"))
                    for k, v in per_class.items()
                },
            })
        df = pd.DataFrame(rows).sort_values("auc", ascending=False)
        df.to_csv(self.metrics_dir / "metrics.csv", index=False)
        if vs_ref_df is not None and not vs_ref_df.empty:
            vs_ref_df.to_csv(self.metrics_dir / "auc_vs_best_pvalues.csv", index=False)
        if mcnemar_vs_ref_df is not None and not mcnemar_vs_ref_df.empty:
            mcnemar_vs_ref_df.to_csv(self.metrics_dir / "mcnemar_vs_best_pvalues.csv", index=False)
        logger.info("Metrics saved (DeLong + McNemar p-values vs reference model)")

    def _save_threshold_comparison(self, results: dict) -> None:
        rows = []
        for name, res in results.items():
            rows.append({
                "model": name,
                "threshold_train_youden_mean": res.get("threshold_train_youden_mean", float("nan")),
                "threshold_eval_youden": res.get("threshold_eval_youden", float("nan")),
                "threshold_fixed": 0.5,
                "accuracy_train_youden": res.get("accuracy", float("nan")),
                "f1_train_youden": res.get("f1", float("nan")),
                "sensitivity_train_youden": res.get("sensitivity", float("nan")),
                "specificity_train_youden": res.get("specificity", float("nan")),
                "accuracy_at_0.5": res.get("accuracy_at_0.5", float("nan")),
                "f1_at_0.5": res.get("f1_at_0.5", float("nan")),
                "sensitivity_at_0.5": res.get("sensitivity_at_0.5", float("nan")),
                "specificity_at_0.5": res.get("specificity_at_0.5", float("nan")),
                "accuracy_eval_youden": res.get("accuracy_eval_youden", float("nan")),
                "f1_eval_youden": res.get("f1_eval_youden", float("nan")),
                "sensitivity_eval_youden": res.get("sensitivity_eval_youden", float("nan")),
                "specificity_eval_youden": res.get("specificity_eval_youden", float("nan")),
                "delta_accuracy_eval_minus_train": res.get(
                    "delta_accuracy_eval_minus_train", float("nan")
                ),
                "delta_f1_eval_minus_train": res.get("delta_f1_eval_minus_train", float("nan")),
            })
        if rows:
            pd.DataFrame(rows).to_csv(
                self.metrics_dir / "metrics_threshold_comparison.csv",
                index=False,
            )
            logger.info("Threshold comparison saved -> metrics_threshold_comparison.csv")

    def _save_cohort_metrics(self, rows: list[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows).sort_values(["model", "cohort"])
        df.to_csv(self.metrics_dir / "metrics_by_cohort.csv", index=False)
        logger.info("Per-cohort metrics saved")

    def _save(self, fig, name):
        for ext in [self.fmt, "png"]:
            fig.savefig(self.fig_models / f"{name}.{ext}", dpi=self.dpi)
        plt.close(fig)

    def _save_shap(self, fig, name):
        for ext in [self.fmt, "png"]:
            fig.savefig(self.fig_shap / f"{name}.{ext}", dpi=self.dpi)
        plt.close(fig)