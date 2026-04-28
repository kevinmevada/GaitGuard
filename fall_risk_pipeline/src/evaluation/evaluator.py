"""
Publication-oriented evaluation using nested subject-grouped validation.

FIXES APPLIED:
  1. Removed fall_probability / laterality_biased / n_trials from feature matrix.
  2. Fast mode now re-fits a checkpoint on train data per fold — no longer evaluates
     the full-data checkpoint in-sample (which inflated metrics).
  3. Replaced non-existent _tune_model_with_budget() call with _run_optuna() +
     _build_pipeline_from_params() — the correct public trainer API.
  4. SHAP is now computed on a held-out subset using the fold-refitted pipeline,
     not the full-data checkpoint.
"""

from __future__ import annotations

import pickle
from pathlib import Path

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

from src.models.trainer import ModelTrainer, NON_FEATURE_COLS


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
        X, y, groups, feat_names = self._load_data()

        model_names = [
            name for name in self.config["models"]["run"]
            if name not in ("cnn_1d", "lstm")
        ]
        all_results: dict[str, dict] = {}

        for name in tqdm(
            model_names,
            desc="Evaluating models",
            colour="red",
            bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
        ):
            if self.fast:
                result = self._fast_evaluate_model(name, X, y, groups)
            else:
                result = self._nested_group_evaluate_model(name, X, y, groups)

            all_results[name] = result
            logger.info(
                f"{name:20s}  "
                f"AUC={result['auc']:.4f}  "
                f"Acc={result['accuracy']:.4f}  "
                f"F1={result['f1']:.4f}"
            )

        if self.config["models"]["ensemble"]["enabled"]:
            if self.fast:
                ensemble_result = self._fast_evaluate_ensemble(X, y, groups, model_names)
            else:
                ensemble_result = self._nested_group_evaluate_ensemble(X, y, groups)
            all_results["ensemble"] = ensemble_result
            logger.info(
                f"{'ensemble':20s}  "
                f"AUC={ensemble_result['auc']:.4f}  "
                f"Acc={ensemble_result['accuracy']:.4f}  "
                f"F1={ensemble_result['f1']:.4f}"
            )

        self._plot_roc_all(all_results)
        self._plot_pr_curves(all_results)
        self._plot_calibration(all_results)
        self._plot_confusion_matrices(all_results)

        best_name = max(all_results, key=lambda n: all_results[n]["auc"])
        logger.info(f"Best model (AUC): {best_name}")

        # SHAP on a held-out fold — not the full-data checkpoint.
        self._shap_analysis_held_out(best_name, X, y, groups, feat_names)

        self._save_all_metrics(all_results, len(np.unique(groups)))
        return all_results

    # ------------------------------------------------------------------
    # FAST MODE — per-fold refit of saved checkpoint architecture
    # ------------------------------------------------------------------

    def _fast_evaluate_model(self, name: str, X, y, groups):
        """
        Load a saved checkpoint to recover its hyperparameters, then re-fit it
        on each LOSO training fold and evaluate on the held-out subject.
        This avoids in-sample evaluation while skipping Optuna retuning.
        """
        checkpoint = self._load_checkpoint(name)
        if checkpoint is None:
            logger.warning(f"Checkpoint not found for {name} — falling back to nested CV")
            return self._nested_group_evaluate_model(name, X, y, groups)

        unique_subjects = np.unique(groups)
        all_probs, all_true = [], []

        for subj in unique_subjects:
            test_idx  = np.where(groups == subj)[0]
            train_idx = np.where(groups != subj)[0]

            if len(np.unique(y[train_idx])) < 2:
                continue

            # Clone the pipeline structure and refit on training fold only.
            import sklearn.base as skbase
            fold_model = skbase.clone(checkpoint)
            fold_model.fit(X[train_idx], y[train_idx])

            prob = fold_model.predict_proba(X[test_idx])[:, 1]
            all_probs.extend(prob.tolist())
            all_true.extend(y[test_idx].tolist())

        logger.info(f"{name} — fast LOSO done ({len(unique_subjects)} subjects)")
        return self._build_metric_payload(name, np.array(all_true), np.array(all_probs))

    def _fast_evaluate_ensemble(self, X, y, groups, model_names: list[str]):
        """
        Soft-vote ensemble: per-fold refit of each checkpoint, then average probabilities.
        """
        import sklearn.base as skbase

        checkpoints = {}
        for name in model_names:
            m = self._load_checkpoint(name)
            if m is not None:
                checkpoints[name] = m

        if not checkpoints:
            logger.warning("No checkpoints found — falling back to nested ensemble CV")
            return self._nested_group_evaluate_ensemble(X, y, groups)

        unique_subjects = np.unique(groups)
        all_probs, all_true = [], []

        for subj in unique_subjects:
            test_idx  = np.where(groups == subj)[0]
            train_idx = np.where(groups != subj)[0]

            if len(np.unique(y[train_idx])) < 2:
                continue

            fold_probs_list = []
            for name, ckpt in checkpoints.items():
                fold_model = skbase.clone(ckpt)
                fold_model.fit(X[train_idx], y[train_idx])
                fold_probs_list.append(fold_model.predict_proba(X[test_idx])[:, 1])

            fold_probs = np.mean(fold_probs_list, axis=0)
            all_probs.extend(fold_probs.tolist())
            all_true.extend(y[test_idx].tolist())

        logger.info(f"Ensemble fast LOSO done ({len(unique_subjects)} subjects, {len(checkpoints)} models)")
        return self._build_metric_payload("ensemble", np.array(all_true), np.array(all_probs))

    # ------------------------------------------------------------------
    # FULL NESTED CV MODE — parallelized LOSO with Optuna retuning
    # Use only for final publication-level reporting
    # ------------------------------------------------------------------

    def _evaluate_one_subject(self, subj, name, X, y, groups):
        """Single LOSO fold — called in parallel."""
        test_idx  = np.where(groups == subj)[0]
        train_idx = np.where(groups != subj)[0]

        if len(np.unique(y[train_idx])) < 2:
            return None, None

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
        best_pipeline = self.trainer._build_pipeline_from_params(name, best_params)
        best_pipeline.fit(X[train_idx], y[train_idx])

        prob = best_pipeline.predict_proba(X[test_idx])[:, 1]
        return prob.tolist(), y[test_idx].tolist()

    def _nested_group_evaluate_model(self, name, X, y, groups):
        """Parallelized nested LOSO-CV with per-fold Optuna retuning."""
        unique_subjects = np.unique(groups)

        fold_results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._evaluate_one_subject)(subj, name, X, y, groups)
            for subj in tqdm(
                unique_subjects,
                desc=f"  {name} folds",
                leave=False,
                colour="red",
                bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
            )
        )

        all_probs, all_true = [], []
        for probs, trues in fold_results:
            if probs is not None:
                all_probs.extend(probs)
            if trues is not None:
                all_true.extend(trues)

        return self._build_metric_payload(name, np.array(all_true), np.array(all_probs))

    def _evaluate_ensemble_one_subject(self, subj, model_names, X, y, groups):
        """Single ensemble LOSO fold — called in parallel."""
        test_idx  = np.where(groups == subj)[0]
        train_idx = np.where(groups != subj)[0]

        if len(np.unique(y[train_idx])) < 2:
            return None, None

        top_k = self.config["models"]["ensemble"]["top_k"]
        tuned_results = []

        for name in model_names:
            # FIX: use the correct trainer API instead of _tune_model_with_budget.
            best_params, best_score = self.trainer._run_optuna(
                name,
                X[train_idx],
                y[train_idx],
                groups[train_idx],
                n_trials=self.nested_trials,
                timeout=self.nested_timeout,
            )
            best_pipeline = self.trainer._build_pipeline_from_params(name, best_params)
            best_pipeline.fit(X[train_idx], y[train_idx])
            tuned_results.append((name, {"pipeline": best_pipeline, "cv_auc": best_score}))

        top_models = sorted(
            tuned_results,
            key=lambda item: item[1]["cv_auc"],
            reverse=True,
        )[:top_k]

        ensemble = self.trainer._build_ensemble(top_models)
        ensemble.fit(X[train_idx], y[train_idx])

        prob = ensemble.predict_proba(X[test_idx])[:, 1]
        return prob.tolist(), y[test_idx].tolist()

    def _nested_group_evaluate_ensemble(self, X, y, groups):
        """Parallelized nested LOSO-CV for the ensemble."""
        model_names = [
            name for name in self.config["models"]["run"]
            if name not in ("cnn_1d", "lstm")
        ]
        unique_subjects = np.unique(groups)

        fold_results = Parallel(n_jobs=-1, prefer="processes")(
            delayed(self._evaluate_ensemble_one_subject)(subj, model_names, X, y, groups)
            for subj in tqdm(
                unique_subjects,
                desc="  ensemble folds",
                leave=False,
                colour="red",
                bar_format="\033[31m{l_bar}{bar}{r_bar}\033[0m",
            )
        )

        all_probs, all_true = [], []
        for probs, trues in fold_results:
            if probs is not None:
                all_probs.extend(probs)
            if trues is not None:
                all_true.extend(trues)

        return self._build_metric_payload("ensemble", np.array(all_true), np.array(all_probs))

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _load_data(self):
        path = self.feat_dir / "patient_features.parquet"
        df = pd.read_parquet(path)

        # FIX: exclude all leakage columns — same set as trainer uses.
        feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        feat_cols = df[feat_cols].select_dtypes(include=np.number).columns.tolist()

        X      = df[feat_cols].values.astype(np.float32)
        y      = df["risk_label"].values.astype(int)
        groups = df["participant_id"].values

        return X, y, groups, feat_cols

    def _load_checkpoint(self, model_name: str):
        path = self.ckpt_dir / f"{model_name}.pkl"
        if not path.exists():
            return None
        with open(path, "rb") as f:
            return pickle.load(f)

    def _optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden = tpr - fpr
        idx = int(np.argmax(youden))
        threshold = float(thresholds[idx])
        if not np.isfinite(threshold):
            return 0.5
        return max(0.0, min(1.0, threshold))

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

    def _build_metric_payload(self, name: str, y_true: np.ndarray, y_prob: np.ndarray):
        if len(y_true) == 0:
            raise ValueError(f"{name} evaluation failed: no valid grouped splits")

        threshold = self._optimal_threshold(y_true, y_prob)
        y_pred      = (y_prob >= threshold).astype(int)
        auc_roc     = roc_auc_score(y_true, y_prob)
        auc_pr      = average_precision_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_ci_low, auc_ci_high = self._bootstrap_auc_ci(y_true, y_prob)

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        sensitivity = tp / (tp + fn + 1e-10)
        specificity = tn / (tn + fp + 1e-10)

        return {
            "model":            name,
            "auc":              float(auc_roc),
            "auc_pr":           float(auc_pr),
            "auc_ci_low":       auc_ci_low,
            "auc_ci_high":      auc_ci_high,
            "decision_threshold": threshold,
            "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy":         float(accuracy_score(y_true, y_pred)),
            "sensitivity":      float(sensitivity),
            "specificity":      float(specificity),
            "fpr":              fpr,
            "tpr":              tpr,
            "y_true":           y_true,
            "y_prob":           y_prob,
            "confusion_matrix": cm,
            "report":           classification_report(y_true, y_pred, output_dict=True),
        }

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

    def _plot_confusion_matrices(self, results):
        for name, res in results.items():
            fig, ax = plt.subplots()
            cm = res["confusion_matrix"]
            ax.imshow(cm, cmap="Blues")
            ax.set_title(name)
            self._save(fig, f"cm_{name}")

    def _shap_analysis_held_out(self, name: str, X, y, groups, feat_names):
        """
        FIX: compute SHAP on a genuine held-out fold, not the full-data checkpoint.
        We take one LOSO fold (the last subject), refit on the remaining data,
        then explain predictions on the held-out subject's data.
        """
        if not self.config["explainability"]["shap_enabled"]:
            return

        unique_subjects = np.unique(groups)
        # Use last subject as the held-out shard for SHAP explanation.
        subj = unique_subjects[-1]
        test_idx  = np.where(groups == subj)[0]
        train_idx = np.where(groups != subj)[0]

        checkpoint = self._load_checkpoint(name)
        if checkpoint is None:
            logger.warning(f"Skipping SHAP — checkpoint not found for {name}")
            return

        import sklearn.base as skbase
        fold_model = skbase.clone(checkpoint)
        fold_model.fit(X[train_idx], y[train_idx])

        n = min(self.config["explainability"]["n_shap_samples"], len(test_idx))
        idx      = test_idx[:n]
        X_sample = X[idx]

        try:
            if hasattr(fold_model, "named_steps") and "clf" in fold_model.named_steps:
                clf    = fold_model.named_steps["clf"]
                X_proc = fold_model[:-1].transform(X_sample)
            else:
                clf    = fold_model
                X_proc = X_sample

            if name in ("xgboost", "lightgbm", "random_forest"):
                explainer = shap.TreeExplainer(clf)
                shap_vals = explainer.shap_values(X_proc)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]
            else:
                explainer = shap.KernelExplainer(
                    lambda x: clf.predict_proba(x)[:, 1],
                    shap.sample(X_proc, min(50, len(X_proc))),
                )
                shap_vals = explainer.shap_values(X_proc, nsamples=50)

        except Exception as exc:
            logger.warning(f"SHAP failed: {exc}")
            return

        shap.summary_plot(shap_vals, X_proc, feature_names=feat_names, show=False)
        fig = plt.gcf()
        self._save_shap(fig, f"shap_{name}")

    def _save_all_metrics(self, results, n_participants: int):
        rows = []
        for name, res in results.items():
            rows.append({
                "model":               name,
                "auc":                 res["auc"],
                "auc_ci_low":          res.get("auc_ci_low", float("nan")),
                "auc_ci_high":         res.get("auc_ci_high", float("nan")),
                "accuracy":            res["accuracy"],
                "f1":                  res["f1"],
                "sensitivity":         res.get("sensitivity", 0.0),
                "specificity":         res.get("specificity", 0.0),
                "decision_threshold":  res.get("decision_threshold", 0.5),
                "evaluation_mode":     "fast" if self.fast else "full_nested",
                "validation_strategy": self.validation_strategy,
                "participants":        n_participants,
            })
        df = pd.DataFrame(rows).sort_values("auc", ascending=False)
        df.to_csv(self.metrics_dir / "metrics.csv", index=False)
        logger.info("Metrics saved")

    def _save(self, fig, name):
        for ext in [self.fmt, "png"]:
            fig.savefig(self.fig_models / f"{name}.{ext}", dpi=self.dpi)
        plt.close(fig)

    def _save_shap(self, fig, name):
        for ext in [self.fmt, "png"]:
            fig.savefig(self.fig_shap / f"{name}.{ext}", dpi=self.dpi)
        plt.close(fig)