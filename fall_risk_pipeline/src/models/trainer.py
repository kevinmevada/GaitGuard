from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.validation")
warnings.filterwarnings("ignore", module="lightgbm")
console = Console()

from src.features.feature_missingness import warn_high_missingness_features
from src.dataset.label_policy import is_binary_task, label_mode_description
from src.evaluation.roc_auc_scoring import roc_auc_from_proba, roc_auc_scoring_name
from src.features.feature_matrix import (
    NON_FEATURE_COLS,
    load_patient_feature_matrix,
    nested_rfecv_column_indices,
)
from src.models.ensemble_builder import (
    build_ensemble_estimator,
    ensemble_model_name,
    primary_ensemble_checkpoint_name,
    resolve_ensemble_methods,
)
from src.utils.checkpoint_io import refresh_manifest, save_checkpoint


class ModelTrainer:

    def __init__(self, config: dict):
        self.config = config

        self.feat_dir = Path(config["paths"]["features"])
        self.ckpt_dir = Path(config["paths"]["checkpoints"])
        self.metrics_dir = Path(config["paths"]["metrics"])

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        tcfg = config["models"]["tuning"]
        self.n_trials = tcfg["n_trials"]
        self.timeout = tcfg["timeout_per_model"]
        self.cv_folds = tcfg["cv_folds"]
        self.cv_jobs = 1  # avoid fork-bomb inside nested Optuna parallel
        self.random_state = config["models"]["evaluation"]["random_state"]
        self._fit_y: np.ndarray | None = None

    @staticmethod
    def _xgb_sample_weights(y: np.ndarray, config: dict) -> np.ndarray | None:
        """Sklearn-style balanced per-sample weights for multiclass XGBoost."""
        y = np.asarray(y).astype(int)
        n_classes = len(np.unique(y))
        if n_classes <= 2 and is_binary_task(y, config):
            return None
        counts = np.bincount(y, minlength=n_classes).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        class_weights = len(y) / (n_classes * counts)
        return class_weights[y]

    @staticmethod
    def _balanced_sample_weights(y: np.ndarray) -> np.ndarray:
        """Inverse-frequency per-sample weights (sklearn 'balanced' style)."""
        y = np.asarray(y).astype(int)
        counts = np.bincount(y).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        class_weights = len(y) / (len(counts) * counts)
        return class_weights[y]

    def _pipeline_fit_params(self, name: str, y: np.ndarray) -> dict[str, Any]:
        if name == "mlp":
            return {"clf__sample_weight": self._balanced_sample_weights(y)}
        if name != "xgboost":
            return {}
        weights = self._xgb_sample_weights(y, self.config)
        if weights is None:
            return {}
        return {"clf__sample_weight": weights}

    def fit_pipeline(self, name: str, pipeline: Pipeline, X, y, **extra: Any) -> Pipeline:
        """Fit a sklearn pipeline; inject XGBoost multiclass sample weights when needed."""
        feat_names = extra.pop("feat_names", None)
        if isinstance(feat_names, str):
            feat_names = None
        warn_high_missingness_features(
            np.asarray(X),
            list(feat_names) if feat_names else None,
            context=f"{name} training",
        )
        fit_params = {**self._pipeline_fit_params(name, y), **extra}
        if fit_params:
            pipeline.fit(X, y, **fit_params)
        else:
            pipeline.fit(X, y)
        return pipeline

    def run(self):
        """Train base models and ensembles.

        Two-pass tuning per base model (intentional):
          1. ``_nested_cv`` — outer StratifiedGroupKFold with **per-outer-fold RFECV**
             on the full feature matrix (unbiased ``cv_auc`` in model_comparison_cv.csv).
          2. ``_run_optuna`` on globally selected deployment features — hyperparameters
             for saved checkpoints match the API/production schema.

        ``cv_auc`` therefore describes nested-CV performance, not the exact params
        in the pickle; use ``tuning_cv_auc`` as a proxy for the deployed model's
        in-sample tune score, or ``deploy_loso_gap.csv`` for deploy-schema LOSO (ML-032).
        """
        X, y, groups, feat_cols_deploy, patient_df = self._load_data()
        X_full, _, _, feat_cols_full, _ = load_patient_feature_matrix(
            self.config, use_selected=False
        )
        counts = dict(zip(*np.unique(y, return_counts=True)))
        logger.info(
            f"Data: {X.shape} | {label_mode_description(self.config)} | "
            f"label counts: {counts}"
        )
        if is_binary_task(y, self.config):
            spw = balanced_scale_pos_weight(y)
            logger.info(f"XGBoost scale_pos_weight (balanced)={spw:.3f}")

        models_to_run = [
            model_name
            for model_name in self.config["models"]["run"]
            if model_name not in ("cnn_1d", "lstm")
        ]
        results = {}

        ensemble_methods = resolve_ensemble_methods(self.config)
        total_steps = len(models_to_run) + len(ensemble_methods)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold magenta]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} tasks"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Preparing training...", total=total_steps)

            for model_name in models_to_run:
                progress.update(task_id, description=f"Training model: {model_name}")
                logger.info(f"Tuning {model_name}...")

                cv_mean, cv_std = self._nested_cv(
                    model_name, X_full, y, groups, feat_cols_full
                )
                # Tune/deploy on globally selected features (deployment schema).
                best_params, tuning_cv_auc = self._run_optuna(
                    model_name, X, y, groups, n_trials=self.n_trials, timeout=self.timeout
                )
                best_pipeline = self._build_pipeline_from_params(model_name, best_params, y)
                self.fit_pipeline(model_name, best_pipeline, X, y)

                results[model_name] = {
                    "pipeline": best_pipeline,
                    "cv_auc": cv_mean,
                    "cv_std": cv_std,
                    "tuning_cv_auc": tuning_cv_auc,
                    "params": best_params,
                }

                logger.info(
                    f"{model_name} nested CV AUC = {cv_mean:.4f} ± {cv_std:.4f} | "
                    f"deployed tune AUC = {tuning_cv_auc:.4f}"
                )
                self._save_model(model_name, best_pipeline)
                progress.advance(task_id)

            if ensemble_methods:
                ens_cfg = self.config["models"]["ensemble"]
                top_k = ens_cfg["top_k"]
                sorted_models = sorted(
                    results.items(), key=lambda item: item[1]["cv_auc"], reverse=True
                )
                top_models = sorted_models[:top_k]

                for method in ensemble_methods:
                    model_key = ensemble_model_name(method)
                    progress.update(
                        task_id, description=f"Training model: {model_key}"
                    )
                    auc_mean, auc_std = self._nested_ensemble_cv(
                        method,
                        top_models,
                        results,
                        X_full,
                        y,
                        groups,
                        feat_cols_full,
                    )

                    # Deploy on global selected features (production schema).
                    ensemble = build_ensemble_estimator(
                        top_models,
                        method,
                        cv_folds=self.cv_folds,
                        random_state=self.random_state,
                    )
                    if method == "stacking":
                        ensemble.fit(X, y, groups=groups)
                    else:
                        ensemble.fit(X, y)

                    results[model_key] = {
                        "pipeline": ensemble,
                        "cv_auc": auc_mean,
                        "cv_std": auc_std,
                        "params": {
                            "ensemble_method": method,
                            "top_models": [name for name, _ in top_models],
                            "feature_selection_protocol": "nested_rfecv_per_outer_fold",
                        },
                    }
                    logger.info(
                        f"{model_key} ({method}) nested ensemble CV AUC = "
                        f"{auc_mean:.4f} ± {auc_std:.4f} (deploy fit uses global RFECV mask)"
                    )
                    self._save_model(model_key, ensemble)
                    if model_key == primary_ensemble_checkpoint_name(self.config):
                        self._save_model("ensemble", ensemble)
                    progress.advance(task_id)

            progress.update(task_id, description="Training complete")

        self._save_comparison(results)
        refresh_manifest(self.ckpt_dir)
        return results

    def _load_data(self):
        X, y, groups, feat_cols, df = load_patient_feature_matrix(self.config)
        logger.info(f"Feature matrix: {X.shape[1]} columns (selection applied if configured)")
        return X, y, groups, feat_cols, df

    def _nested_cv(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feat_cols: list[str],
    ) -> tuple[float, float]:
        """Nested CV with per-outer-fold RFECV on the outer-train split (ML-014)."""
        outer_cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        outer_scores: list[float] = []

        for train_idx, val_idx in outer_cv.split(X, y, groups):
            col_idx = nested_rfecv_column_indices(
                self.config, X, y, groups, feat_cols, train_idx
            )
            X_train, X_val = X[train_idx][:, col_idx], X[val_idx][:, col_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]

            best_params, _ = self._run_optuna(
                name, X_train, y_train, groups_train,
                n_trials=self.n_trials, timeout=self.timeout,
            )

            pipeline = self._build_pipeline_from_params(name, best_params, y_train)
            self.fit_pipeline(name, pipeline, X_train, y_train)
            proba = pipeline.predict_proba(X_val)
            outer_scores.append(self._roc_auc_from_proba(y_val, proba))

        return float(np.mean(outer_scores)), float(np.std(outer_scores))

    def _nested_ensemble_cv(
        self,
        method: str,
        top_models: list[tuple[str, dict]],
        results: dict[str, dict],
        X_full: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feat_cols_full: list[str],
    ) -> tuple[float, float]:
        """
        Nested CV with per-outer-fold RFECV and Optuna for ensemble scoring (ML-023/ML-039).

        Each outer fold re-tunes every base model on the outer-train split before fitting
        the ensemble — matching ``_nested_cv`` and avoiding optimistic reuse of full-data
        hyperparameters in ``model_comparison_cv.csv``.
        """
        del results  # retained for call-site compatibility; bases tuned per outer fold
        outer_cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        cv_folds = int(
            self.config.get("models", {}).get("ensemble", {}).get("stacking", {}).get(
                "cv_folds", self.cv_folds
            )
        )
        outer_scores: list[float] = []

        for train_idx, val_idx in outer_cv.split(X_full, y, groups):
            col_idx = nested_rfecv_column_indices(
                self.config, X_full, y, groups, feat_cols_full, train_idx
            )
            X_tr = X_full[train_idx][:, col_idx]
            X_val = X_full[val_idx][:, col_idx]
            y_train = y[train_idx]
            groups_train = groups[train_idx]

            fold_bases: list[tuple[str, dict]] = []
            for name, _ in top_models:
                best_params, _ = self._run_optuna(
                    name,
                    X_tr,
                    y_train,
                    groups_train,
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                )
                pipe = self._build_pipeline_from_params(name, best_params, y_train)
                self.fit_pipeline(name, pipe, X_tr, y_train)
                fold_bases.append((name, {"pipeline": pipe}))

            ensemble = build_ensemble_estimator(
                fold_bases,
                method,
                cv_folds=cv_folds,
                random_state=self.random_state,
            )
            if method == "stacking":
                ensemble.fit(X_tr, y_train, groups=groups_train)
            else:
                ensemble.fit(X_tr, y_train)
            proba = ensemble.predict_proba(X_val)
            outer_scores.append(self._roc_auc_from_proba(y[val_idx], proba))

        return float(np.mean(outer_scores)), float(np.std(outer_scores))

    def _run_optuna(self, name: str, X, y, groups, n_trials: int, timeout: int) -> tuple[dict, float]:
        """Inner Optuna tuning loop confined to the provided (X, y, groups) split."""
        warn_high_missingness_features(
            np.asarray(X), context=f"{name} optuna tuning set"
        )
        cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        # Safe with evaluator Parallel(prefer="processes"): each worker gets its own Trainer copy.
        self._fit_y = np.asarray(y).astype(int)

        scoring = roc_auc_scoring_name(y, self.config)

        def objective(trial):
            pipeline = self._build_pipeline(name, trial)
            fit_params = self._pipeline_fit_params(name, y)
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                groups=groups,
                n_jobs=self.cv_jobs,
                fit_params=fit_params if fit_params else None,
            )
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )
        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
        finally:
            self._fit_y = None
        return study.best_params, study.best_value

    def _build_pipeline(self, name, trial):
        clf = self._sample_classifier(name, trial)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def _build_pipeline_from_params(self, name, params, y: np.ndarray | None = None):
        clf = self._build_classifier_from_params(name, params, y)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def _roc_auc_from_proba(self, y_true: np.ndarray, proba: np.ndarray) -> float:
        return roc_auc_from_proba(y_true, proba, self.config)

    def _xgboost_kwargs(self, y: np.ndarray) -> dict:
        y = np.asarray(y).astype(int)
        n_classes = len(np.unique(y))
        if n_classes <= 2 and is_binary_task(y, self.config):
            return {
                "scale_pos_weight": balanced_scale_pos_weight(y),
                "eval_metric": "logloss",
            }
        return {
            "objective": "multi:softprob",
            "num_class": int(n_classes),
            "eval_metric": "mlogloss",
        }

    def _resolve_y(self, y: np.ndarray | None) -> np.ndarray:
        if y is not None:
            return np.asarray(y).astype(int)
        if self._fit_y is not None:
            return self._fit_y
        raise ValueError("y is required to set balanced class weights for XGBoost")

    def _sample_classifier(self, name, trial):
        if name == "xgboost":
            y_ref = self._resolve_y(None)
            return xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 0.9),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
                reg_alpha=trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                random_state=self.random_state,
                n_jobs=1,
                **self._xgboost_kwargs(y_ref),
            )

        if name == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 63),
                subsample=trial.suggest_float("subsample", 0.5, 0.9),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
                reg_alpha=trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=1,
                verbose=-1,
            )

        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_float("max_features", 0.3, 0.8),
                random_state=self.random_state,
                n_jobs=1,
                class_weight="balanced",
            )

        if name == "svm":
            return SVC(
                C=trial.suggest_float("C", 0.1, 10, log=True),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto", 0.001, 0.01, 0.1]),
                probability=True,
                class_weight="balanced",
                kernel="rbf",
                random_state=self.random_state,
            )

        if name == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(trial.suggest_int("units", 64, 256),),
                alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                # No early_stopping: validation_fraction is not group-aware and leaks
                # subjects when fit inside StratifiedGroupKFold / LOSO folds.
                early_stopping=False,
                max_iter=500,
                random_state=self.random_state,
            )

        raise ValueError(f"Unknown model {name}")

    def _build_classifier_from_params(
        self, name, params, y: np.ndarray | None = None
    ):
        if name == "xgboost":
            y_ref = self._resolve_y(y)
            clean = {
                k: v
                for k, v in params.items()
                if k not in ("scale_pos_weight", "objective", "num_class", "eval_metric", "_sample_weight")
            }
            return xgb.XGBClassifier(
                **clean,
                random_state=self.random_state,
                n_jobs=1,
                **self._xgboost_kwargs(y_ref),
            )
        if name == "lightgbm":
            lgb_params = dict(params)
            lgb_params.setdefault("class_weight", "balanced")
            return lgb.LGBMClassifier(
                **lgb_params,
                random_state=self.random_state,
                n_jobs=1,
                verbose=-1,
            )
        if name == "random_forest":
            return RandomForestClassifier(**params, random_state=self.random_state, n_jobs=1, class_weight="balanced")
        if name == "svm":
            return SVC(**params, probability=True, kernel="rbf", class_weight="balanced")
        if name == "mlp":
            mlp_params = dict(params)
            if "units" in mlp_params:
                mlp_params["hidden_layer_sizes"] = (mlp_params.pop("units"),)
            mlp_params.setdefault("early_stopping", False)
            mlp_params.setdefault("max_iter", 500)
            return MLPClassifier(**mlp_params, random_state=self.random_state)
        raise ValueError(name)

    def _save_model(self, name, pipeline):
        path = self.ckpt_dir / f"{name}.pkl"
        save_checkpoint(path, pipeline, manifest_dir=self.ckpt_dir)

    def _save_comparison(self, results):
        rows = [
            {
                "model": key,
                "cv_auc": value["cv_auc"],
                "cv_std": value.get("cv_std", float("nan")),
                "tuning_cv_auc": value.get("tuning_cv_auc", float("nan")),
                "deployed_params": json.dumps(value.get("params", {}), sort_keys=True),
                "validation": "nested_stratified_group_kfold",
                "feature_selection_protocol": "nested_rfecv_per_outer_fold",
                "deploy_feature_selection": "global_selected_features_json",
                "cv_auc_source": (
                    "nested_rfecv_ensemble_cv"
                    if isinstance(value.get("params"), dict)
                    and "ensemble_method" in value["params"]
                    else "nested_cv"
                ),
            }
            for key, value in results.items()
        ]

        df = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
        path = self.metrics_dir / "model_comparison_cv.csv"
        df.to_csv(path, index=False)
        params_path = self.metrics_dir / "model_deployed_params.json"
        params_path.write_text(
            json.dumps(
                {key: value.get("params", {}) for key, value in results.items()},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        logger.info("\n" + df.to_string(index=False))