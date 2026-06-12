from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any
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

from src.dataset.label_balance import balanced_scale_pos_weight
from src.dataset.label_policy import is_binary_task, label_mode_description
from src.evaluation.roc_auc_scoring import roc_auc_from_proba, roc_auc_scoring_name
from src.features.feature_matrix import NON_FEATURE_COLS, load_patient_feature_matrix
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

    def run(self):
        """Train base models and ensembles.

        Two-pass tuning per base model (intentional):
          1. ``_nested_cv`` — outer StratifiedGroupKFold estimates generalisation
             (``cv_auc`` / ``cv_std`` in model_comparison_cv.csv). Each outer fold
             runs its own Optuna search on the outer-train split only.
          2. ``_run_optuna`` on full data — selects hyperparameters for the saved
             checkpoint (``deployed_params``). Its inner-CV score is ``tuning_cv_auc``.

        ``cv_auc`` therefore describes nested-CV performance, not the exact params
        in the pickle; use ``tuning_cv_auc`` as a proxy for the deployed model's
        in-sample tune score, or hold-out evaluation for deployment performance.
        """
        X, y, groups, _, patient_df = self._load_data()
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

                cv_mean, cv_std = self._nested_cv(model_name, X, y, groups)
                # Second pass: tune on all data so the checkpoint uses full-sample
                # hyperparameters (may differ from any single nested-CV outer fold).
                best_params, tuning_cv_auc = self._run_optuna(
                    model_name, X, y, groups, n_trials=self.n_trials, timeout=self.timeout
                )
                best_pipeline = self._build_pipeline_from_params(model_name, best_params, y)
                best_pipeline.fit(X, y)

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
                cv = StratifiedGroupKFold(
                    n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
                )

                for method in ensemble_methods:
                    model_key = ensemble_model_name(method)
                    progress.update(
                        task_id, description=f"Training model: {model_key}"
                    )
                    ensemble = build_ensemble_estimator(
                        top_models,
                        method,
                        cv_folds=self.cv_folds,
                        random_state=self.random_state,
                    )
                    fit_params = (
                        {"groups": groups} if method == "stacking" else None
                    )
                    _cv_fit_params = fit_params or {}
                    scores = cross_val_score(
                        ensemble,
                        X,
                        y,
                        cv=cv,
                        scoring=roc_auc_scoring_name(y, self.config),
                        groups=groups,
                        fit_params=_cv_fit_params,
                    )
                    auc_mean = float(np.mean(scores))
                    auc_std = float(np.std(scores))

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
                        },
                    }
                    logger.info(
                        f"{model_key} ({method}) AUC = {auc_mean:.4f} ± {auc_std:.4f}"
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

    def _nested_cv(self, name: str, X, y, groups) -> tuple[float, float]:
        """Nested CV: outer loop estimates generalisation, inner loop tunes.

        Hyperparameters from each outer fold are discarded; ``run()`` performs a
        separate full-data Optuna pass for the deployable checkpoint.  Returns
        (outer_mean_auc, outer_std_auc) for unbiased model comparison only.
        """
        outer_cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        outer_scores: list[float] = []

        for train_idx, val_idx in outer_cv.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]

            best_params, _ = self._run_optuna(
                name, X_train, y_train, groups_train,
                n_trials=self.n_trials, timeout=self.timeout,
            )

            pipeline = self._build_pipeline_from_params(name, best_params, y_train)
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_val)
            outer_scores.append(self._roc_auc_from_proba(y_val, proba))

        return float(np.mean(outer_scores)), float(np.std(outer_scores))

    def _run_optuna(self, name: str, X, y, groups, n_trials: int, timeout: int) -> tuple[dict, float]:
        """Inner Optuna tuning loop confined to the provided (X, y, groups) split."""
        cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        # Safe with evaluator Parallel(prefer="processes"): each worker gets its own Trainer copy.
        self._fit_y = np.asarray(y).astype(int)

        scoring = roc_auc_scoring_name(y, self.config)

        def objective(trial):
            pipeline = self._build_pipeline(name, trial)
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                groups=groups,
                n_jobs=self.cv_jobs,
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
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
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
                if k not in ("scale_pos_weight", "objective", "num_class", "eval_metric")
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
            return MLPClassifier(**mlp_params, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, max_iter=500)
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
                "cv_auc_source": (
                    "ensemble_cv"
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