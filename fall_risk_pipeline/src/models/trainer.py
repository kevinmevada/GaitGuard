from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
console = Console()

# Columns that must never enter the feature matrix.
# fall_probability  — a hand-coded cohort lookup; directly encodes the label.
# laterality_biased — True only for HipOA / CVA (both high-risk); encodes cohort identity.
# n_trials          — administrative count, not a gait measurement.
NON_FEATURE_COLS = {
    "participant_id",
    "cohort",
    "risk_label",
    "fall_probability",
    "laterality_biased",
    "n_trials",
}


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
        self.cv_jobs = -1
        self.random_state = config["models"]["evaluation"]["random_state"]

    def run(self):
        X, y, groups = self._load_data()
        logger.info(f"Data: {X.shape}, Labels: {np.bincount(y)}")

        models_to_run = [
            model_name
            for model_name in self.config["models"]["run"]
            if model_name not in ("cnn_1d", "lstm")
        ]
        results = {}

        total_steps = len(models_to_run) + int(self.config["models"]["ensemble"]["enabled"])
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

                best_params, cv_mean, cv_std = self._nested_cv(model_name, X, y, groups)

                best_pipeline = self._build_pipeline_from_params(model_name, best_params)
                best_pipeline.fit(X, y)

                results[model_name] = {
                    "pipeline": best_pipeline,
                    "cv_auc": cv_mean,
                    "cv_std": cv_std,
                    "params": best_params,
                }

                logger.info(f"{model_name} AUC = {cv_mean:.4f} ± {cv_std:.4f}")
                self._save_model(model_name, best_pipeline)
                progress.advance(task_id)

            if self.config["models"]["ensemble"]["enabled"]:
                progress.update(task_id, description="Training model: ensemble")
                top_k = self.config["models"]["ensemble"]["top_k"]
                sorted_models = sorted(results.items(), key=lambda item: item[1]["cv_auc"], reverse=True)
                top_models = sorted_models[:top_k]

                ensemble = self._build_ensemble(top_models)

                cv = StratifiedGroupKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
                scores = cross_val_score(
                    ensemble,
                    X,
                    y,
                    cv=cv,
                    scoring="roc_auc",
                    groups=groups,
                )

                auc_mean = float(np.mean(scores))
                auc_std = float(np.std(scores))

                ensemble.fit(X, y)

                results["ensemble"] = {
                    "pipeline": ensemble,
                    "cv_auc": auc_mean,
                    "cv_std": auc_std,
                    "params": {"top_models": [name for name, _ in top_models]},
                }

                logger.info(f"ensemble AUC = {auc_mean:.4f} ± {auc_std:.4f}")
                self._save_model("ensemble", ensemble)
                progress.advance(task_id)

            progress.update(task_id, description="Training complete")

        self._save_comparison(results)
        return results

    def _load_data(self):
        path = self.feat_dir / "patient_features.parquet"
        df = pd.read_parquet(path)

        # Exclude all non-feature columns — including leakage columns.
        feat_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        feat_cols = df[feat_cols].select_dtypes(include=np.number).columns.tolist()

        X = df[feat_cols].values.astype(np.float32)
        y = df["risk_label"].values.astype(int)
        groups = df["participant_id"].values
        return X, y, groups

    def _nested_cv(self, name: str, X, y, groups) -> tuple[dict, float, float]:
        """Nested CV: outer loop estimates generalisation, inner loop tunes.
        Returns (best_params_from_last_inner_fold, outer_mean_auc, outer_std_auc).
        """
        outer_cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        outer_scores: list[float] = []
        last_best_params: dict = {}

        for train_idx, val_idx in outer_cv.split(X, y, groups):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]

            best_params, _ = self._run_optuna(
                name, X_train, y_train, groups_train,
                n_trials=self.n_trials, timeout=self.timeout,
            )
            last_best_params = best_params

            pipeline = self._build_pipeline_from_params(name, best_params)
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_val)[:, 1]

            from sklearn.metrics import roc_auc_score
            outer_scores.append(roc_auc_score(y_val, proba))

        return last_best_params, float(np.mean(outer_scores)), float(np.std(outer_scores))

    def _run_optuna(self, name: str, X, y, groups, n_trials: int, timeout: int) -> tuple[dict, float]:
        """Inner Optuna tuning loop confined to the provided (X, y, groups) split."""
        cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        def objective(trial):
            pipeline = self._build_pipeline(name, trial)
            scores = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring="roc_auc",
                groups=groups,
                n_jobs=self.cv_jobs,
            )
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        return study.best_params, study.best_value

    def _build_pipeline(self, name, trial):
        clf = self._sample_classifier(name, trial)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def _build_pipeline_from_params(self, name, params):
        clf = self._build_classifier_from_params(name, params)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def _sample_classifier(self, name, trial):
        if name == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 2, 6),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                subsample=trial.suggest_float("subsample", 0.5, 0.9),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 0.9),
                reg_alpha=trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True),
                random_state=self.random_state,
                n_jobs=1,
                eval_metric="logloss",
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
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
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
            )

        raise ValueError(f"Unknown model {name}")

    def _build_classifier_from_params(self, name, params):
        if name == "xgboost":
            return xgb.XGBClassifier(**params, random_state=self.random_state, n_jobs=1, eval_metric="logloss")
        if name == "lightgbm":
            return lgb.LGBMClassifier(**params, random_state=self.random_state, n_jobs=1, verbose=-1)
        if name == "random_forest":
            return RandomForestClassifier(**params, random_state=self.random_state, n_jobs=1, class_weight="balanced")
        if name == "svm":
            return SVC(**params, probability=True, kernel="rbf", class_weight="balanced")
        if name == "mlp":
            return MLPClassifier(**params, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, max_iter=500)
        raise ValueError(name)

    def _build_ensemble(self, top_models: list) -> VotingClassifier:
        """Build a VotingClassifier from full pipelines (imputer + scaler + clf)."""
        estimators = [
            (name, res["pipeline"])
            for name, res in top_models
        ]
        return VotingClassifier(estimators=estimators, voting="soft")

    def _save_model(self, name, pipeline):
        path = self.ckpt_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)

    def _save_comparison(self, results):
        rows = [
            {
                "model": key,
                "cv_auc": value["cv_auc"],
                "cv_std": value.get("cv_std", float("nan")),
                "validation": "nested_stratified_group_kfold",
            }
            for key, value in results.items()
        ]

        df = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
        path = self.metrics_dir / "model_comparison_cv.csv"
        df.to_csv(path, index=False)
        logger.info("\n" + df.to_string(index=False))