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
        self.cv_jobs = -1  # Use all CPU cores for cross-validation
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
                best_pipeline, best_score, best_params = self._tune_model(model_name, X, y, groups)

                results[model_name] = {
                    "pipeline": best_pipeline,
                    "cv_auc": best_score,
                    "params": best_params,
                }

                logger.info(f"{model_name} AUC = {best_score:.4f}")
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

                auc = float(np.mean(scores))
                ensemble.fit(X, y)
                results["ensemble"] = {
                    "pipeline": ensemble,
                    "cv_auc": auc,
                    "params": {"top_models": [name for name, _ in top_models]},
                }

                logger.info(f"ensemble AUC = {auc:.4f}")
                self._save_model("ensemble", ensemble)
                progress.advance(task_id)

            progress.update(task_id, description="Training complete")

        self._save_comparison(results)
        return results

    def _load_data(self):
        path = self.feat_dir / "patient_features.parquet"
        df = pd.read_parquet(path)

        meta_cols = ["participant_id", "cohort", "risk_label"]
        feat_cols = [c for c in df.columns if c not in meta_cols]
        feat_cols = df[feat_cols].select_dtypes(include=np.number).columns.tolist()

        X = df[feat_cols].values.astype(np.float32)
        y = df["risk_label"].values.astype(int)
        groups = df["participant_id"].values
        return X, y, groups

    def _tune_model(self, name, X, y, groups):
        return self._tune_model_with_budget(
            name,
            X,
            y,
            groups,
            n_trials=self.n_trials,
            timeout=self.timeout,
        )

    def _tune_model_with_budget(self, name, X, y, groups, n_trials, timeout):
        cv = StratifiedGroupKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

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

        best_params = study.best_params
        best_score = study.best_value

        best_pipeline = self._build_pipeline_from_params(name, best_params)
        best_pipeline.fit(X, y)

        return best_pipeline, best_score, best_params

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
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=self.random_state,
                n_jobs=1,
                eval_metric="logloss",
            )

        if name == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 20, 150),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=self.random_state,
                n_jobs=1,
                verbose=-1,
            )

        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 200, 800),
                max_depth=trial.suggest_int("max_depth", 5, 30),
                random_state=self.random_state,
                n_jobs=1,
                class_weight="balanced",
            )

        if name == "svm":
            return SVC(
                C=trial.suggest_float("C", 0.1, 100, log=True),
                gamma="scale",
                probability=True,
                class_weight="balanced",
            )

        if name == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(trial.suggest_int("units", 64, 256),),
                max_iter=500,
            )

        raise ValueError(f"Unknown model {name}")

    def _build_classifier_from_params(self, name, params):
        if name == "xgboost":
            return xgb.XGBClassifier(**params, random_state=self.random_state, n_jobs=1, eval_metric="logloss")
        if name == "lightgbm":
            return lgb.LGBMClassifier(**params, random_state=self.random_state, n_jobs=1, verbose=-1)
        if name == "random_forest":
            return RandomForestClassifier(**params, random_state=self.random_state, n_jobs=1)
        if name == "svm":
            return SVC(**params, probability=True)
        if name == "mlp":
            return MLPClassifier(**params)
        raise ValueError(name)

    def _build_ensemble(self, top_models):
        estimators = [
            (name, res["pipeline"].named_steps["clf"])
            for name, res in top_models
        ]

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", VotingClassifier(estimators=estimators, voting="soft")),
        ])

    def _save_model(self, name, pipeline):
        path = self.ckpt_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(pipeline, f)

    def _save_comparison(self, results):
        rows = [
            {"model": key, "cv_auc": value["cv_auc"], "validation": "stratified_group_kfold"}
            for key, value in results.items()
        ]

        df = pd.DataFrame(rows).sort_values("cv_auc", ascending=False)
        path = self.metrics_dir / "model_comparison_cv.csv"
        df.to_csv(path, index=False)
        logger.info("\n" + df.to_string(index=False))
