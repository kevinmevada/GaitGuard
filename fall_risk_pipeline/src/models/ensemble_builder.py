"""
Ensemble construction and subject-grouped prediction for clinical ML comparison.

Methods:
  - soft_voting: mean of base-model positive-class probabilities (VotingClassifier).
  - stacking: logistic regression meta-learner on out-of-fold base probabilities
    (inner StratifiedGroupKFold on the LOSO train fold).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import sklearn.base as skbase
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold


def resolve_ensemble_methods(config: dict) -> list[str]:
    """Return ensemble methods to train/evaluate (default: soft_voting + stacking)."""
    ens = config.get("models", {}).get("ensemble", {})
    if not ens.get("enabled", False):
        return []
    if ens.get("compare_methods", True):
        raw = ens.get("methods", ["soft_voting", "stacking"])
    else:
        raw = [ens.get("method", "soft_voting")]
    methods = []
    for m in raw:
        if m not in methods:
            methods.append(m)
    return methods


def ensemble_model_name(method: str) -> str:
    if method == "soft_voting":
        return "ensemble_soft_voting"
    if method == "stacking":
        return "ensemble_stacking"
    return f"ensemble_{method}"


def primary_ensemble_checkpoint_name(config: dict) -> str:
    """Legacy API checkpoint alias (``ensemble.pkl``)."""
    method = config.get("models", {}).get("ensemble", {}).get("method", "soft_voting")
    return ensemble_model_name(method)


class GroupStackingEnsemble(BaseEstimator, ClassifierMixin):
    """Pickle-friendly stacking ensemble with group-aware meta-learner fitting."""

    _estimator_type = "classifier"

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def __init__(
        self,
        estimators: list[tuple[str, Any]],
        *,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.estimators = estimators
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.estimator_names = [name for name, _ in estimators]
        self.estimator_templates = [pipe for _, pipe in estimators]
        self.fitted_estimators_: list[Any] = []
        self.meta_learner_: LogisticRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None) -> GroupStackingEnsemble:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n = len(y)
        n_base = len(self.estimator_templates)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if groups is not None and len(np.unique(groups)) >= 2:
            cv = StratifiedGroupKFold(
                n_splits=min(self.cv_folds, len(np.unique(groups))),
                shuffle=True,
                random_state=self.random_state,
            )
            splits = list(cv.split(X, y, groups))
        else:
            from sklearn.model_selection import StratifiedKFold

            cv = StratifiedKFold(
                n_splits=min(self.cv_folds, max(2, int(np.min(np.bincount(y))))),
                shuffle=True,
                random_state=self.random_state,
            )
            splits = list(cv.split(X, y))

        if self.n_classes_ <= 2:
            meta_X = np.zeros((n, n_base), dtype=float)
            for j, template in enumerate(self.estimator_templates):
                for train_idx, val_idx in splits:
                    fold_model = skbase.clone(template)
                    fold_model.fit(X[train_idx], y[train_idx])
                    meta_X[val_idx, j] = fold_model.predict_proba(X[val_idx])[:, 1]
        else:
            meta_X = np.zeros((n, n_base * self.n_classes_), dtype=float)
            for j, template in enumerate(self.estimator_templates):
                for train_idx, val_idx in splits:
                    fold_model = skbase.clone(template)
                    fold_model.fit(X[train_idx], y[train_idx])
                    proba = fold_model.predict_proba(X[val_idx])
                    meta_X[val_idx, j * self.n_classes_:(j + 1) * self.n_classes_] = proba

        self.meta_learner_ = LogisticRegression(
            solver="lbfgs",
            class_weight="balanced",
            max_iter=2000,
            random_state=self.random_state,
        )
        self.meta_learner_.fit(meta_X, y)

        self.fitted_estimators_ = []
        for template in self.estimator_templates:
            fitted = skbase.clone(template)
            fitted.fit(X, y)
            self.fitted_estimators_.append(fitted)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.meta_learner_ is None:
            raise RuntimeError("GroupStackingEnsemble is not fitted")
        X = np.asarray(X, dtype=float)
        if self.n_classes_ <= 2:
            cols = np.column_stack(
                [est.predict_proba(X)[:, 1] for est in self.fitted_estimators_]
            )
        else:
            blocks = []
            for est in self.fitted_estimators_:
                proba = est.predict_proba(X)
                if proba.shape[1] != self.n_classes_:
                    full = np.zeros((len(X), self.n_classes_), dtype=float)
                    full[:, :proba.shape[1]] = proba
                    proba = full
                blocks.append(proba)
            cols = np.column_stack(blocks)
        return self.meta_learner_.predict_proba(cols)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def build_soft_voting_ensemble(top_models: list[tuple[str, dict]]) -> VotingClassifier:
    estimators = [(name, res["pipeline"]) for name, res in top_models]
    return VotingClassifier(estimators=estimators, voting="soft")


def build_stacking_ensemble(
    top_models: list[tuple[str, dict]],
    *,
    cv_folds: int,
    random_state: int,
) -> GroupStackingEnsemble:
    estimators = [(name, res["pipeline"]) for name, res in top_models]
    return GroupStackingEnsemble(estimators, cv_folds=cv_folds, random_state=random_state)


def build_ensemble_estimator(
    top_models: list[tuple[str, dict]],
    method: str,
    *,
    cv_folds: int,
    random_state: int,
) -> Any:
    if method == "soft_voting":
        return build_soft_voting_ensemble(top_models)
    if method == "stacking":
        return build_stacking_ensemble(
            top_models, cv_folds=cv_folds, random_state=random_state
        )
    raise ValueError(f"Unknown ensemble method: {method}")


def predict_ensemble_oof_proba(
    method: str,
    top_models: list[tuple[str, dict]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_test: np.ndarray,
    *,
    cv_folds: int,
    random_state: int,
) -> np.ndarray:
    """
    LOSO-fold ensemble probabilities: bases tuned on train_idx; meta/threshold unbiased.

    Binary: positive-class score (1d). Multiclass: full (n_samples, n_classes) matrix.
    Stacking uses ``GroupStackingEnsemble`` for binary and multiclass (ML-026).
    """
    y_train = np.asarray(y_train).astype(int)
    n_classes = len(np.unique(y_train))

    if method == "soft_voting":
        fitted = build_soft_voting_ensemble(top_models)
        fitted.fit(X_train, y_train)
        proba = fitted.predict_proba(X_test)
        return proba[:, 1] if n_classes <= 2 else proba

    if method == "stacking":
        fitted = build_stacking_ensemble(
            top_models, cv_folds=cv_folds, random_state=random_state
        )
        fitted.fit(X_train, y_train, groups=groups_train)
        proba = fitted.predict_proba(X_test)
        return proba[:, 1] if n_classes <= 2 else proba

    raise ValueError(f"Unknown ensemble method: {method}")
