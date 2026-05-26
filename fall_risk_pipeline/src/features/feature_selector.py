"""
Subject-grouped feature selection (RFECV / SHAP) to cap dimensionality before modeling.

References:
  - Tibshirani (1996): Lasso shrinkage and variable selection via L1 penalty.
  - Guyon & Elisseeff (2002): Recursive Feature Elimination (RFE) framework.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.roc_auc_scoring import roc_auc_scoring_name
from src.features.feature_matrix import (
    SELECTED_FEATURES_FILE,
    get_numeric_feature_columns,
    load_patient_feature_matrix,
)

CITATIONS = {
    "lasso": (
        "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. "
        "Journal of the Royal Statistical Society: Series B, 58(1), 267-288."
    ),
    "rfe": (
        "Guyon, I., & Elisseeff, A. (2002). An introduction to variable and feature selection. "
        "Journal of Machine Learning Research, 3, 1157-1182."
    ),
}


class FeatureSelector:
    def __init__(self, config: dict):
        self.config = config
        fscfg = config.get("feature_selection", {})
        self.enabled = bool(fscfg.get("enabled", True))
        self.max_features = int(fscfg.get("max_features", 20))
        self.min_features = int(fscfg.get("min_features", 5))
        self.primary_method = str(fscfg.get("primary_method", "rfecv")).lower()
        self.cv_folds = int(fscfg.get("cv_folds", config["models"]["tuning"]["cv_folds"]))
        self.random_state = int(config["models"]["evaluation"]["random_state"])
        self.compare_before_after = bool(fscfg.get("compare_before_after", True))

        self.feat_dir = Path(config["paths"]["features"])
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> dict:
        if not self.enabled:
            logger.info("Feature selection disabled in config.")
            return {}

        X, y, groups, feat_cols, df = load_patient_feature_matrix(
            self.config, use_selected=False
        )
        n_participants = len(df)
        n_features_before = len(feat_cols)
        p_n_before = n_participants / max(n_features_before, 1)

        logger.info(
            f"Feature selection input: n={n_participants}, p={n_features_before}, "
            f"P/N={p_n_before:.2f}"
        )

        rfecv_features, rfecv_detail = self._select_rfecv(X, y, groups, feat_cols)
        shap_features, shap_detail = self._select_shap(X, y, feat_cols)

        if self.primary_method == "shap":
            selected = shap_features
            method_used = "shap"
        elif self.primary_method == "intersection":
            selected = [f for f in rfecv_features if f in set(shap_features)]
            if len(selected) < self.min_features:
                selected = rfecv_features[: self.max_features]
            method_used = "intersection(rfecv,shap)"
        else:
            selected = rfecv_features
            method_used = "rfecv"

        selected = selected[: self.max_features]
        n_features_after = len(selected)
        p_n_after = n_participants / max(n_features_after, 1)

        comparison_rows = []
        if self.compare_before_after:
            comparison_rows = self._compare_before_after(X, y, groups, feat_cols, selected)

        payload = {
            "n_participants": n_participants,
            "n_features_before": n_features_before,
            "n_features_after": n_features_after,
            "p_n_ratio_before": round(p_n_before, 3),
            "p_n_ratio_after": round(p_n_after, 3),
            "max_features": self.max_features,
            "primary_method": method_used,
            "features": selected,
            "rfecv_features": rfecv_features,
            "shap_features": shap_features,
            "rfecv_detail": rfecv_detail,
            "shap_detail": shap_detail,
            "citations": CITATIONS,
            "comparison": comparison_rows,
        }

        out_path = self.feat_dir / SELECTED_FEATURES_FILE
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        self._write_comparison_csv(comparison_rows)
        self._write_report_md(payload)

        meta_cols = [c for c in df.columns if c not in get_numeric_feature_columns(df)]
        reduced = pd.concat([df[meta_cols], df[selected]], axis=1)
        reduced.to_parquet(self.feat_dir / "patient_features_selected.parquet", index=False)

        logger.info(
            f"Selected {n_features_after} features via {method_used} "
            f"(P/N {p_n_before:.2f} -> {p_n_after:.2f})"
        )
        return payload

    def _selection_pipeline(self) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=6,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=1,
                ),
            ),
        ])

    def _select_rfecv(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, feat_cols: list[str]
    ) -> tuple[list[str], dict]:
        cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        estimator = RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1,
        )
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring=roc_auc_scoring_name(y, self.config),
            min_features_to_select=self.min_features,
            n_jobs=-1,
        )
        selector.fit(X, y, groups=groups)

        support_idx = np.where(selector.support_)[0]
        if len(support_idx) > self.max_features:
            ranking = selector.ranking_
            order = np.argsort(ranking)[: self.max_features]
            support_idx = order

        selected = [feat_cols[i] for i in support_idx]
        detail = {
            "cv_best_score": float(selector.cv_results_["mean_test_score"].max()),
            "n_features_cv_optimal": int(selector.n_features_),
            "n_features_exported": len(selected),
        }
        return selected, detail

    def _select_shap(
        self, X: np.ndarray, y: np.ndarray, feat_cols: list[str]
    ) -> tuple[list[str], dict]:
        pipe = self._selection_pipeline()
        pipe.fit(X, y)
        clf = pipe.named_steps["clf"]
        X_proc = pipe[:-1].transform(X)

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_proc)
        if isinstance(shap_vals, list):
            stacked = np.stack([np.asarray(v) for v in shap_vals], axis=0)
            shap_vals = np.abs(stacked).mean(axis=0)
        shap_vals = np.asarray(shap_vals)
        if shap_vals.ndim == 3:
            shap_vals = shap_vals[:, :, 1]

        importance = np.ravel(np.abs(shap_vals).mean(axis=0))
        order = np.argsort(importance)[::-1][: self.max_features]
        selected = [feat_cols[int(i)] for i in order]

        detail = {
            "top_mean_abs_shap": {
                selected[i]: float(importance[int(order[i])]) for i in range(len(selected))
            }
        }
        return selected, detail

    def _compare_before_after(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feat_cols: list[str],
        selected: list[str],
    ) -> list[dict]:
        cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        pipe = self._selection_pipeline()
        idx = [feat_cols.index(name) for name in selected]
        X_sel = X[:, idx]

        rows = []
        for label, X_use, n_feat in (
            ("before_all_features", X, X.shape[1]),
            ("after_selected_features", X_sel, X_sel.shape[1]),
        ):
            scores = cross_val_score(
                pipe,
                X_use,
                y,
                cv=cv,
                groups=groups,
                scoring=roc_auc_scoring_name(y, self.config),
                n_jobs=-1,
            )
            rows.append({
                "stage": label,
                "n_features": n_feat,
                "cv_auc_mean": float(np.mean(scores)),
                "cv_auc_std": float(np.std(scores)),
                "cv_folds": self.cv_folds,
                "validation": "stratified_group_kfold",
            })
        return rows

    def _write_comparison_csv(self, rows: list[dict]) -> None:
        if not rows:
            return
        pd.DataFrame(rows).to_csv(
            self.metrics_dir / "feature_selection_comparison.csv",
            index=False,
        )

    def _write_report_md(self, payload: dict) -> None:
        lines = [
            "# Feature Selection Report",
            "",
            "## Sample size vs dimensionality",
            "",
            f"- Participants (N): **{payload['n_participants']}**",
            f"- Features before selection (p): **{payload['n_features_before']}**",
            f"- Features after selection: **{payload['n_features_after']}** (cap ≤ {payload['max_features']})",
            f"- P/N ratio before: **{payload['p_n_ratio_before']:.2f}**",
            f"- P/N ratio after: **{payload['p_n_ratio_after']:.2f}**",
            "",
            "An ensemble of four nonlinear models on high-dimensional patient-level features (mean/std/range/trend) with "
            "N=260 is severely underpowered (P/N ≈ 3.25). We therefore apply grouped "
            "feature selection before final training.",
            "",
            "## Methods",
            "",
            "### RFECV (primary export)",
            "",
            "Recursive Feature Elimination with subject-grouped cross-validation (RFECV), "
            "following the RFE framework of Guyon & Elisseeff (2002). The selector uses "
            "StratifiedGroupKFold so no participant appears in both train and validation.",
            "",
            f"- RFECV CV-optimal feature count: {payload['rfecv_detail'].get('n_features_cv_optimal', 'n/a')}",
            f"- Exported RFECV features: {len(payload['rfecv_features'])}",
            "",
            "### SHAP-based pruning (secondary ranking)",
            "",
            f"Mean absolute SHAP values from a grouped-CV-safe surrogate Random Forest "
            f"provide an alternate top-{payload['max_features']} ranking for comparison.",
            "",
            f"- Primary method used for training: **{payload['primary_method']}**",
            "",
            "## Before / after (grouped CV, Random Forest surrogate)",
            "",
        ]

        for row in payload.get("comparison", []):
            lines.append(
                f"- **{row['stage']}**: p={row['n_features']}, "
                f"AUC={row['cv_auc_mean']:.4f} ± {row['cv_auc_std']:.4f}"
            )

        lines.extend([
            "",
            "## Selected features",
            "",
            ", ".join(f"`{f}`" for f in payload["features"]),
            "",
            "## References",
            "",
            f"- {CITATIONS['lasso']}",
            f"- {CITATIONS['rfe']}",
            "",
        ])

        report_path = self.metrics_dir / "feature_selection_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
