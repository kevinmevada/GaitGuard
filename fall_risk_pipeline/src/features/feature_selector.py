"""
Subject-grouped feature selection (RFECV / SHAP) to cap dimensionality before modeling.

References:
  - Tibshirani (1996): Lasso shrinkage and variable selection via L1 penalty.
  - Guyon & Elisseeff (2002): Recursive Feature Elimination (RFE) framework.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.roc_auc_scoring import roc_auc_scoring_name
from src.evaluation.shap_multiclass import (
    global_mean_abs_importance,
    multiclass_display_name,
    per_class_mean_abs_importance,
    split_shap_by_class,
)
from src.features.feature_matrix import (
    SELECTED_FEATURES_FILE,
    get_numeric_feature_columns,
    load_patient_feature_matrix,
)
from src.features.feature_missingness import write_feature_missingness_report
from src.utils.progress import blocking_progress, progress_bar


class PermutationImportanceRandomForest(RandomForestClassifier):
    """
    Random Forest that stores permutation importances for RFECV RFE ranking.

    RFECV's ``importance_getter`` only receives the fitted estimator (not X/y),
    so permutation scores are computed at the end of ``fit`` and exposed via
    ``_permutation_importances_``. Permutation importance is far less biased
    than Gini/MDI when p >> n (Strobl et al., 2007).
    """

    def __init__(
        self,
        *,
        n_estimators: int = 120,
        max_depth: int = 6,
        class_weight: str | dict | None = "balanced",
        random_state: int = 0,
        n_jobs: int = 1,
        permutation_n_repeats: int = 5,
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.permutation_n_repeats = permutation_n_repeats

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight=sample_weight)
        result = permutation_importance(
            self,
            X,
            y,
            n_repeats=self.permutation_n_repeats,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._permutation_importances_ = result.importances_mean
        return self


def _rfe_pipeline_importance(estimator: Pipeline) -> np.ndarray:
    """Permutation (RFECV) or Gini importances from imputer→scaler→clf pipeline."""
    clf = estimator.named_steps["clf"]
    if hasattr(clf, "_permutation_importances_"):
        return np.asarray(clf._permutation_importances_)
    return np.asarray(clf.feature_importances_)


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
        self.required_feature_substrings = [
            str(s).strip().lower()
            for s in fscfg.get("required_feature_substrings", [])
            if str(s).strip()
        ]
        self.cv_folds = int(fscfg.get("cv_folds", config["models"]["tuning"]["cv_folds"]))
        self.random_state = int(config["models"]["evaluation"]["random_state"])
        self.compare_before_after = bool(fscfg.get("compare_before_after", True))
        self.parallel_jobs = self._resolve_parallel_jobs(fscfg)
        self.nested_n_jobs = max(int(fscfg.get("nested_n_jobs", 1)), 1)

        self.feat_dir = Path(config["paths"]["features"])
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_parallel_jobs(fscfg: dict) -> int:
        """Explicit job count for global RFECV (avoid sklearn ``n_jobs=-1`` oversubscription)."""
        if fscfg.get("n_jobs") is not None:
            return max(int(fscfg["n_jobs"]), 1)
        return max((os.cpu_count() or 1) - 1, 1)

    def run(self) -> dict:
        if not self.enabled:
            logger.info("Feature selection disabled in config.")
            return {}

        X, y, groups, feat_cols, df = load_patient_feature_matrix(
            self.config, use_selected=False
        )
        n_participants = len(df)
        n_features_before = len(feat_cols)
        p_n_before = n_features_before / max(n_participants, 1)

        logger.info(
            f"Feature selection input: n={n_participants}, p={n_features_before}, "
            f"P/N={p_n_before:.2f} (features per participant)"
        )

        missingness_df = write_feature_missingness_report(
            X, feat_cols, self.metrics_dir
        )
        high_missing = (
            missingness_df.loc[missingness_df["exceeds_threshold"], "feature"].tolist()
            if not missingness_df.empty
            else []
        )

        n_steps = 3 if self.compare_before_after else 2
        selection_steps = progress_bar(
            total=n_steps,
            desc="select_features",
            unit="step",
        )

        rfecv_features, rfecv_detail = self._select_rfecv(
            X, y, groups, feat_cols, n_jobs=self.parallel_jobs
        )
        selection_steps.update(1)

        shap_features, shap_detail = self._select_shap(X, y, feat_cols, groups=groups)
        selection_steps.update(1)

        if self.primary_method == "shap":
            selected = shap_features
            method_used = "shap"
        elif self.primary_method == "intersection":
            selected = [f for f in rfecv_features if f in set(shap_features)]
            if len(selected) < self.min_features:
                selected = rfecv_features[: self.max_features]
            rfecv_tag = "rfecv_capped" if rfecv_detail.get("capped_to_max_features") else "rfecv"
            method_used = f"intersection({rfecv_tag},shap)"
        else:
            selected = rfecv_features
            method_used = (
                "rfecv_capped"
                if rfecv_detail.get("capped_to_max_features")
                else "rfecv"
            )

        selected, forced_required, dropped_required, required_shap_audit = (
            self._enforce_required_features(
                selected, feat_cols, shap_detail.get("full_mean_abs_shap")
            )
        )
        n_features_after = len(selected)
        p_n_after = n_features_after / max(n_participants, 1)

        comparison_rows = []
        if self.compare_before_after:
            comparison_rows = self._compare_before_after(X, y, groups, feat_cols, selected)
            selection_steps.update(1)
        selection_steps.close()

        payload = {
            "n_participants": n_participants,
            "n_features_before": n_features_before,
            "n_features_after": n_features_after,
            "p_n_ratio_before": round(p_n_before, 3),
            "p_n_ratio_after": round(p_n_after, 3),
            "max_features": self.max_features,
            "primary_method": method_used,
            "rfecv_capped_to_max_features": bool(
                rfecv_detail.get("capped_to_max_features", False)
            ),
            "n_features_rfecv_cv_optimal": rfecv_detail.get("n_features_cv_optimal"),
            "required_feature_substrings": self.required_feature_substrings,
            "forced_required_features": forced_required,
            "dropped_required_features": dropped_required,
            "required_feature_shap_audit": required_shap_audit,
            "features": selected,
            "rfecv_features": rfecv_features,
            "shap_features": shap_features,
            "rfecv_detail": rfecv_detail,
            "shap_detail": shap_detail,
            "citations": CITATIONS,
            "comparison": comparison_rows,
            "high_missingness_features": high_missing,
            "feature_missingness_report": "feature_missingness_report.csv",
        }

        out_path = self.feat_dir / SELECTED_FEATURES_FILE
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        self._write_comparison_csv(comparison_rows)
        self._write_required_shap_audit_csv(required_shap_audit)
        self._write_missingness_section_md(missingness_df)
        self._write_report_md(payload)

        meta_cols = [c for c in df.columns if c not in get_numeric_feature_columns(df)]
        reduced = pd.concat([df[meta_cols], df[selected]], axis=1)
        reduced.to_parquet(self.feat_dir / "patient_features_selected.parquet", index=False)

        logger.info(
            f"Selected {n_features_after} features via {method_used} "
            f"(P/N {p_n_before:.2f} -> {p_n_after:.2f})"
        )
        return payload

    def _random_forest_classifier(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=120,
            max_depth=6,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1,
        )

    def _selection_pipeline(self) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", self._random_forest_classifier()),
        ])

    def _rfecv_pipeline(self) -> Pipeline:
        """RFECV pipeline: permutation importances drive RFE elimination (p >> n safe)."""
        fscfg = self.config.get("feature_selection", {})
        method = str(fscfg.get("rfecv_importance_method", "permutation")).lower()
        if method == "gini":
            clf: RandomForestClassifier | PermutationImportanceRandomForest = (
                self._random_forest_classifier()
            )
        else:
            n_repeats = int(fscfg.get("rfecv_permutation_n_repeats", 5))
            clf = PermutationImportanceRandomForest(
                n_estimators=120,
                max_depth=6,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=1,
                permutation_n_repeats=n_repeats,
            )
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def select_feature_names(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feat_cols: list[str],
        *,
        n_jobs: int | None = None,
    ) -> list[str]:
        """
        Group-aware RFECV (+ optional required-feature enforcement) on ``X``.

        Call on LOSO **train folds only** during evaluation to avoid leaking
        held-out subjects into the feature mask.
        """
        if n_jobs is None:
            n_jobs = self.nested_n_jobs
        rfecv_features, _ = self._select_rfecv(
            X, y, groups, feat_cols, n_jobs=n_jobs
        )
        if self.primary_method == "shap":
            shap_features, _ = self._select_shap(X, y, feat_cols, groups=groups)
            selected = shap_features
        elif self.primary_method == "intersection":
            shap_features, _ = self._select_shap(X, y, feat_cols, groups=groups)
            selected = [f for f in rfecv_features if f in set(shap_features)]
            if len(selected) < self.min_features:
                selected = rfecv_features[: self.max_features]
        else:
            selected = rfecv_features

        shap_importance: dict[str, float] = {}
        if self.required_feature_substrings:
            _, shap_detail = self._select_shap(X, y, feat_cols, groups=groups)
            shap_importance = shap_detail.get("full_mean_abs_shap", {})

        selected, _, _, _ = self._enforce_required_features(
            selected, feat_cols, shap_importance
        )
        return selected

    def _select_rfecv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feat_cols: list[str],
        *,
        n_jobs: int | None = None,
    ) -> tuple[list[str], dict]:
        if n_jobs is None:
            n_jobs = self.parallel_jobs
        cv = StratifiedGroupKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )
        # Match training pipeline: impute → scale → RF (ML-005).
        # Permutation importances (not Gini) drive RFE ranking under p >> n.
        estimator = self._rfecv_pipeline()
        selector = RFECV(
            estimator=estimator,
            step=0.05,
            cv=cv,
            scoring=roc_auc_scoring_name(y, self.config),
            min_features_to_select=self.min_features,
            n_jobs=n_jobs,
            importance_getter=_rfe_pipeline_importance,
        )
        rfecv_desc = f"rfecv p={len(feat_cols)}"
        with blocking_progress(rfecv_desc, unit="s", interval=30.0):
            selector.fit(X, y, groups=groups)

        n_cv_optimal = int(selector.n_features_)
        support_idx = np.where(selector.support_)[0]
        capped = len(support_idx) > self.max_features
        if capped:
            ranking = selector.ranking_
            order = np.argsort(ranking)[: self.max_features]
            support_idx = order
            logger.warning(
                "RFECV optimal={} but capped to max_features={} by config "
                "(exporting top {} by RFE ranking, not the CV-optimal set).",
                n_cv_optimal,
                self.max_features,
                self.max_features,
            )

        selected = [feat_cols[i] for i in support_idx]
        fscfg = self.config.get("feature_selection", {})
        detail = {
            "cv_best_score": float(selector.cv_results_["mean_test_score"].max()),
            "n_features_cv_optimal": n_cv_optimal,
            "n_features_exported": len(selected),
            "capped_to_max_features": capped,
            "max_features_cap": self.max_features,
            "importance_method": str(
                fscfg.get("rfecv_importance_method", "permutation")
            ).lower(),
        }
        return selected, detail

    def _rank_required_candidates(
        self, required: list[str], shap_importance: dict[str, float] | None
    ) -> list[str]:
        """ML-040: keep highest-SHAP required features when slots are capped."""
        if not shap_importance:
            return sorted(required)
        return sorted(
            required,
            key=lambda name: shap_importance.get(name, 0.0),
            reverse=True,
        )

    def _required_feature_shap_audit_rows(
        self,
        all_features: list[str],
        required_kept: list[str],
        forced: list[str],
        dropped_required: list[str],
        shap_importance: dict[str, float] | None,
    ) -> list[dict]:
        kept_set = set(required_kept)
        forced_set = set(forced)
        dropped_set = set(dropped_required)
        rows: list[dict] = []
        for feature in all_features:
            if not any(
                tok in feature.lower() for tok in self.required_feature_substrings
            ):
                continue
            if feature in kept_set:
                status = "forced_into_set" if feature in forced_set else "rfecv_or_shap_selected"
            elif feature in dropped_set:
                status = "dropped_by_max_required"
            else:
                status = "not_in_final_set"
            rows.append({
                "feature": feature,
                "mean_abs_shap": float(shap_importance.get(feature, float("nan")))
                if shap_importance
                else float("nan"),
                "status": status,
                "required_family": next(
                    tok
                    for tok in self.required_feature_substrings
                    if tok in feature.lower()
                ),
            })
        rows.sort(
            key=lambda row: (
                row["status"] != "forced_into_set",
                row["status"] != "rfecv_or_shap_selected",
                -row["mean_abs_shap"]
                if np.isfinite(row["mean_abs_shap"])
                else float("inf"),
            )
        )
        return rows

    def _enforce_required_features(
        self,
        selected: list[str],
        all_features: list[str],
        shap_importance: dict[str, float] | None = None,
    ) -> tuple[list[str], list[str], list[str], list[dict]]:
        """
        Force-keep required feature families while respecting max_features (ML-029/ML-040).

        When ``max_required_features`` caps the nonlinear family, candidates are ranked
        by mean |SHAP| (when available) rather than column order so weak predictors
        do not crowd out stronger RFECV slots.
        """
        if not self.required_feature_substrings:
            return selected[: self.max_features], [], [], []

        required = [
            f for f in all_features
            if any(tok in f.lower() for tok in self.required_feature_substrings)
        ]
        if not required:
            logger.warning(
                f"No features matched required_feature_substrings={self.required_feature_substrings}"
            )
            return selected[: self.max_features], [], [], []

        fscfg = self.config.get("feature_selection", {})
        max_required = int(fscfg.get("max_required_features", max(1, self.max_features // 2)))
        ranked_required = self._rank_required_candidates(required, shap_importance)
        required_kept = ranked_required[:max_required]
        dropped_required = ranked_required[max_required:]
        remaining_slots = max(self.max_features - len(required_kept), 0)
        rfecv_slots = [
            f for f in selected if f not in required_kept
        ][:remaining_slots]
        merged = required_kept + rfecv_slots
        n_rfecv_in_merged = len(rfecv_slots)

        forced = [f for f in required_kept if f not in selected]
        assert len(merged) <= self.max_features, (
            f"Feature count exceeds cap: merged={len(merged)}, "
            f"max_features={self.max_features}, forced={len(forced)}"
        )

        preview = merged[:20]
        suffix = "..." if len(merged) > 20 else ""
        logger.info(
            "Merged feature set after required enforcement (n=%d, required=%d, rfecv=%d): %s%s",
            len(merged),
            len(required_kept),
            n_rfecv_in_merged,
            preview,
            suffix,
        )
        if shap_importance and forced:
            logger.info(
                "Required features ranked by mean |SHAP| before cap (ML-040); "
                "see required_feature_shap_audit.csv for forced-vs-dropped comparison."
            )
        if n_rfecv_in_merged == 0 and selected:
            logger.warning(
                "Required features filled all %d slots; no RFECV-ranked features in final set. "
                "Lower max_required_features, narrow required_feature_substrings, or raise "
                "feature_selection.max_features (currently %d).",
                len(merged),
                self.max_features,
            )
        if forced:
            logger.info(
                f"Forced {len(forced)} required features into selected set: {forced[:8]}"
            )
        if dropped_required:
            logger.warning(
                "Required features exceed max_required_features=%d; dropped %d lowest-SHAP "
                "matches. Increase max_features or max_required_features only if audit shows gain.",
                max_required,
                len(dropped_required),
            )
        audit_rows = self._required_feature_shap_audit_rows(
            all_features,
            required_kept,
            forced,
            dropped_required,
            shap_importance,
        )
        return merged, forced, dropped_required, audit_rows

    def _select_shap(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feat_cols: list[str],
        groups: np.ndarray | None = None,
    ) -> tuple[list[str], dict]:
        pipe = self._selection_pipeline()
        if groups is not None:
            cv = StratifiedGroupKFold(
                n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
            )
            train_idx, _ = next(cv.split(X, y, groups))
            pipe.fit(X[train_idx], y[train_idx])
            X_proc = pipe[:-1].transform(X[train_idx])
        else:
            pipe.fit(X, y)
            X_proc = pipe[:-1].transform(X)
        clf = pipe.named_steps["clf"]

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(X_proc)
        per_class = split_shap_by_class(shap_vals, n_classes=int(len(np.unique(y))))
        importance = global_mean_abs_importance(per_class)
        order = np.argsort(importance)[::-1][: self.max_features]
        selected = [feat_cols[int(i)] for i in order]

        per_class_tops: dict[str, dict[str, float]] = {}
        for class_idx, class_imp in per_class_mean_abs_importance(per_class).items():
            class_order = np.argsort(class_imp)[::-1][: self.max_features]
            class_name = multiclass_display_name(class_idx)
            per_class_tops[class_name] = {
                feat_cols[int(i)]: float(class_imp[int(i)]) for i in class_order
            }

        detail = {
            "top_mean_abs_shap": {
                selected[i]: float(importance[int(order[i])]) for i in range(len(selected))
            },
            "full_mean_abs_shap": {
                feat_cols[i]: float(importance[i]) for i in range(len(feat_cols))
            },
            "aggregation": "global_class_average",
            "per_class_top_mean_abs_shap": per_class_tops,
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
            (
                "after_global_selection_grouped_cv_exploratory",
                X_sel,
                X_sel.shape[1],
            ),
        ):
            scores = cross_val_score(
                pipe,
                X_use,
                y,
                cv=cv,
                groups=groups,
                scoring=roc_auc_scoring_name(y, self.config),
                n_jobs=self.parallel_jobs,
            )
            rows.append({
                "stage": label,
                "n_features": n_feat,
                "cv_auc_mean": float(np.mean(scores)),
                "cv_auc_std": float(np.std(scores)),
                "cv_folds": self.cv_folds,
                "validation": "stratified_group_kfold",
                "nested_selection": False,
                "global_rfecv_mask": label == "after_global_selection_grouped_cv_exploratory",
            })
        return rows

    def _write_comparison_csv(self, rows: list[dict]) -> None:
        if not rows:
            return
        pd.DataFrame(rows).to_csv(
            self.metrics_dir / "feature_selection_comparison.csv",
            index=False,
        )

    def _write_required_shap_audit_csv(self, rows: list[dict]) -> None:
        if not rows:
            return
        pd.DataFrame(rows).to_csv(
            self.metrics_dir / "required_feature_shap_audit.csv",
            index=False,
        )

    def _write_missingness_section_md(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        lines = [
            "## Feature missingness (MED-006)",
            "",
            "Non-finite rate per feature before selection (`feature_missingness_report.csv`). "
            "Features with >15% missing may produce unstable median imputation across LOSO folds.",
            "",
            "| Feature | Missing % | Flagged |",
            "|---|---:|:---:|",
        ]
        for row in df.head(15).itertuples(index=False):
            flag = "yes" if row.exceeds_threshold else ""
            lines.append(
                f"| {row.feature} | {row.missing_pct:.1f} | {flag} |"
            )
        if len(df) > 15:
            lines.append(f"| … | ({len(df) - 15} more in CSV) | |")
        lines.append("")
        path = self.metrics_dir / "feature_missingness_summary.md"
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_report_md(self, payload: dict) -> None:
        lines = [
            "# Feature Selection Report",
            "",
            "## Sample size vs dimensionality",
            "",
            f"- Participants (N): **{payload['n_participants']}**",
            f"- Features before selection (p): **{payload['n_features_before']}**",
            f"- Features exported for training: **{payload['n_features_after']}**",
            f"- Configured `max_features` cap: **{payload['max_features']}**",
            f"- P/N ratio before (p/N): **{payload['p_n_ratio_before']:.2f}**",
            f"- P/N ratio after (p/N): **{payload['p_n_ratio_after']:.2f}**",
            "",
            (
                f"Patient-level features aggregate each trial biomarker as mean/std/range/trend, "
                f"yielding p={payload['n_features_before']} columns for N={payload['n_participants']} "
                f"participants (P/N \u2248 {payload['p_n_ratio_before']:.2f}). "
                f"With p \u226b N, grouped feature selection is applied before final training "
                f"to reduce P/N to \u2248 {payload['p_n_ratio_after']:.2f}."
            ),
            "",
            "## Methods",
            "",
            "### RFECV ranking (with optional dimensionality cap)",
            "",
            "Recursive Feature Elimination with subject-grouped cross-validation (RFECV), "
            "following the RFE framework of Guyon & Elisseeff (2002). The selector uses "
            "StratifiedGroupKFold so no participant appears in both train and validation. "
            "RFE elimination ranks features by **permutation importance** (not Gini/MDI) "
            "to avoid high-variance bias when p >> n.",
            "",
            f"- RFECV grouped-CV optimal feature count: "
            f"**{payload['rfecv_detail'].get('n_features_cv_optimal', 'n/a')}**",
            f"- Features exported after cap: **{len(payload['rfecv_features'])}**",
        ]

        if payload.get("rfecv_capped_to_max_features"):
            lines.extend([
                "",
                "> **Cap applied:** RFECV cross-validation favoured "
                f"{payload['rfecv_detail'].get('n_features_cv_optimal', 'n/a')} features, "
                f"but `max_features={payload['max_features']}` deliberately limits the export "
                "to the top-ranked features (dimensionality cap to lower P/N). "
                "Do **not** report this as 'RFECV selected "
                f"{payload['n_features_after']} features' — report primary method as "
                f"**{payload['primary_method']}**.",
            ])

        if payload.get("forced_required_features"):
            lines.extend([
                "",
                "### Required nonlinear families (ML-040)",
                "",
                f"- Forced into final set: **{len(payload['forced_required_features'])}**",
                f"- Dropped by `max_required_features` cap: "
                f"**{len(payload.get('dropped_required_features', []))}**",
                "- Candidates are ranked by mean |SHAP| before the cap is applied.",
                "- Review `required_feature_shap_audit.csv` after rerun to compare "
                "forced vs dropped nonlinear features.",
                "- **Sensitivity (MED-005):** re-run with `required_feature_substrings: []` "
                "and compare LOSO AUC / SHAP ranks to this default.",
            ])

        lines.extend([
            "",
            "### SHAP-based pruning (secondary ranking)",
            "",
            f"Mean absolute SHAP values from a grouped-CV-safe surrogate Random Forest "
            f"provide an alternate top-{payload['max_features']} ranking for comparison. "
            f"Global rankings average |SHAP| across classes; see `shap_detail.per_class_top_mean_abs_shap` "
            f"in selected_features.json for tier-specific rankings.",
            "",
            f"- Primary method used for training: **{payload['primary_method']}**",
            "",
            "## Before / after (grouped CV, Random Forest surrogate)",
            "",
            "> The **after_global_selection** row uses a feature mask fit on all "
            "participants and is **exploratory only** — not a nested selection "
            "estimate. Primary LOSO evaluation uses per-fold selection when "
            "`nested_in_evaluation: true`.",
            "",
        ])

        for row in payload.get("comparison", []):
            nested = "nested" if row.get("nested_selection") else "exploratory (global mask)"
            lines.append(
                f"- **{row['stage']}** ({nested}): p={row['n_features']}, "
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
