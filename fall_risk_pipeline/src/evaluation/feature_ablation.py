"""
Feature-group ablation study (LOSO macro-OVR AUC).

Scenarios:
  1. all_features_nested_rfecv — full patient matrix intersected with per-fold nested RFECV (ML-035)
  2. top10_shap — top-K features by mean |SHAP| on LOSO held-out folds
     (same per-fold nested RFECV ∩ scenario columns as `_loso_evaluate`, ML-033)
  3. minus_<group> — leave-one feature group out (temporal, spectral, …)
  4. minus_lyapunov — drop only lyapunov_* columns (nonlinear dynamics probe)
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn.base as skbase
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.dataset.label_policy import is_binary_task, label_mode_description
from src.evaluation.clinical_threshold import (
    fixed_threshold_when_inner_cv_unavailable,
    threshold_from_inner_oof,
)
from src.evaluation.multiclass_metrics import build_multiclass_metric_payload, predict_multiclass
from src.evaluation.shap_multiclass import global_mean_abs_importance, split_shap_by_class
from src.features.feature_groups import (
    count_trial_features,
    patient_columns_for_trial_bases,
    patient_columns_minus_group,
    patient_columns_minus_trial_bases,
    summarize_ablation_groups,
    trial_feature_groups,
)
from src.features.feature_matrix import (
    intersect_nested_rfecv_columns,
    load_patient_feature_matrix,
)
from src.models.trainer import ModelTrainer
from src.utils.checkpoint_io import CheckpointIntegrityError, load_checkpoint
from src.utils.progress import progress_bar
from src.utils.reproducibility import get_pipeline_seed

BASELINE_ABLATION_SCENARIO = "all_features_nested_rfecv"


class FeatureAblationStudy:
    def __init__(self, config: dict):
        self.config = config
        ab_cfg = config.get("ablation", {})
        self.reference_model = str(ab_cfg.get("reference_model", "xgboost"))
        self.top_k_shap = int(ab_cfg.get("top_k_shap", 10))
        self.n_bootstrap = int(ab_cfg.get("n_bootstrap", 1000))
        self.random_state = get_pipeline_seed(config)

        self.feat_dir = Path(config["paths"]["features"])
        self.ckpt_dir = Path(config["paths"]["checkpoints"])
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.fig_dir = Path(config["paths"]["figures_models"])
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.trainer = ModelTrainer(config)
        self.dpi = int(config.get("reporting", {}).get("figure_dpi", 300))
        self.fmt = config.get("reporting", {}).get("figure_format", "pdf")

    def run(self) -> pd.DataFrame:
        logger.info(label_mode_description(self.config))
        X, y, groups, feat_names, df = load_patient_feature_matrix(
            self.config, use_selected=False
        )
        cohorts = df["cohort"].astype(str).values
        n_trial = count_trial_features(self.config)
        logger.info(
            f"Ablation matrix: {len(feat_names)} patient-level columns "
            f"from {n_trial} trial-level features × aggregations"
        )

        group_summary = summarize_ablation_groups(feat_names, self.config)
        group_summary.to_csv(self.metrics_dir / "ablation_group_column_counts.csv", index=False)

        top_shap = self._resolve_top_shap_features(X, y, groups, feat_names)
        top_path = self.metrics_dir / "ablation_top_shap_features.json"
        top_path.write_text(json.dumps(top_shap, indent=2), encoding="utf-8")

        scenarios = self._build_scenarios(feat_names, top_shap)
        checkpoint = self._load_checkpoint(self.reference_model)
        if checkpoint is None:
            raise FileNotFoundError(
                f"Ablation requires {self.reference_model}.pkl in checkpoints "
                "(run train stage first)."
            )

        rows: list[dict[str, Any]] = []
        for scenario_id, columns, description in progress_bar(
            scenarios,
            desc="ablation",
            unit="scenario",
        ):
            col_idx = [feat_names.index(c) for c in columns]
            result = self._loso_evaluate(
                scenario_id,
                checkpoint,
                X,
                y,
                groups,
                cohorts,
                feat_names=feat_names,
                scenario_col_idx=col_idx,
            )
            rows.append(
                {
                    "scenario": scenario_id,
                    "description": description,
                    "reference_model": self.reference_model,
                    "n_features": len(columns),
                    "n_trial_features_config": n_trial,
                    "auc": result["auc"],
                    "auc_ci_low": result.get("auc_ci_low", float("nan")),
                    "auc_ci_high": result.get("auc_ci_high", float("nan")),
                    "macro_f1": result.get("f1", float("nan")),
                    "accuracy": result.get("accuracy", float("nan")),
                    "validation": "loso_subject_grouped",
                    "feature_list": json.dumps(columns),
                }
            )
            logger.info(
                f"Ablation {scenario_id}: n={len(columns)} "
                f"AUC={result['auc']:.4f} "
                f"[{result.get('auc_ci_low', float('nan')):.3f}, "
                f"{result.get('auc_ci_high', float('nan')):.3f}]"
            )

        out_df = pd.DataFrame(rows).sort_values("auc", ascending=False)
        out_df.to_csv(self.metrics_dir / "feature_ablation.csv", index=False)
        self._write_markdown(out_df, n_trial, top_shap)
        self._plot_ablation_bars(out_df)
        return out_df

    def _build_scenarios(
        self,
        all_columns: list[str],
        top_shap: list[str],
    ) -> list[tuple[str, list[str], str]]:
        scenarios: list[tuple[str, list[str], str]] = []

        scenarios.append(
            (
                BASELINE_ABLATION_SCENARIO,
                list(all_columns),
                "Full patient-level matrix ∩ per-fold nested RFECV "
                "(nested_in_ablation; not unmasked all_features)",
            )
        )

        top_cols = [c for c in top_shap if c in all_columns]
        if len(top_cols) < self.top_k_shap:
            logger.warning(
                f"Only {len(top_cols)} SHAP top features found in matrix "
                f"(requested {self.top_k_shap})"
            )
        scenarios.append(
            (
                f"top{self.top_k_shap}_shap",
                top_cols,
                f"Top {self.top_k_shap} features by LOSO mean |SHAP| "
                f"(nested RFECV per fold, ML-033)",
            )
        )

        for group_name in trial_feature_groups(self.config):
            kept = patient_columns_minus_group(all_columns, group_name, self.config)
            scenarios.append(
                (
                    f"minus_{group_name}",
                    kept,
                    f"All features except {group_name} group",
                )
            )

        lyap_cols = patient_columns_for_trial_bases(all_columns, ["lyapunov"])
        if lyap_cols:
            scenarios.append(
                (
                    "minus_lyapunov",
                    patient_columns_minus_trial_bases(all_columns, ["lyapunov"]),
                    "All features except lyapunov (mean/std/range/trend)",
                )
            )

        return scenarios

    def _resolve_top_shap_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feat_names: list[str],
    ) -> list[str]:
        ckpt_path = self.ckpt_dir / f"{self.reference_model}.pkl"
        ckpt_mtime = int(ckpt_path.stat().st_mtime) if ckpt_path.exists() else 0
        cache = self.metrics_dir / (
            f"ablation_top_shap_features_{self.reference_model}_{self.seed}_{ckpt_mtime}.json"
        )
        if cache.exists():
            try:
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if (
                    isinstance(cached, dict)
                    and cached.get("source") == "loso_aggregate_nested_rfecv"
                    and cached.get("cache_key") == f"{self.reference_model}:{self.seed}:{ckpt_mtime}"
                    and isinstance(cached.get("features"), list)
                    and len(cached["features"]) >= 1
                ):
                    return [str(x) for x in cached["features"][: self.top_k_shap]]
            except json.JSONDecodeError:
                pass

        importance = self._loso_shap_importance(X, y, groups, feat_names)
        order = sorted(importance, key=importance.get, reverse=True)[: self.top_k_shap]
        payload = {
            "source": "loso_aggregate_nested_rfecv",
            "cache_key": f"{self.reference_model}:{self.seed}:{ckpt_mtime}",
            "features": order,
        }
        cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"SHAP top-{self.top_k_shap} (LOSO aggregate): {order}")
        return order

    def _train_fold_youden_binary(
        self,
        fold_model: Any,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        train_idx: np.ndarray,
    ) -> float:
        """Inner grouped OOF Youden on LOSO train fold (ML-017 / ML-037)."""
        X_tr = X[train_idx]
        y_tr = y[train_idx]
        g_tr = groups[train_idx]
        n_groups = len(np.unique(g_tr))
        if n_groups < 3 or len(np.unique(y_tr)) < 2:
            return fixed_threshold_when_inner_cv_unavailable()[0]

        n_splits = min(5, n_groups)
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )
        oof_prob = np.full(len(y_tr), np.nan, dtype=float)
        for inner_train, inner_val in cv.split(X_tr, y_tr, g_tr):
            if len(np.unique(y_tr[inner_train])) < 2:
                continue
            inner_model = skbase.clone(fold_model)
            self.trainer.fit_pipeline(
                model_name, inner_model, X_tr[inner_train], y_tr[inner_train]
            )
            oof_prob[inner_val] = inner_model.predict_proba(X_tr[inner_val])[:, 1]

        return threshold_from_inner_oof(y_tr, oof_prob, n_splits=n_splits)[0]

    def _loso_shap_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        feat_names: list[str],
    ) -> dict[str, float]:
        """Mean |SHAP| per feature from LOSO held-out explanations (train-only fit).

        Uses the same per-fold nested RFECV column mask as ``_loso_evaluate`` (ML-033).
        """
        checkpoint = self._load_checkpoint(self.reference_model)
        if checkpoint is None:
            raise FileNotFoundError("Cannot compute SHAP top features without a checkpoint.")

        accum: dict[str, list[float]] = defaultdict(list)
        all_col_idx = list(range(len(feat_names)))

        for subj in np.unique(groups):
            test_idx = np.where(groups == subj)[0]
            train_idx = np.where(groups != subj)[0]
            if len(np.unique(y[train_idx])) < 2:
                continue

            fold_cols = intersect_nested_rfecv_columns(
                self.config,
                X,
                y,
                groups,
                feat_names,
                train_idx,
                all_col_idx,
            )
            if not fold_cols:
                continue

            fold_model = skbase.clone(checkpoint)
            self.trainer.fit_pipeline(
                self.reference_model,
                fold_model,
                X[train_idx][:, fold_cols],
                y[train_idx],
            )
            clf = fold_model.named_steps.get("clf")
            if clf is None or not hasattr(clf, "predict_proba"):
                raise TypeError("Reference model has no tree classifier for SHAP.")

            X_proc = self._transform_for_shap(fold_model, X[test_idx][:, fold_cols])
            explainer = shap.TreeExplainer(clf)
            shap_vals = explainer.shap_values(X_proc)
            per_class = split_shap_by_class(
                shap_vals, n_classes=int(len(np.unique(y)))
            )
            mean_abs = global_mean_abs_importance(per_class)
            for local_idx, col_idx in enumerate(fold_cols):
                accum[feat_names[col_idx]].append(float(mean_abs[local_idx]))

        if not accum:
            raise RuntimeError("No LOSO SHAP values collected for ablation top-K.")

        return {name: float(np.mean(vals)) for name, vals in accum.items()}

    @staticmethod
    def _transform_for_shap(model: Any, X: np.ndarray) -> np.ndarray:
        X_proc = X
        imputer = model.named_steps.get("imputer")
        scaler = model.named_steps.get("scaler")
        if imputer is not None:
            X_proc = imputer.transform(X_proc)
        if scaler is not None:
            X_proc = scaler.transform(X_proc)
        return X_proc

    def _loso_evaluate(
        self,
        scenario_id: str,
        checkpoint: Any,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        cohorts: np.ndarray,
        *,
        feat_names: list[str],
        scenario_col_idx: list[int],
    ) -> dict[str, Any]:
        if not scenario_col_idx:
            return {"auc": float("nan"), "f1": float("nan"), "accuracy": float("nan")}

        binary_task = is_binary_task(y, self.config)
        all_true: list[int] = []
        all_probs: list[Any] = []
        all_pred: list[int] = []
        all_cohorts: list[str] = []

        for subj in np.unique(groups):
            test_idx = np.where(groups == subj)[0]
            train_idx = np.where(groups != subj)[0]
            if len(np.unique(y[train_idx])) < 2:
                continue

            fold_cols = intersect_nested_rfecv_columns(
                self.config,
                X,
                y,
                groups,
                feat_names,
                train_idx,
                scenario_col_idx,
            )
            if not fold_cols:
                continue

            fold_model = skbase.clone(checkpoint)
            self.trainer.fit_pipeline(
                self.reference_model, fold_model, X[train_idx][:, fold_cols], y[train_idx]
            )

            if binary_task:
                proba = fold_model.predict_proba(X[test_idx][:, fold_cols])
                score = proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
                thresh = self._train_fold_youden_binary(
                    fold_model,
                    self.reference_model,
                    X[:, fold_cols],
                    y,
                    groups,
                    train_idx,
                )
                all_probs.extend(score.tolist())
                all_pred.extend((score >= thresh).astype(int).tolist())
            else:
                proba, pred = predict_multiclass(fold_model, X[test_idx][:, fold_cols])
                all_probs.append(proba)
                all_pred.extend(pred.tolist())

            all_true.extend(y[test_idx].tolist())
            all_cohorts.extend(cohorts[test_idx].tolist())

        y_true = np.asarray(all_true, dtype=int)
        if binary_task:
            y_prob = np.asarray(all_probs, dtype=float)
            y_pred = np.asarray(all_pred, dtype=int)
            payload = self._binary_payload(scenario_id, y_true, y_prob, y_pred)
        else:
            y_proba = np.vstack(all_probs) if all_probs else np.empty((0, 3))
            y_pred = np.asarray(all_pred, dtype=int)
            payload = build_multiclass_metric_payload(
                scenario_id,
                y_true,
                y_proba,
                y_pred,
                seed=self.random_state,
                cohorts=np.asarray(all_cohorts),
            )
            ci_low, ci_high = self._bootstrap_macro_auc_ci(y_true, y_proba)
            payload["auc_ci_low"] = ci_low
            payload["auc_ci_high"] = ci_high

        return payload

    def _binary_payload(
        self,
        name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, Any]:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")
        ci_low, ci_high = self._bootstrap_binary_auc_ci(y_true, y_prob)
        from sklearn.metrics import accuracy_score, f1_score

        return {
            "auc": auc,
            "auc_ci_low": ci_low,
            "auc_ci_high": ci_high,
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
        }

    def _bootstrap_binary_auc_ci(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> tuple[float, float]:
        rng = np.random.default_rng(self.random_state)
        idx_all = np.arange(len(y_true))
        samples: list[float] = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            yt, yp = y_true[idx], y_prob[idx]
            if len(np.unique(yt)) < 2:
                continue
            samples.append(float(roc_auc_score(yt, yp)))
        if not samples:
            return float("nan"), float("nan")
        return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

    def _bootstrap_macro_auc_ci(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> tuple[float, float]:
        rng = np.random.default_rng(self.random_state)
        idx_all = np.arange(len(y_true))
        labels = sorted(np.unique(y_true))
        samples: list[float] = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(idx_all, size=len(idx_all), replace=True)
            yt = y_true[idx]
            yp = y_proba[idx]
            if len(np.unique(yt)) < 2:
                continue
            try:
                samples.append(
                    float(
                        roc_auc_score(
                            yt,
                            yp,
                            multi_class="ovr",
                            average="macro",
                            labels=labels,
                        )
                    )
                )
            except ValueError:
                continue
        if not samples:
            return float("nan"), float("nan")
        return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

    def _load_checkpoint(self, name: str) -> Any | None:
        path = self.ckpt_dir / f"{name}.pkl"
        if not path.exists():
            return None
        try:
            return load_checkpoint(
                path,
                manifest_dir=self.ckpt_dir,
                require_manifest=False,
            )
        except CheckpointIntegrityError as exc:
            logger.warning("Checkpoint verification failed for %s: %s", name, exc)
            return None

    def _write_markdown(
        self,
        df: pd.DataFrame,
        n_trial: int,
        top_shap: list[str],
    ) -> None:
        lines = [
            "# Feature ablation (LOSO macro-OVR AUC)",
            "",
            f"Reference classifier: **{self.reference_model}** (checkpoint hyperparameters, "
            "re-fit per LOSO fold).",
            "",
            f"Trial-level features in config: **{n_trial}**; patient-level columns vary by "
            "aggregation (mean, std, range, trend).",
            "",
            f"Top-{self.top_k_shap} SHAP features (LOSO aggregate, nested RFECV per fold): "
            + ", ".join(f"`{f}`" for f in top_shap),
            "",
            "| Scenario | n features | AUC | 95% CI | Macro F1 |",
            "|---|---:|---:|---|---:|",
        ]
        for row in df.itertuples(index=False):
            ci = (
                f"[{row.auc_ci_low:.3f}, {row.auc_ci_high:.3f}]"
                if pd.notna(row.auc_ci_low) and pd.notna(row.auc_ci_high)
                else "—"
            )
            lines.append(
                f"| {row.scenario} | {int(row.n_features)} | "
                f"{row.auc:.3f} | {ci} | {row.macro_f1:.3f} |"
            )
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                f"- Compare `{BASELINE_ABLATION_SCENARIO}` vs `top{self.top_k_shap}_shap`: "
                "if AUC is similar, a compact SHAP subset may suffice.",
                f"- Compare each `minus_*` row to `{BASELINE_ABLATION_SCENARIO}`: "
                "larger AUC drops indicate groups that contribute most.",
                "- `minus_lyapunov` isolates the Lyapunov exponent (under `trunk_dynamics`); compare to `minus_trunk_dynamics`.",
                "",
                "Outputs: `feature_ablation.csv`, `ablation_group_column_counts.csv`, "
                "`figures/models/feature_ablation_bars.*`",
            ]
        )
        path = self.metrics_dir / "feature_ablation.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Ablation report -> {path}")

    def _plot_ablation_bars(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        plot_df = df.sort_values("auc", ascending=True)
        fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(plot_df))))
        y_pos = np.arange(len(plot_df))
        aucs = plot_df["auc"].values.astype(float)
        colors = [
            "#1976D2" if s == BASELINE_ABLATION_SCENARIO else "#64B5F6"
            for s in plot_df["scenario"]
        ]
        ax.barh(y_pos, aucs, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df["scenario"].tolist(), fontsize=9)
        ax.set_xlabel("Macro one-vs-rest AUC (LOSO)")
        ax.set_title(f"Feature ablation ({self.reference_model})")
        for i, row in enumerate(plot_df.itertuples(index=False)):
            if pd.notna(row.auc_ci_low) and pd.notna(row.auc_ci_high):
                ax.plot(
                    [row.auc_ci_low, row.auc_ci_high],
                    [i, i],
                    color="black",
                    linewidth=2,
                )
        fig.tight_layout()
        out = self.fig_dir / f"feature_ablation_bars.{self.fmt}"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Ablation figure -> {out}")


def run_feature_ablation(config: dict) -> pd.DataFrame:
    return FeatureAblationStudy(config).run()
