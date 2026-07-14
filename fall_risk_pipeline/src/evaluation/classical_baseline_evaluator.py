"""
Classical baselines on Phase 1 + Phase 2 handcrafted features (competitor paradigm 1).

Models (literature-aligned):
  - SVM (RBF) — Moon 2020, Trabassi 2022, Li 2025, Prisco 2025
  - Random Forest (Optuna-tuned; SMOTE when class imbalance is severe)
  - Logistic Regression L2 and L1 — Moon 2020, Dempster 2019/2021
  - k-NN — Moon 2020, Li 2025

Metrics (competitor matrix): F1 weighted, balanced accuracy, MCC, AUROC,
sensitivity, specificity, precision, Cohen's κ.
Validation: leave-one-subject-out on trial- or patient-level Phase 1+2 features.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import optuna
from joblib import Parallel, delayed
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.dataset.label_policy import is_binary_task
from src.dataset.subject_split import assert_loso_fold_disjoint
from src.evaluation.multiclass_metrics import predict_multiclass
from src.features.phase12_features import (
    load_phase12_patient_matrix,
    load_phase12_trial_matrix,
    phase12_trial_bases,
)
from src.utils.reproducibility import get_pipeline_seed
from src.utils.progress import progress_bar

optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_SVM_RBF = "svm_rbf"
MODEL_RANDOM_FOREST = "random_forest"
MODEL_LOGISTIC_L2 = "logistic_regression_l2"
MODEL_LOGISTIC_L1 = "logistic_regression_l1"
MODEL_KNN = "knn"

DEFAULT_MODELS = (
    MODEL_SVM_RBF,
    MODEL_RANDOM_FOREST,
    MODEL_LOGISTIC_L2,
    MODEL_LOGISTIC_L1,
    MODEL_KNN,
)

LITERATURE_REFERENCES = {
    MODEL_SVM_RBF: "Moon 2020; Trabassi 2022; Li 2025; Prisco 2025",
    MODEL_RANDOM_FOREST: "Navita 2025; Moon 2020; Trabassi 2022",
    MODEL_LOGISTIC_L2: "Moon 2020; Dempster 2019/2021",
    MODEL_LOGISTIC_L1: "Moon 2020; Dempster 2019/2021",
    MODEL_KNN: "Moon 2020; Li 2025",
}


def _baseline_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("classical_baselines") or {}


def _severe_class_imbalance(y: np.ndarray, ratio_threshold: float) -> bool:
    counts = np.bincount(np.asarray(y, dtype=int))
    counts = counts[counts > 0]
    if len(counts) < 2:
        return False
    return float(counts.max() / counts.min()) >= ratio_threshold


def _smote_step(random_state: int):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return None
    return ("smote", SMOTE(random_state=random_state, k_neighbors=3))


def _base_steps(random_state: int) -> list[tuple[str, Any]]:
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]


def _build_estimator(
    model_name: str,
    config: dict[str, Any],
    *,
    y_train: np.ndarray | None = None,
    rf_params: dict[str, Any] | None = None,
) -> Pipeline:
    bcfg = _baseline_cfg(config)
    rs = get_pipeline_seed(config)
    steps = _base_steps(rs)

    if model_name == MODEL_SVM_RBF:
        svm_cfg = bcfg.get("svm_rbf") or {}
        clf = SVC(
            kernel="rbf",
            C=float(svm_cfg.get("C", 1.0)),
            gamma=str(svm_cfg.get("gamma", "scale")),
            probability=True,
            class_weight="balanced",
            random_state=rs,
        )
    elif model_name == MODEL_RANDOM_FOREST:
        if y_train is not None:
            smote_ratio = float((bcfg.get("random_forest") or {}).get("smote_imbalance_ratio_min", 2.5))
            smote = _smote_step(rs)
            if smote is not None and _severe_class_imbalance(y_train, smote_ratio):
                steps.append(smote)
        rf_cfg = rf_params or (bcfg.get("random_forest") or {}).get("params") or {}
        clf = RandomForestClassifier(
            n_estimators=int(rf_cfg.get("n_estimators", 200)),
            max_depth=rf_cfg.get("max_depth"),
            min_samples_split=int(rf_cfg.get("min_samples_split", 2)),
            min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 1)),
            max_features=rf_cfg.get("max_features", "sqrt"),
            class_weight="balanced",
            random_state=rs,
            n_jobs=-1,
        )
    elif model_name == MODEL_LOGISTIC_L2:
        clf = LogisticRegression(
            penalty="l2",
            C=float((bcfg.get("logistic_regression") or {}).get("C", 1.0)),
            class_weight="balanced",
            max_iter=int((bcfg.get("logistic_regression") or {}).get("max_iter", 2000)),
            random_state=rs,
        )
    elif model_name == MODEL_LOGISTIC_L1:
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=float((bcfg.get("logistic_regression") or {}).get("C", 1.0)),
            class_weight="balanced",
            max_iter=int((bcfg.get("logistic_regression") or {}).get("max_iter", 2000)),
            random_state=rs,
        )
    elif model_name == MODEL_KNN:
        knn_cfg = bcfg.get("knn") or {}
        clf = KNeighborsClassifier(
            n_neighbors=int(knn_cfg.get("n_neighbors", 5)),
            weights=str(knn_cfg.get("weights", "distance")),
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Unknown classical baseline model: {model_name}")

    steps.append(("clf", clf))
    return Pipeline(steps)


def _tune_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    bcfg = _baseline_cfg(config)
    rf_cfg = bcfg.get("random_forest") or {}
    n_trials = int(rf_cfg.get("optuna_trials", 30))
    rs = get_pipeline_seed(config)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        }
        pipe = _build_estimator(MODEL_RANDOM_FOREST, config, y_train=y_train, rf_params=params)
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=max(2, min(5, len(np.unique(y_train)))),
            shuffle=True,
            random_state=rs,
        )
        scores: list[float] = []
        for tr, va in skf.split(X_train, y_train):
            if len(np.unique(y_train[va])) < 2:
                continue
            pipe.fit(X_train[tr], y_train[tr])
            if is_binary_task(y_train, config):
                proba = pipe.predict_proba(X_train[va])
                score = proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
                scores.append(float(roc_auc_score(y_train[va], score)))
            else:
                proba, _ = predict_multiclass(pipe, X_train[va])
                labels = sorted(np.unique(y_train).tolist())
                scores.append(
                    float(
                        roc_auc_score(
                            y_train[va],
                            proba,
                            multi_class="ovr",
                            average="macro",
                            labels=labels,
                        )
                    )
                )
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=rs))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _classical_fold_one(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: dict[str, Any],
    *,
    rf_best_params: dict[str, Any] | None,
    meta_index: np.ndarray | None,
    binary: bool,
) -> tuple[list[int], list[int], np.ndarray, list[str], list[int]] | None:
    """Single LOSO fold for classical baselines (process-parallel worker)."""
    y_tr, y_te = y[train_idx], y[test_idx]
    if len(np.unique(y_tr)) < 2:
        return None
    assert_loso_fold_disjoint(
        groups[train_idx], groups[test_idx], held_out_subject=str(groups[test_idx][0])
    )

    if model_name == MODEL_RANDOM_FOREST and rf_best_params is None:
        fold_rf_params = _tune_random_forest(X[train_idx], y_tr, config)
    else:
        fold_rf_params = rf_best_params

    pipe = _build_estimator(
        model_name,
        config,
        y_train=y_tr,
        rf_params=fold_rf_params,
    )
    pipe.fit(X[train_idx], y_tr)

    if binary:
        proba = pipe.predict_proba(X[test_idx])
        scores = proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
        pred = (scores >= 0.5).astype(int)
        proba_out = np.column_stack([1.0 - scores, scores])
    else:
        proba_out, pred = predict_multiclass(pipe, X[test_idx])

    meta_idxs = meta_index[test_idx].tolist() if meta_index is not None else list(test_idx)
    return (
        y_te.tolist(),
        pred.tolist(),
        proba_out,
        groups[test_idx].astype(str).tolist(),
        meta_idxs,
    )


def _pool_oof_predictions(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    config: dict[str, Any],
    *,
    rf_best_params: dict[str, Any] | None = None,
    meta_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups))
    binary = is_binary_task(y, config)

    fold_results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(_classical_fold_one)(
            train_idx,
            test_idx,
            model_name,
            X,
            y,
            groups,
            config,
            rf_best_params=rf_best_params,
            meta_index=meta_index,
            binary=binary,
        )
        for train_idx, test_idx in progress_bar(
            splits, desc=f"  classical {model_name} LOSO", leave=False, unit="fold"
        )
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    y_proba_chunks: list[np.ndarray] = []
    participant_ids: list[str] = []
    trial_indices: list[int] = []
    for result in fold_results:
        if result is None:
            continue
        yt, yp, proba, pids, idxs = result
        y_true.extend(yt)
        y_pred.extend(yp)
        y_proba_chunks.append(proba)
        participant_ids.extend(pids)
        trial_indices.extend(idxs)

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    y_proba_arr = np.vstack(y_proba_chunks) if y_proba_chunks else np.empty((0, 0))
    pid_arr = np.asarray(participant_ids, dtype=str)
    idx_arr = np.asarray(trial_indices, dtype=int) if trial_indices else np.arange(len(y_true_arr))
    return y_true_arr, y_pred_arr, y_proba_arr, pid_arr, idx_arr


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    config: dict[str, Any],
) -> dict[str, float]:
    from src.evaluation.competitor_metrics import compute_discriminative_metrics

    return compute_discriminative_metrics(
        y_true, y_pred, y_proba=y_proba, config=config
    )


def _export_classical_oof(
    metrics_dir: Path,
    model_name: str,
    meta_df: pd.DataFrame,
    trial_indices: np.ndarray,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    participant_ids: np.ndarray,
    config: dict[str, Any],
) -> None:
    from src.evaluation.loso_oof_scores import binary_positive_score, build_oof_export_frame, save_model_oof

    if len(y_true) == 0:
        return
    sub = meta_df.iloc[trial_indices].reset_index(drop=True)
    if "risk_label" in sub.columns:
        y_bin = sub["risk_label"].astype(int).values
    else:
        y_bin = (np.asarray(y_true, dtype=int) > 0).astype(int)
    scores = binary_positive_score(y_proba, binary=True)
    trial_ids = (
        sub["trial_id"].astype(str).values
        if "trial_id" in sub.columns
        else np.asarray(trial_indices, dtype=str)
    )
    cohorts = sub["cohort"].astype(str).values if "cohort" in sub.columns else None
    frame = build_oof_export_frame(trial_ids, participant_ids, y_bin, scores, cohorts=cohorts)
    save_model_oof(metrics_dir, model_name, frame)


def _load_matrix(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    level = str(_baseline_cfg(config).get("evaluation_level", "trial")).lower()
    if level == "patient":
        return load_phase12_patient_matrix(config)
    return load_phase12_trial_matrix(config)


def run_classical_baselines(config: dict) -> pd.DataFrame:
    bcfg = _baseline_cfg(config)
    if not bcfg.get("enabled", True):
        logger.info("Classical baselines disabled in config")
        return pd.DataFrame()

    X, y, groups, feat_cols, meta_df = _load_matrix(config)
    models = list(bcfg.get("models") or DEFAULT_MODELS)
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    meta_indices = np.arange(len(meta_df))

    logger.info(
        "Classical baselines (Phase 1+2): {} models, {} samples, {} features",
        len(models),
        len(y),
        len(feat_cols),
    )

    rows: list[dict[str, Any]] = []

    for model_name in progress_bar(models, desc="Classical baselines", unit="model"):
        logger.info("Classical baseline LOSO: {}", model_name)
        y_true, y_pred, y_proba, oof_pids, oof_idx = _pool_oof_predictions(
            model_name,
            X,
            y,
            groups,
            config,
            meta_index=meta_indices,
        )
        metrics = _compute_metrics(y_true, y_pred, y_proba, config)
        if bcfg.get("export_oof_scores", True):
            _export_classical_oof(
                metrics_dir, model_name, meta_df, oof_idx, y_true, y_proba, oof_pids, config
            )
        rows.append(
            {
                "model": model_name,
                "paradigm": "competitor_paradigm_1",
                "feature_set": "phase1_phase2_handcrafted",
                "feature_groups": ", ".join(phase12_trial_bases(config)[:3]) + ", ...",
                "n_features": len(feat_cols),
                "n_samples": int(len(y_true)),
                "evaluation_level": str(bcfg.get("evaluation_level", "trial")),
                "validation": "loso_subject_grouped",
                "literature": LITERATURE_REFERENCES.get(model_name, ""),
                "smote_used": model_name == MODEL_RANDOM_FOREST and _smote_step(get_pipeline_seed(config)) is not None,
                **metrics,
            }
        )
        logger.info(
            "  {} — F1w={:.4f} MCC={:.4f} AUROC={:.4f} κ={:.4f}",
            model_name,
            metrics.get("f1_weighted", float("nan")),
            metrics.get("mcc", float("nan")),
            metrics.get("auroc", float("nan")),
            metrics.get("cohen_kappa", float("nan")),
        )

    df = pd.DataFrame(rows).sort_values("auroc", ascending=False, na_position="last")
    csv_path = metrics_dir / "classical_baseline_metrics.csv"
    df.to_csv(csv_path, index=False)
    _write_competitor_matrix_md(metrics_dir / "classical_baseline_metrics.md", df)
    (metrics_dir / "classical_baseline_summary.json").write_text(
        json.dumps(
            {
                "paradigm": "competitor_paradigm_1",
                "feature_set": "phase1_phase2_handcrafted",
                "models": models,
                "best_auroc_model": str(df.iloc[0]["model"]) if not df.empty else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Classical baseline metrics → {}", csv_path)
    return df


def _write_competitor_matrix_md(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Classical baselines — competitor paradigm 1",
        "",
        "Phase 1 + Phase 2 handcrafted features, LOSO OOF.",
        "",
        "| Model | F1 (w) | Bal. Acc. | MCC | AUROC | Sens. | Spec. | Prec. | κ | Literature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    display = {
        MODEL_SVM_RBF: "SVM (RBF)",
        MODEL_RANDOM_FOREST: "Random Forest (tuned)",
        MODEL_LOGISTIC_L2: "Logistic Regression (L2)",
        MODEL_LOGISTIC_L1: "Logistic Regression (L1)",
        MODEL_KNN: "k-NN",
    }
    for row in df.itertuples(index=False):
        name = display.get(str(row.model), str(row.model))
        lines.append(
            f"| {name} | {float(row.f1_weighted):.4f} | {float(row.balanced_accuracy):.4f} | "
            f"{float(getattr(row, 'mcc', float('nan'))):.4f} | {float(row.auroc):.4f} | "
            f"{float(getattr(row, 'sensitivity', float('nan'))):.4f} | "
            f"{float(getattr(row, 'specificity', float('nan'))):.4f} | "
            f"{float(getattr(row, 'precision', float('nan'))):.4f} | "
            f"{float(getattr(row, 'cohen_kappa', float('nan'))):.4f} | {row.literature} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
