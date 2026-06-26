"""
Deep learning baselines — competitor paradigm 2 (raw IMU windows, LOSO).

Run order (increasing compute):
  1. MINIROCKET + Ridge (Dempster 2021)
  2. ROCKET + Ridge (Dempster 2019, 10k kernels)
  3. InceptionTime (Fawaz 2020)
  4. DeepConvLSTM (Ordóñez & Roggen 2016)

Outputs ``dl_baseline_metrics.csv`` + ``dl_competitor_matrix.md`` including BiLSTM-AE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset.label_policy import is_binary_task
from src.dataset.subject_split import assert_loso_fold_disjoint
from src.evaluation.competitor_metrics import compute_discriminative_metrics, nan_discriminative_metrics
from src.evaluation.multiclass_metrics import predict_multiclass
from src.features.rocket_features import MiniRocketTransform, RocketTransform
from src.models.deep_models import DEEP_MODEL_REGISTRY, trial_to_tensor
from src.models.deep_trainer import DeepLearningPipeline
from src.preprocessing.windowing import parse_window_spec, window_single_trial
from src.utils.reproducibility import get_pipeline_seed
from src.utils.progress import progress_bar

MODEL_MINIROCKET = "minirocket"
MODEL_ROCKET = "rocket"
MODEL_INCEPTION_TIME = "inception_time"
MODEL_DEEP_CONV_LSTM = "deep_conv_lstm"
MODEL_BILSTM_AE = "bilstm_ae_ensemble"

DL_BASELINE_ORDER = (
    MODEL_MINIROCKET,
    MODEL_ROCKET,
    MODEL_INCEPTION_TIME,
    MODEL_DEEP_CONV_LSTM,
)

LITERATURE = {
    MODEL_MINIROCKET: "Dempster 2021",
    MODEL_ROCKET: "Dempster 2019",
    MODEL_INCEPTION_TIME: "Ismail Fawaz 2020",
    MODEL_DEEP_CONV_LSTM: "Ordóñez & Roggen 2016",
    MODEL_BILSTM_AE: "GaitGuard (yours)",
}

DISPLAY_NAMES = {
    MODEL_MINIROCKET: "MINIROCKET",
    MODEL_ROCKET: "ROCKET",
    MODEL_INCEPTION_TIME: "InceptionTime",
    MODEL_DEEP_CONV_LSTM: "DeepConvLSTM",
    MODEL_BILSTM_AE: "BiLSTM-AE",
}


class _NullProgress:
    def set_postfix(self, **kwargs: Any) -> None:
        pass

    def update(self, n: int = 1) -> None:
        pass


def _dl_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("dl_baselines") or {}


def _trial_window_records(config: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    processed = Path(config["paths"]["processed_data"])
    signals_dir = processed / "signals_clean"
    meta = pd.read_csv(processed / "trial_metadata.csv")
    meta = meta[~meta["trial_id"].astype(str).str.startswith("daphnet_")].reset_index(drop=True)
    spec = parse_window_spec(config)
    sensor_positions = config["dataset"]["sensor_positions"]
    label_mode = str(config.get("dataset", {}).get("label_mode", "multiclass")).lower()

    trial_ids: list[str] = []
    groups: list[str] = []
    labels: list[int] = []
    windows_by_trial: dict[str, np.ndarray] = {}

    for _, row in meta.iterrows():
        tid = str(row["trial_id"])
        arr = trial_to_tensor(tid, signals_dir, sensor_positions, require_all_sensors=True)
        if arr is None or arr.shape[1] < spec.window_len:
            continue
        wins = window_single_trial(arr, spec)
        if len(wins) == 0:
            continue
        trial_ids.append(tid)
        groups.append(str(row["participant_id"]))
        if label_mode == "multiclass" and "multiclass_label" in row:
            labels.append(int(row["multiclass_label"]))
        else:
            labels.append(int(row["risk_label"]))
        windows_by_trial[tid] = wins

    if not trial_ids:
        raise RuntimeError("No trial windows for DL baselines")

    return (
        trial_ids,
        np.asarray(groups),
        np.asarray(labels, dtype=int),
        windows_by_trial,
    )


def _trial_mean_features(
    trial_ids: list[str],
    windows_by_trial: dict[str, np.ndarray],
    transform: RocketTransform | MiniRocketTransform,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for tid in trial_ids:
        wins = windows_by_trial[tid]
        rows.append(transform.transform(wins).mean(axis=0))
    return np.vstack(rows)


def _loso_rocket_ridge(
    model_key: str,
    transform_cls: type,
    config: dict[str, Any],
    *,
    export_oof: bool = True,
) -> dict[str, float]:
    bcfg = _dl_cfg(config)
    rk_cfg = (bcfg.get(model_key) or {}) or (config.get("features", {}).get("phase3_deep", {}).get(model_key) or {})
    n_kernels = int(rk_cfg.get("n_kernels", 10_000))
    max_fit = int(bcfg.get("max_fit_windows", 20_000))
    rs = get_pipeline_seed(config)

    trial_ids, groups, y, windows_by_trial = _trial_window_records(config)
    tid_to_idx = {tid: i for i, tid in enumerate(trial_ids)}
    X_all = np.zeros((len(trial_ids), 1), dtype=np.float32)

    logo = LeaveOneGroupOut()
    oof_proba: list[np.ndarray] = []
    oof_true: list[int] = []
    oof_pred: list[int] = []
    oof_tids: list[str] = []
    oof_pids: list[str] = []

    for train_idx, test_idx in logo.split(X_all, y, groups):
        train_tids = [trial_ids[i] for i in train_idx]
        test_tids = [trial_ids[i] for i in test_idx]
        assert_loso_fold_disjoint(groups[train_idx], groups[test_idx])

        fit_windows = np.concatenate([windows_by_trial[tid] for tid in train_tids], axis=0)
        if len(fit_windows) > max_fit:
            rng = np.random.default_rng(rs)
            fit_windows = fit_windows[rng.choice(len(fit_windows), max_fit, replace=False)]

        transform = transform_cls(n_kernels, seed=rs).fit(fit_windows)
        X_train = _trial_mean_features(train_tids, windows_by_trial, transform)
        X_test = _trial_mean_features(test_tids, windows_by_trial, transform)
        y_train = y[train_idx]

        if len(np.unique(y_train)) < 2:
            continue

        alpha = float((bcfg.get("ridge") or {}).get("alpha", 1.0))
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RidgeClassifier(alpha=alpha, class_weight="balanced")),
            ]
        )
        pipe.fit(X_train, y_train)

        if is_binary_task(y, config):
            proba = pipe.predict_proba(X_test)
            scores = proba[:, 1] if proba.shape[1] > 1 else proba.ravel()
            pred = (scores >= 0.5).astype(int)
            proba_full = np.column_stack([1.0 - scores, scores])
        else:
            proba_full, pred = predict_multiclass(pipe, X_test)

        for i, tid in enumerate(test_tids):
            oof_true.append(int(y[tid_to_idx[tid]]))
            oof_pred.append(int(pred[i]))
            oof_proba.append(proba_full[i])
            oof_tids.append(tid)
            oof_pids.append(groups[tid_to_idx[tid]])

    y_true = np.asarray(oof_true, dtype=int)
    y_pred = np.asarray(oof_pred, dtype=int)
    y_proba = np.vstack(oof_proba) if oof_proba else np.empty((0, 0))
    metrics = compute_discriminative_metrics(y_true, y_pred, y_proba=y_proba, config=config)
    if export_oof and len(y_true):
        _export_dl_oof(config, model_key, oof_tids, oof_pids, y_true, y_proba)
    return metrics


def _export_dl_oof(
    config: dict[str, Any],
    model_name: str,
    trial_ids: list[str],
    participant_ids: list[str],
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> None:
    from src.evaluation.loso_oof_scores import binary_positive_score, build_oof_export_frame, save_model_oof

    metrics_dir = Path(config["paths"]["metrics"])
    y_bin = (np.asarray(y_true, dtype=int) > 0).astype(int)
    if is_binary_task(y_true, config):
        y_bin = np.asarray(y_true, dtype=int)
    scores = binary_positive_score(y_proba, binary=is_binary_task(y_true, config))
    frame = build_oof_export_frame(
        trial_ids,
        np.asarray(participant_ids, dtype=str),
        y_bin,
        scores,
    )
    save_model_oof(metrics_dir, model_name, frame)


def _loso_deep_classifier(model_name: str, config: dict[str, Any], *, export_oof: bool = True) -> dict[str, float]:
    if model_name not in DEEP_MODEL_REGISTRY:
        raise ValueError(f"Unknown DL model: {model_name}")

    cfg = copy.deepcopy(config)
    dl = dict(cfg.get("deep_learning") or {})
    dl["models"] = [model_name]
    override = (_dl_cfg(config).get("deep") or {})
    if override.get("max_epochs") is not None:
        dl["max_epochs"] = int(override["max_epochs"])
    if override.get("early_stopping_patience") is not None:
        dl["early_stopping_patience"] = int(override["early_stopping_patience"])
    if override.get("loso_hyperparameter_tuning") is not None:
        dl["loso_hyperparameter_tuning"] = override["loso_hyperparameter_tuning"]
    else:
        dl["loso_hyperparameter_tuning"] = {"enabled": False}
    cfg["deep_learning"] = dl

    pipeline = DeepLearningPipeline(cfg)
    trial_signals, meta = pipeline._load_trial_data()
    participants, n_channels = pipeline._build_participant_windows(trial_signals, meta)
    if n_channels == 0:
        return nan_discriminative_metrics()

    result = pipeline._loso_evaluate(model_name, participants, n_channels, _NullProgress())
    y_true = np.asarray(result.get("y_true", []), dtype=int)
    y_pred = np.asarray(result.get("y_pred", []), dtype=int)
    y_proba = np.asarray(result.get("y_proba_full", []), dtype=float)
    if len(y_true) == 0:
        return nan_discriminative_metrics()
    metrics = compute_discriminative_metrics(y_true, y_pred, y_proba=y_proba, config=config)
    if export_oof:
        pids = result.get("participant_ids", [])
        if pids and len(pids) == len(y_true):
            _export_dl_oof(
                config,
                model_name,
                [str(p) for p in pids],
                [str(p) for p in pids],
                y_true,
                y_proba,
            )
    return metrics


def _load_bilstm_ae_row(metrics_dir: Path) -> dict[str, Any] | None:
    path = metrics_dir / "bilstm_ae_anomaly_metrics.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    ens = df[df["method"] == MODEL_BILSTM_AE]
    if ens.empty:
        ens = df.sort_values("auc", ascending=False).head(1)
    if ens.empty:
        return None
    r = ens.iloc[0]
    return {
        "model": MODEL_BILSTM_AE,
        "f1_weighted": float(r.get("f1_weighted", r.get("f1", float("nan")))),
        "balanced_accuracy": float(r.get("balanced_accuracy", float("nan"))),
        "mcc": float(r.get("mcc", float("nan"))),
        "auroc": float(r.get("auc", float("nan"))),
        "sensitivity": float(r.get("sensitivity", float("nan"))),
        "specificity": float(r.get("specificity", float("nan"))),
        "precision": float(r.get("precision", float("nan"))),
        "cohen_kappa": float(r.get("cohen_kappa", float("nan"))),
        "n_samples": int(r.get("n_trials_scored", 0)),
        "source": "bilstm_ae_anomaly_metrics.csv",
    }


def run_dl_baselines(config: dict) -> pd.DataFrame:
    bcfg = _dl_cfg(config)
    if not bcfg.get("enabled", True):
        logger.info("DL baselines disabled")
        return pd.DataFrame()

    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models = list(bcfg.get("models") or DL_BASELINE_ORDER)

    rows: list[dict[str, Any]] = []
    for model_name in progress_bar(models, desc="DL baselines", unit="model"):
        if model_name not in DL_BASELINE_ORDER:
            logger.warning("Skipping unknown DL baseline {}", model_name)
            continue
        logger.info("DL baseline LOSO: {}", model_name)
        try:
            if model_name == MODEL_MINIROCKET:
                metrics = _loso_rocket_ridge(model_name, MiniRocketTransform, config)
            elif model_name == MODEL_ROCKET:
                metrics = _loso_rocket_ridge(model_name, RocketTransform, config)
            else:
                metrics = _loso_deep_classifier(model_name, config)
        except Exception as exc:
            logger.error("DL baseline {} failed: {}", model_name, exc)
            metrics = nan_discriminative_metrics()

        rows.append(
            {
                "model": model_name,
                "display_name": DISPLAY_NAMES.get(model_name, model_name),
                "paradigm": "competitor_paradigm_2_dl",
                "literature": LITERATURE.get(model_name, ""),
                "n_kernels": int((bcfg.get(model_name) or {}).get("n_kernels", 10_000))
                if model_name in (MODEL_ROCKET, MODEL_MINIROCKET)
                else None,
                "validation": "loso_subject_grouped",
                **metrics,
            }
        )
        logger.info(
            "  {} — F1w={:.4f} MCC={:.4f} AUROC={:.4f}",
            DISPLAY_NAMES.get(model_name, model_name),
            metrics.get("f1_weighted", float("nan")),
            metrics.get("mcc", float("nan")),
            metrics.get("auroc", float("nan")),
        )

    bilstm = _load_bilstm_ae_row(metrics_dir)
    if bilstm and bcfg.get("include_bilstm_ae", True):
        rows.append(
            {
                "model": MODEL_BILSTM_AE,
                "display_name": DISPLAY_NAMES[MODEL_BILSTM_AE],
                "paradigm": "gaitguard_primary",
                "literature": LITERATURE[MODEL_BILSTM_AE],
                "n_kernels": None,
                "validation": "loso_trial_oof_healthy_reference",
                "source": bilstm.get("source"),
                **{k: bilstm[k] for k in (
                    "f1_weighted", "balanced_accuracy", "mcc", "auroc",
                    "sensitivity", "specificity", "precision", "cohen_kappa",
                ) if k in bilstm},
                "f1": bilstm.get("f1_weighted", float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = metrics_dir / "dl_baseline_metrics.csv"
    df.to_csv(csv_path, index=False)
    _write_competitor_matrix(metrics_dir / "dl_competitor_matrix.md", df)
    (metrics_dir / "dl_baseline_summary.json").write_text(
        json.dumps(
            {
                "run_order": list(DL_BASELINE_ORDER),
                "models_evaluated": [r["model"] for r in rows],
                "best_auroc": str(df.sort_values("auroc", ascending=False).iloc[0]["model"])
                if not df.empty and df["auroc"].notna().any()
                else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("DL baseline metrics → {}", csv_path)
    return df


def _write_competitor_matrix(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "# DL competitor matrix",
        "",
        "Raw-IMU deep baselines (LOSO) + GaitGuard BiLSTM-AE primary endpoint.",
        "",
        "| Model | Reference | F1 (w) | Bal. Acc. | MCC | AUROC | Sens. | Spec. | Prec. | κ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        name = getattr(row, "display_name", str(row.model))
        lit = getattr(row, "literature", "")
        lines.append(
            f"| {name} | {lit} | {float(row.f1_weighted):.4f} | {float(row.balanced_accuracy):.4f} | "
            f"{float(getattr(row, 'mcc', float('nan'))):.4f} | {float(row.auroc):.4f} | "
            f"{float(getattr(row, 'sensitivity', float('nan'))):.4f} | "
            f"{float(getattr(row, 'specificity', float('nan'))):.4f} | "
            f"{float(getattr(row, 'precision', float('nan'))):.4f} | "
            f"{float(getattr(row, 'cohen_kappa', float('nan'))):.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
