"""
Compute overhead reporting — training & inference latency per competitor model.

Benchmark protocol (manuscript-facing):
  - One representative LOSO fold (first valid held-out participant).
  - Training time: wall-clock fit on train split (single fold).
  - Inference time: median of repeated predict/score passes on test split.
  - LOSO estimate: ``train_time_s * n_participants`` (full cross-validation budget).

Outputs ``compute_overhead_metrics.csv`` + ``compute_overhead_report.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import LeaveOneGroupOut

from src.dataset.subject_split import assert_loso_fold_disjoint
from src.evaluation.classical_baseline_evaluator import (
    DEFAULT_MODELS,
    MODEL_RANDOM_FOREST,
    _baseline_cfg,
    _build_estimator,
    _load_matrix,
)
from src.evaluation.compute_timing import device_label, host_info, time_callable, time_training
from src.evaluation.dl_baseline_evaluator import (
    MODEL_DEEP_CONV_LSTM,
    MODEL_INCEPTION_TIME,
    MODEL_MINIROCKET,
    MODEL_ROCKET,
    _dl_cfg,
    _trial_mean_features,
    _trial_window_records,
)
from src.evaluation.primary_endpoint import ENDPOINT_BILSTM_AE_ENSEMBLE
from src.evaluation.statistical_benchmark_evaluator import DISPLAY_NAMES
from src.features.rocket_features import MiniRocketTransform, RocketTransform
from src.models.bilstm_ae_scoring import (
    combine_ensemble_scores,
    fit_latent_one_class_models,
    load_voisard_trial_windows,
    score_latent_one_class,
    train_healthy_ae,
    trial_mean_scores,
)
from src.utils.progress import progress_bar, stage_spinner
from src.utils.reproducibility import get_pipeline_seed
from src.utils.torch_device import resolve_torch_device

DISPLAY = {
    **DISPLAY_NAMES,
    ENDPOINT_BILSTM_AE_ENSEMBLE: "BiLSTM-AE (GaitGuard)",
}


def _overhead_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("compute_overhead") or {}


def _select_loso_fold(
    groups: np.ndarray,
    cohorts: np.ndarray | None = None,
    *,
    min_healthy_train: int = 3,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    logo = LeaveOneGroupOut()
    y_dummy = np.zeros(len(groups), dtype=int)
    for train_idx, test_idx in logo.split(y_dummy, y_dummy, groups):
        held_out = str(groups[test_idx][0])
        assert_loso_fold_disjoint(groups[train_idx], groups[test_idx], held_out_subject=held_out)
        if cohorts is not None:
            healthy_train = int(np.sum(cohorts[train_idx] == "Healthy"))
            if healthy_train < min_healthy_train:
                continue
        return train_idx, test_idx, held_out
    return None


def _row(
    model: str,
    paradigm: str,
    *,
    train_time_s: float,
    infer_median_s: float,
    n_infer_units: int,
    infer_unit: str,
    n_folds: int,
    n_train: int,
    n_test: int,
    held_out_participant: str,
    batch_size: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    infer_ms = (infer_median_s / max(n_infer_units, 1)) * 1000.0
    row: dict[str, Any] = {
        "model": model,
        "display_name": DISPLAY.get(model, model),
        "paradigm": paradigm,
        "device": device_label(),
        "benchmark_protocol": "single_loso_fold_median_inference",
        "held_out_participant": held_out_participant,
        "n_loso_folds": n_folds,
        "n_train_samples": n_train,
        "n_test_samples": n_test,
        "train_time_s_per_fold": float(train_time_s),
        "train_time_s_loso_estimated": float(train_time_s * n_folds),
        "inference_median_s_per_pass": float(infer_median_s),
        "inference_n_units": int(n_infer_units),
        "inference_unit": infer_unit,
        "inference_ms_per_unit": float(infer_ms),
        "throughput_units_per_s": float(n_infer_units / infer_median_s) if infer_median_s > 0 else float("nan"),
        "batch_size": batch_size,
    }
    if extra:
        row.update(extra)
    return row


def _benchmark_classical(
    config: dict[str, Any],
    *,
    n_warmup: int,
    n_repeat: int,
    skip_rf_tuning: bool,
) -> list[dict[str, Any]]:
    X, y, groups, _, _ = _load_matrix(config)
    fold = _select_loso_fold(groups)
    if fold is None:
        return []
    train_idx, test_idx, held_out = fold
    n_folds = len(np.unique(groups))
    models = list(_baseline_cfg(config).get("models") or DEFAULT_MODELS)
    rows: list[dict[str, Any]] = []

    for model_name in progress_bar(models, desc="compute_overhead classical", unit="model"):
        rf_params = None
        if model_name == MODEL_RANDOM_FOREST and skip_rf_tuning:
            rf_params = (_baseline_cfg(config).get("random_forest") or {}).get("params") or {
                "n_estimators": 100,
                "max_depth": 10,
            }
        pipe = _build_estimator(model_name, config, y_train=y[train_idx], rf_params=rf_params)

        def fit() -> Any:
            pipe.fit(X[train_idx], y[train_idx])
            return pipe

        _, train_s = time_training(fit)

        def infer() -> np.ndarray:
            return pipe.predict_proba(X[test_idx])

        _, infer_s, _ = time_callable(infer, n_warmup=n_warmup, n_repeat=n_repeat)
        rows.append(
            _row(
                model_name,
                "classical_paradigm_1",
                train_time_s=train_s,
                infer_median_s=infer_s,
                n_infer_units=len(test_idx),
                infer_unit="trial",
                n_folds=n_folds,
                n_train=len(train_idx),
                n_test=len(test_idx),
                held_out_participant=held_out,
            )
        )
    return rows


def _benchmark_rocket_family(
    config: dict[str, Any],
    model_key: str,
    transform_cls: type,
    *,
    n_warmup: int,
    n_repeat: int,
) -> dict[str, Any] | None:
    ocfg = _overhead_cfg(config)
    bcfg = _dl_cfg(config)
    rk_cfg = (bcfg.get(model_key) or {}) or (config.get("features", {}).get("phase3_deep", {}).get(model_key) or {})
    n_kernels = int(rk_cfg.get("n_kernels", 10_000))
    max_fit = int(ocfg.get("max_fit_windows", bcfg.get("max_fit_windows", 20_000)))
    rs = get_pipeline_seed(config)

    trial_ids, groups, y, windows_by_trial = _trial_window_records(config)
    fold = _select_loso_fold(groups)
    if fold is None:
        return None
    train_idx, test_idx, held_out = fold
    n_folds = len(np.unique(groups))
    train_tids = [trial_ids[i] for i in train_idx]
    test_tids = [trial_ids[i] for i in test_idx]

    fit_windows = np.concatenate([windows_by_trial[tid] for tid in train_tids], axis=0)
    if len(fit_windows) > max_fit:
        rng = np.random.default_rng(rs)
        fit_windows = fit_windows[rng.choice(len(fit_windows), max_fit, replace=False)]

    def fit_transform() -> tuple[Any, np.ndarray, np.ndarray]:
        from sklearn.linear_model import RidgeClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        transform = transform_cls(n_kernels, seed=rs).fit(fit_windows)
        X_train = _trial_mean_features(train_tids, windows_by_trial, transform)
        X_test = _trial_mean_features(test_tids, windows_by_trial, transform)
        alpha = float((bcfg.get("ridge") or {}).get("alpha", 1.0))
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RidgeClassifier(alpha=alpha, class_weight="balanced")),
            ]
        )
        pipe.fit(X_train, y[train_idx])
        return pipe, X_test, transform

    (pipe, X_test, _), train_s = time_training(fit_transform)

    def infer() -> np.ndarray:
        return pipe.predict(X_test)

    _, infer_s, _ = time_callable(infer, n_warmup=n_warmup, n_repeat=n_repeat)
    return _row(
        model_key,
        "competitor_paradigm_2_dl",
        train_time_s=train_s,
        infer_median_s=infer_s,
        n_infer_units=len(test_tids),
        infer_unit="trial",
        n_folds=n_folds,
        n_train=len(train_tids),
        n_test=len(test_tids),
        held_out_participant=held_out,
        extra={"n_kernels": n_kernels},
    )


def _benchmark_bilstm_ae(config: dict[str, Any], *, n_warmup: int, n_repeat: int) -> dict[str, Any] | None:
    device = resolve_torch_device(config)
    rs = get_pipeline_seed(config)
    bundle = load_voisard_trial_windows(config, require_all_sensors=True)
    fold = _select_loso_fold(bundle.participant_ids, bundle.cohorts, min_healthy_train=3)
    if fold is None:
        return None
    train_idx, test_idx, held_out = fold
    train_tids = [bundle.trial_ids[i] for i in train_idx]
    test_tids = [bundle.trial_ids[i] for i in test_idx]
    healthy_train_tids = [
        bundle.trial_ids[i]
        for i in train_idx
        if bundle.cohorts[i] == "Healthy"
    ]

    healthy_windows = np.concatenate(
        [bundle.windows[tid] for tid in healthy_train_tids if tid in bundle.windows],
        axis=0,
    )
    ocfg = _overhead_cfg(config)
    max_fit = int(ocfg.get("max_fit_windows", 30_000))
    if len(healthy_windows) > max_fit:
        rng = np.random.default_rng(rs)
        healthy_windows = healthy_windows[rng.choice(len(healthy_windows), max_fit, replace=False)]

    def fit() -> tuple[Any, Any, Any]:
        model, norm = train_healthy_ae(
            healthy_windows, bundle.sensor_slices, config, device=device, checkpoint_path=None
        )
        _, lat_tr = trial_mean_scores(model, bundle, train_tids, norm, device=device)
        healthy_lat_rows = [
            lat_tr[i]
            for i, tid in enumerate(train_tids)
            if tid in healthy_train_tids and np.isfinite(lat_tr[i]).all()
        ]
        healthy_latent = np.vstack(healthy_lat_rows) if healthy_lat_rows else lat_tr
        oc_models = fit_latent_one_class_models(healthy_latent, random_state=rs)
        return model, norm, oc_models

    (model, norm, oc_models), train_s = time_training(fit)
    n_windows = sum(len(bundle.windows[tid]) for tid in test_tids)

    def infer() -> np.ndarray:
        from src.models.bilstm_ae_scoring import METHOD_AE_RECON, METHOD_IF_LATENT, METHOD_OCSVM_LATENT

        recon_te, lat_te = trial_mean_scores(model, bundle, test_tids, norm, device=device)
        if_te, svm_te = score_latent_one_class(oc_models, lat_te)
        test_methods = {
            METHOD_AE_RECON: recon_te,
            METHOD_IF_LATENT: if_te,
            METHOD_OCSVM_LATENT: svm_te,
        }
        recon_tr, lat_tr = trial_mean_scores(model, bundle, train_tids, norm, device=device)
        if_tr, svm_tr = score_latent_one_class(oc_models, lat_tr)
        healthy_train_mask = np.array([tid in healthy_train_tids for tid in train_tids])
        train_methods = {
            METHOD_AE_RECON: recon_tr,
            METHOD_IF_LATENT: if_tr,
            METHOD_OCSVM_LATENT: svm_tr,
        }
        ref = {k: train_methods[k][healthy_train_mask] for k in train_methods}
        return combine_ensemble_scores(test_methods, config, reference_scores=ref)

    _, infer_s, _ = time_callable(infer, n_warmup=n_warmup, n_repeat=n_repeat)
    n_folds = len(np.unique(bundle.participant_ids))
    batch_size = int(
        ((config.get("primary_model") or {}).get("bilstm_ae_ensemble") or {}).get("batch_size", 64)
    )
    return _row(
        ENDPOINT_BILSTM_AE_ENSEMBLE,
        "gaitguard_primary",
        train_time_s=train_s,
        infer_median_s=infer_s,
        n_infer_units=n_windows,
        infer_unit="window",
        n_folds=n_folds,
        n_train=len(train_tids),
        n_test=len(test_tids),
        held_out_participant=held_out,
        batch_size=batch_size,
        extra={"inference_ms_per_trial": float((infer_s / max(len(test_tids), 1)) * 1000.0)},
    )


def _benchmark_deep_classifier(
    config: dict[str, Any],
    model_name: str,
    *,
    n_warmup: int,
    n_repeat: int,
) -> dict[str, Any] | None:
    import copy

    from src.models.deep_trainer import DeepLearningPipeline

    ocfg = _overhead_cfg(config)
    cfg = copy.deepcopy(config)
    dl = dict(cfg.get("deep_learning") or {})
    dl["models"] = [model_name]
    max_epochs = ocfg.get("deep_max_epochs")
    if max_epochs is not None:
        dl["max_epochs"] = int(max_epochs)
    dl["loso_hyperparameter_tuning"] = {"enabled": False}
    cfg["deep_learning"] = dl

    pipeline = DeepLearningPipeline(cfg)
    trial_signals, meta = pipeline._load_trial_data()
    participants, n_channels = pipeline._build_participant_windows(trial_signals, meta)
    if n_channels == 0 or not participants:
        return None

    pids = sorted(participants.keys())
    test_pid = pids[0]
    train_pids = [p for p in pids if p != test_pid]
    if len(train_pids) < 2:
        return None

    test_data = participants[test_pid]
    X_test_raw = test_data["windows"]
    n_folds = len(pids)

    inner_train, inner_val = pipeline.split_inner_train_val_participants(
        train_pids,
        {pid: int(participants[pid]["label"]) for pid in train_pids},
        seed=pipeline.seed,
        val_fraction=0.1,
    )
    X_train_f_raw, y_train_f = pipeline.concat_participant_windows(inner_train, participants)
    X_val_raw, y_val = pipeline.concat_participant_windows(inner_val, participants)
    if len(X_train_f_raw) == 0 or len(X_val_raw) == 0:
        return None

    X_train_norm, X_test_norm = pipeline._normalize(X_train_f_raw, X_test_raw)
    _, X_val_norm = pipeline._normalize(X_train_f_raw, X_val_raw)
    n_classes = len(set(p["label"] for p in participants.values()))

    def fit() -> Any:
        from src.models.deep_models import build_deep_model

        trainer = pipeline._make_trainer()
        model = build_deep_model(model_name, n_channels, pipeline.window_len, n_classes)
        return trainer.train(model, X_train_norm, y_train_f, X_val_norm, y_val)

    model, train_s = time_training(fit)

    def infer() -> np.ndarray:
        trainer = pipeline._make_trainer()
        return trainer.predict_proba(model, X_test_norm)

    _, infer_s, _ = time_callable(infer, n_warmup=n_warmup, n_repeat=n_repeat)
    n_windows = len(X_test_norm)
    return _row(
        model_name,
        "competitor_paradigm_2_dl",
        train_time_s=train_s,
        infer_median_s=infer_s,
        n_infer_units=n_windows,
        infer_unit="window",
        n_folds=n_folds,
        n_train=len(X_train_norm),
        n_test=n_windows,
        held_out_participant=str(test_pid),
        batch_size=int(dl.get("batch_size", 64)),
        extra={"deep_max_epochs": int(dl.get("max_epochs", 0))},
    )


def run_compute_overhead_benchmark(config: dict) -> pd.DataFrame:
    ocfg = _overhead_cfg(config)
    if not ocfg.get("enabled", True):
        logger.info("Compute overhead benchmark disabled")
        return pd.DataFrame()

    n_warmup = int(ocfg.get("n_warmup", 1))
    n_repeat = int(ocfg.get("n_inference_repeats", 5))
    skip_rf_tuning = bool(ocfg.get("skip_rf_tuning", True))
    benchmark_deep = bool(ocfg.get("benchmark_deep_models", False))

    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    if ocfg.get("classical", True) and _baseline_cfg(config).get("enabled", True):
        try:
            rows.extend(
                _benchmark_classical(
                    config,
                    n_warmup=n_warmup,
                    n_repeat=n_repeat,
                    skip_rf_tuning=skip_rf_tuning,
                )
            )
        except Exception as exc:
            logger.warning("Classical overhead benchmark failed: {}", exc)

    if ocfg.get("dl_rocket", True) and _dl_cfg(config).get("enabled", True):
        for model_key, cls in progress_bar(
            ((MODEL_MINIROCKET, MiniRocketTransform), (MODEL_ROCKET, RocketTransform)),
            desc="compute_overhead rocket",
            unit="model",
        ):
            try:
                row = _benchmark_rocket_family(
                    config, model_key, cls, n_warmup=n_warmup, n_repeat=n_repeat
                )
                if row:
                    rows.append(row)
            except Exception as exc:
                logger.warning("Rocket overhead benchmark {} failed: {}", model_key, exc)

    if ocfg.get("bilstm_ae", True):
        with stage_spinner("compute_overhead bilstm_ae"):
            try:
                row = _benchmark_bilstm_ae(config, n_warmup=n_warmup, n_repeat=n_repeat)
                if row:
                    rows.append(row)
            except Exception as exc:
                logger.warning("BiLSTM-AE overhead benchmark failed: {}", exc)

    if benchmark_deep and _dl_cfg(config).get("enabled", True):
        for model_name in progress_bar(
            (MODEL_INCEPTION_TIME, MODEL_DEEP_CONV_LSTM),
            desc="compute_overhead deep",
            unit="model",
        ):
            try:
                row = _benchmark_deep_classifier(
                    config, model_name, n_warmup=n_warmup, n_repeat=n_repeat
                )
                if row:
                    rows.append(row)
            except Exception as exc:
                logger.warning("Deep overhead benchmark {} failed: {}", model_name, exc)

    if not rows:
        logger.warning("No compute overhead rows collected — check data paths and stage prerequisites")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("train_time_s_loso_estimated", ascending=True)
    csv_path = metrics_dir / "compute_overhead_metrics.csv"
    df.to_csv(csv_path, index=False)

    summary = {
        "host": host_info(),
        "benchmark_protocol": (
            "Single representative LOSO fold; median inference over repeated passes; "
            "LOSO training budget = per_fold_train_s × n_participants."
        ),
        "n_warmup": n_warmup,
        "n_inference_repeats": n_repeat,
        "benchmark_deep_models": benchmark_deep,
        "n_models": int(len(df)),
        "fastest_inference_ms_per_unit": float(df["inference_ms_per_unit"].min()),
        "slowest_inference_ms_per_unit": float(df["inference_ms_per_unit"].max()),
        "models": df["model"].tolist(),
    }
    (metrics_dir / "compute_overhead_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_markdown(metrics_dir / "compute_overhead_report.md", df, summary)
    logger.info("Compute overhead metrics → {} ({} models)", csv_path, len(df))
    return df


def _write_markdown(path: Path, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    host = summary.get("host", {})
    lines = [
        "# Compute overhead — training & inference latency",
        "",
        f"**Device:** {host.get('device', '—')}  ",
        f"**Platform:** {host.get('platform', '—')}",
        "",
        "Protocol: one LOSO fold; training time = single-fold fit; inference = median of "
        f"{summary.get('n_inference_repeats', 5)} repeated scoring passes. "
        "LOSO training estimate = per-fold × number of participants.",
        "",
        "| Model | Paradigm | Train / fold (s) | LOSO train est. (s) | Infer. (ms/unit) | Unit | Throughput (/s) |",
        "|---|---|---:|---:|---:|---|---:|",
    ]
    for row in df.itertuples(index=False):
        lines.append(
            f"| {row.display_name} | {row.paradigm} | {float(row.train_time_s_per_fold):.3f} | "
            f"{float(row.train_time_s_loso_estimated):.1f} | {float(row.inference_ms_per_unit):.3f} | "
            f"{row.inference_unit} | {float(row.throughput_units_per_s):.1f} |"
        )
    if summary.get("benchmark_deep_models") is False:
        lines.extend(
            [
                "",
                "_Deep classifiers (InceptionTime, DeepConvLSTM) omitted — set "
                "`compute_overhead.benchmark_deep_models: true` to include (uses reduced epochs)._",
            ]
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
