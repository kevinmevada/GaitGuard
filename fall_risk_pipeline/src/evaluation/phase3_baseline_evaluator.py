"""
Phase 3 baseline comparison: ROCKET / MINIROCKET + shallow classifier LOSO.

Dempster 2019/2021 — 10,000-kernel transforms as competitor matrix baselines.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.phase3_deep import fit_phase3_models, extract_phase3_trial_features
from src.features.rocket_features import MiniRocketTransform, RocketTransform


def _trial_rocket_matrix(
    trial_ids: list[str],
    signals_dir: Path,
    sensor_positions: list[str],
    transform: RocketTransform | MiniRocketTransform,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-trial mean ROCKET/MINIROCKET vectors."""
    bundle = fit_phase3_models(config)
    X_rows: list[np.ndarray] = []
    y_list: list[int] = []
    groups: list[str] = []
    meta = pd.read_csv(Path(config["paths"]["processed_data"]) / "trial_metadata.csv")
    meta = meta.set_index("trial_id")

    for tid in trial_ids:
        if tid not in meta.index:
            continue
        feats = extract_phase3_trial_features(
            tid, signals_dir, sensor_positions, bundle, config
        )
        prefix = "rk_f" if isinstance(transform, RocketTransform) else "mr_f"
        vec = np.array([feats[k] for k in sorted(feats) if k.startswith(prefix)], dtype=float)
        if vec.size == 0:
            continue
        X_rows.append(vec)
        row = meta.loc[tid]
        y_list.append(int(row["risk_label"]))
        groups.append(str(row["participant_id"]))

    return np.vstack(X_rows), np.asarray(y_list), np.asarray(groups)


def run_phase3_rocket_baselines(config: dict[str, Any]) -> dict[str, Any]:
    """
    LOSO RandomForest on truncated ROCKET/MINIROCKET trial features (binary high-risk).
    """
    p3 = (config.get("features") or {}).get("phase3_deep") or {}
    if not p3.get("run_baseline_eval", False):
        return {}

    processed = Path(config["paths"]["processed_data"])
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    signals_dir = processed / "signals_clean"
    meta = pd.read_csv(processed / "trial_metadata.csv")
    sensor_positions = config["dataset"]["sensor_positions"]
    trial_ids = meta["trial_id"].tolist()

    results: dict[str, Any] = {"baselines": []}
    logo = LeaveOneGroupOut()

    for name, cls in (("rocket", RocketTransform), ("minirocket", MiniRocketTransform)):
        rk_cfg = p3.get(name) or {}
        if not rk_cfg.get("enabled", True):
            continue
        n_k = int(rk_cfg.get("n_kernels", 10_000))
        ckpt = Path(config["paths"]["checkpoints"]) / f"phase3_{name}_kernels.npz"
        if ckpt.is_file():
            transform = cls.load(ckpt)
        else:
            from src.features.phase3_deep import _collect_train_fit_windows

            X_fit, _ = _collect_train_fit_windows(config, max_windows=20_000)
            transform = cls(n_k, seed=42).fit(X_fit)
            transform.save(ckpt)

        X, y, groups = _trial_rocket_matrix(
            trial_ids, signals_dir, sensor_positions, transform, config
        )
        if len(np.unique(y)) < 2:
            continue

        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ]
        )
        oof = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in logo.split(X, y, groups):
            pipe.fit(X[train_idx], y[train_idx])
            proba = pipe.predict_proba(X[test_idx])
            if proba.shape[1] == 2:
                oof[test_idx] = proba[:, 1]
            else:
                oof[test_idx] = proba[:, -1]

        auc = float(roc_auc_score(y, oof))
        row = {"model": name, "n_kernels": n_k, "auc": auc, "n_trials": int(len(y))}
        results["baselines"].append(row)
        logger.info("Phase 3 baseline {} LOSO AUC = {:.4f}", name, auc)

    out_path = metrics_dir / "phase3_rocket_baselines.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results
