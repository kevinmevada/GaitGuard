"""
BiLSTM-AE scoring + 3-method one-class ensemble (primary model).

Methods (all fit on Healthy train-fold windows only):
  1. AE reconstruction error (per-sensor MSE; LB slice for DAPHNET)
  2. Isolation Forest on pooled latent activations h_t
  3. One-class SVM boundary distance on latent activations
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.features.phase3_deep import compute_sensor_slices
from src.models.anomaly_scoring import normalise_scores
from src.models.bilstm_autoencoder import BiLSTMAutoencoder, SensorChannelSlice, train_bilstm_autoencoder
from src.models.deep_models import trial_to_tensor
from src.preprocessing.fold_normalization import PerChannelZNorm, fit_per_channel_znorm
from src.preprocessing.windowing import parse_window_spec, window_single_trial

METHOD_AE_RECON = "ae_reconstruction"
METHOD_IF_LATENT = "isolation_forest_latent"
METHOD_OCSVM_LATENT = "one_class_svm_latent"
METHOD_ENSEMBLE = "bilstm_ae_ensemble"

ENSEMBLE_METHODS = (METHOD_AE_RECON, METHOD_IF_LATENT, METHOD_OCSVM_LATENT)

# Manuscript sensor ablation: 4 > 2 > 1 in-distribution; 4-sensor train → DAPHNET LB transfer.
SENSOR_ABLATION_CONFIGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "4_sensor": ("HE+LB+LF+RF", ("head", "lower_back", "left_foot", "right_foot")),
    "2_sensor_he_lb": ("HE+LB", ("head", "lower_back")),
    "1_sensor_lb": ("LB", ("lower_back",)),
}


@dataclass
class TrialWindowBundle:
    trial_ids: list[str]
    participant_ids: np.ndarray
    cohorts: np.ndarray
    windows: dict[str, np.ndarray]
    sensor_slices: list[SensorChannelSlice]
    n_channels: int
    window_len: int


@dataclass
class LatentOneClassModels:
    latent_scaler: StandardScaler
    isolation_forest: IsolationForest
    one_class_svm: OneClassSVM


def _primary_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return (config.get("primary_model") or {}).get("bilstm_ae_ensemble") or {}


def ensemble_weights(config: dict[str, Any]) -> dict[str, float]:
    p = _primary_cfg(config)
    w = {
        METHOD_AE_RECON: float(p.get("ae_reconstruction_weight", 0.40)),
        METHOD_IF_LATENT: float(p.get("isolation_forest_latent_weight", 0.33)),
        METHOD_OCSVM_LATENT: float(p.get("one_class_svm_latent_weight", 0.27)),
    }
    total = sum(w.values())
    if total <= 0:
        return {k: 1.0 / len(w) for k in w}
    return {k: v / total for k, v in w.items()}


def mask_inactive_sensors(
    windows: np.ndarray,
    sensor_slices: list[SensorChannelSlice],
    active_sensors: tuple[str, ...],
) -> np.ndarray:
    """Zero channel blocks for sensors outside *active_sensors* (full layout preserved)."""
    active = set(active_sensors)
    out = windows.copy()
    for sl in sensor_slices:
        if sl.name not in active:
            out[:, sl.start : sl.end, :] = 0.0
    return out


def apply_sensor_mask_to_bundle(
    bundle: TrialWindowBundle,
    active_sensors: tuple[str, ...],
) -> TrialWindowBundle:
    masked = {
        tid: mask_inactive_sensors(wins, bundle.sensor_slices, active_sensors)
        for tid, wins in bundle.windows.items()
    }
    return TrialWindowBundle(
        trial_ids=bundle.trial_ids,
        participant_ids=bundle.participant_ids,
        cohorts=bundle.cohorts,
        windows=masked,
        sensor_slices=bundle.sensor_slices,
        n_channels=bundle.n_channels,
        window_len=bundle.window_len,
    )


def load_voisard_trial_windows(
    config: dict[str, Any],
    *,
    require_all_sensors: bool = True,
    active_sensors: tuple[str, ...] | None = None,
) -> TrialWindowBundle:
    processed = Path(config["paths"]["processed_data"])
    signals_dir = processed / "signals_clean"
    meta = pd.read_csv(processed / "trial_metadata.csv")
    meta = meta[~meta["trial_id"].astype(str).str.startswith("daphnet_")].reset_index(drop=True)
    spec = parse_window_spec(config)
    sensor_positions = config["dataset"]["sensor_positions"]

    windows: dict[str, np.ndarray] = {}
    trial_ids: list[str] = []
    ref_slices: list[SensorChannelSlice] = []

    for _, row in meta.iterrows():
        tid = str(row["trial_id"])
        arr = trial_to_tensor(
            tid, signals_dir, sensor_positions, require_all_sensors=require_all_sensors
        )
        if arr is None or arr.shape[1] < spec.window_len:
            continue
        if not ref_slices:
            ref_slices = compute_sensor_slices(tid, signals_dir, sensor_positions)
        wins = window_single_trial(arr, spec)
        if len(wins) == 0:
            continue
        windows[tid] = wins
        trial_ids.append(tid)

    if not windows:
        raise RuntimeError("No Voisard trial windows for BiLSTM-AE ensemble")

    sub = meta.set_index("trial_id").loc[trial_ids]
    n_ch = next(iter(windows.values())).shape[1]
    bundle = TrialWindowBundle(
        trial_ids=trial_ids,
        participant_ids=sub["participant_id"].astype(str).values,
        cohorts=sub["cohort"].astype(str).values,
        windows=windows,
        sensor_slices=ref_slices,
        n_channels=n_ch,
        window_len=spec.window_len,
    )
    if active_sensors is not None:
        bundle = apply_sensor_mask_to_bundle(bundle, active_sensors)
    return bundle


def lb_slice_from_slices(slices: list[SensorChannelSlice]) -> SensorChannelSlice | None:
    for sl in slices:
        if sl.name == "lower_back":
            return sl
    return None


def score_windows(
    model: BiLSTMAutoencoder,
    windows: np.ndarray,
    norm: PerChannelZNorm,
    *,
    device: torch.device,
    lb_slice: SensorChannelSlice | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    if len(windows) == 0:
        return np.array([]), np.empty((0, model.latent_dim), dtype=np.float32)

    X = norm.transform(windows)
    recon_errs: list[float] = []
    latents: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.tensor(X[start : start + batch_size], dtype=torch.float32, device=device)
            recon, h = model(batch)
            per_mse = model.per_sensor_mse(batch, recon)
            if lb_slice is not None and lb_slice.name in per_mse:
                err = per_mse[lb_slice.name]
            else:
                err = per_mse["total"]
            recon_errs.extend(err.cpu().numpy().tolist())
            latents.append(h.mean(dim=1).cpu().numpy())

    return np.asarray(recon_errs, dtype=np.float32), np.vstack(latents)


def trial_mean_scores(
    model: BiLSTMAutoencoder,
    bundle: TrialWindowBundle,
    trial_ids: list[str],
    norm: PerChannelZNorm,
    *,
    device: torch.device,
    lb_slice: SensorChannelSlice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    recon_list: list[float] = []
    latent_list: list[np.ndarray] = []
    for tid in trial_ids:
        wins = bundle.windows.get(tid)
        if wins is None or len(wins) == 0:
            recon_list.append(float("nan"))
            latent_list.append(np.full(model.latent_dim, np.nan))
            continue
        r, lat = score_windows(model, wins, norm, device=device, lb_slice=lb_slice)
        recon_list.append(float(np.nanmean(r)))
        latent_list.append(np.nanmean(lat, axis=0))
    return np.asarray(recon_list, dtype=np.float32), np.vstack(latent_list)


def fit_latent_one_class_models(latent_train: np.ndarray, *, random_state: int) -> LatentOneClassModels:
    scaler = StandardScaler()
    Z = scaler.fit_transform(latent_train)
    iso = IsolationForest(contamination=0.1, random_state=random_state, n_estimators=100)
    iso.fit(Z)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
    svm.fit(Z)
    return LatentOneClassModels(latent_scaler=scaler, isolation_forest=iso, one_class_svm=svm)


def score_latent_one_class(
    models: LatentOneClassModels,
    latent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    Z = models.latent_scaler.transform(latent)
    return (
        np.asarray(-models.isolation_forest.decision_function(Z), dtype=np.float32),
        np.asarray(-models.one_class_svm.decision_function(Z), dtype=np.float32),
    )


def train_healthy_ae(
    healthy_windows: np.ndarray,
    sensor_slices: list[SensorChannelSlice],
    config: dict[str, Any],
    *,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> tuple[BiLSTMAutoencoder, PerChannelZNorm]:
    norm = fit_per_channel_znorm(healthy_windows)
    X_norm = norm.transform(healthy_windows)
    ckpt = checkpoint_path or Path(config["paths"]["checkpoints"]) / "_tmp_ae.pt"
    model = train_bilstm_autoencoder(
        X_norm,
        sensor_slices=sensor_slices,
        config=config,
        checkpoint_path=ckpt,
    )
    model.to(device)
    model.eval()
    return model, norm


def combine_ensemble_scores(
    method_scores: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    reference_scores: dict[str, np.ndarray],
) -> np.ndarray:
    weights = ensemble_weights(config)
    layers = [
        normalise_scores(method_scores[k], reference_scores[k]) * weights[k]
        for k in ENSEMBLE_METHODS
    ]
    return np.sum(np.stack(layers, axis=0), axis=0)


def build_fold_trial_scores(
    bundle: TrialWindowBundle,
    train_trial_ids: list[str],
    test_trial_ids: list[str],
    healthy_train_trial_ids: list[str],
    config: dict[str, Any],
    *,
    device: torch.device,
    random_state: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    healthy_windows = np.concatenate(
        [bundle.windows[tid] for tid in healthy_train_trial_ids if tid in bundle.windows],
        axis=0,
    )
    pcfg = _primary_cfg(config)
    max_fit = int(pcfg.get("max_windows_per_fold_fit", 30_000))
    if len(healthy_windows) > max_fit:
        rng = np.random.default_rng(random_state)
        healthy_windows = healthy_windows[rng.choice(len(healthy_windows), max_fit, replace=False)]

    model, norm = train_healthy_ae(
        healthy_windows, bundle.sensor_slices, config, device=device, checkpoint_path=None
    )

    recon_tr, lat_tr = trial_mean_scores(model, bundle, train_trial_ids, norm, device=device)
    recon_te, lat_te = trial_mean_scores(model, bundle, test_trial_ids, norm, device=device)

    healthy_lat_rows = [
        lat_tr[i]
        for i, tid in enumerate(train_trial_ids)
        if tid in healthy_train_trial_ids and np.isfinite(lat_tr[i]).all()
    ]
    healthy_latent = np.vstack(healthy_lat_rows) if healthy_lat_rows else lat_tr
    oc_models = fit_latent_one_class_models(healthy_latent, random_state=random_state)

    if_tr, svm_tr = score_latent_one_class(oc_models, lat_tr)
    if_te, svm_te = score_latent_one_class(oc_models, lat_te)

    healthy_train_mask = np.array([tid in healthy_train_trial_ids for tid in train_trial_ids])
    train_methods = {
        METHOD_AE_RECON: recon_tr,
        METHOD_IF_LATENT: if_tr,
        METHOD_OCSVM_LATENT: svm_tr,
    }
    test_methods = {
        METHOD_AE_RECON: recon_te,
        METHOD_IF_LATENT: if_te,
        METHOD_OCSVM_LATENT: svm_te,
    }
    ref = {k: train_methods[k][healthy_train_mask] for k in ENSEMBLE_METHODS}
    train_methods[METHOD_ENSEMBLE] = combine_ensemble_scores(train_methods, config, reference_scores=ref)
    test_methods[METHOD_ENSEMBLE] = combine_ensemble_scores(test_methods, config, reference_scores=ref)
    return train_methods, test_methods


def build_fold_trial_latents(
    bundle: TrialWindowBundle,
    train_trial_ids: list[str],
    test_trial_ids: list[str],
    healthy_train_trial_ids: list[str],
    config: dict[str, Any],
    *,
    device: torch.device,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train Healthy-only AE per LOSO fold; return pooled trial latent means."""
    healthy_windows = np.concatenate(
        [bundle.windows[tid] for tid in healthy_train_trial_ids if tid in bundle.windows],
        axis=0,
    )
    pcfg = _primary_cfg(config)
    max_fit = int(pcfg.get("max_windows_per_fold_fit", 30_000))
    if len(healthy_windows) > max_fit:
        rng = np.random.default_rng(random_state)
        healthy_windows = healthy_windows[rng.choice(len(healthy_windows), max_fit, replace=False)]

    model, norm = train_healthy_ae(
        healthy_windows, bundle.sensor_slices, config, device=device, checkpoint_path=None
    )
    _, lat_tr = trial_mean_scores(model, bundle, train_trial_ids, norm, device=device)
    _, lat_te = trial_mean_scores(model, bundle, test_trial_ids, norm, device=device)
    return lat_tr, lat_te
