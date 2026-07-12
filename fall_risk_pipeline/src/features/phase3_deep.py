"""
Phase 3 deep representation features.

  - BiLSTM-AE latent activations h_t + per-sensor reconstruction error
  - ROCKET / MINIROCKET convolution features (PPV + max)
  - InceptionTime multi-scale context maps (kernels 10/20/40)

Perf note (batched extraction):

The original extraction loop called `extract_phase3_trial_features()`
once per trial, which meant:
  - RocketTransform.transform() / MiniRocketTransform.transform() were
    each invoked ~N_trials times, rebuilding per-group weight matrices
    every time (fixed in rocket_features.py) AND paying Python/dispatch
    overhead per call instead of amortizing it across many windows.
  - The BiLSTM-AE and InceptionTime forward passes ran on tiny
    per-trial batches (a few dozen windows) instead of large batches,
    so kernel-launch / interpreter overhead dominated actual compute.

`run_phase3_feature_extraction()` below still produces exactly one
feature row per trial with the same columns and the same aggregation
semantics (mean over a trial's own windows, matching std for the
latent stats), but the raw per-window computation — ROCKET,
MiniROCKET, AE, InceptionTime — is now run on large batches spanning
many trials at once (`extraction_batch_trials` trials per batch,
configurable), and results are aggregated back to per-trial rows with
a single pandas groupby instead of a Python-level running mean. This
does not change which trials/windows are used or how they're
combined — it only changes how many transform()/forward() calls it
takes to get there.

The original single-trial helpers (`extract_phase3_trial_features`,
`_ae_trial_features`, `_rocket_trial_features`,
`_inception_trial_features`) are kept as-is for on-demand /
single-trial inference (e.g. scoring one new trial at serve time),
where batching across trials isn't applicable anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from src.utils.progress import progress_bar
from src.utils.torch_device import resolve_torch_device

from src.features.rocket_features import MiniRocketTransform, RocketTransform
from src.utils.reproducibility import get_pipeline_seed
from src.models.bilstm_autoencoder import (
    BiLSTMAutoencoder,
    SensorChannelSlice,
    load_bilstm_autoencoder,
    train_bilstm_autoencoder,
)
from src.models.deep_models import (
    CHANNEL_ORDER,
    InceptionMultiscaleExtractor,
    trial_to_tensor,
)
from src.preprocessing.fold_normalization import (
    PerChannelZNorm,
    apply_per_channel_znorm,
    fit_per_channel_znorm,
    reconstruction_threshold_train_only,
)
from src.preprocessing.windowing import parse_window_spec, window_single_trial


def _phase3_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return (config.get("features") or {}).get("phase3_deep") or {}


def compute_sensor_slices(
    trial_id: str,
    signals_dir: Path,
    sensor_positions: list[str],
    channels: list[str] | None = None,
) -> list[SensorChannelSlice]:
    channels = channels or CHANNEL_ORDER
    slices: list[SensorChannelSlice] = []
    start = 0
    for pos in sensor_positions:
        path = signals_dir / f"{trial_id}_{pos}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        n_ch = len([c for c in channels if c in df.columns])
        if n_ch:
            slices.append(SensorChannelSlice(pos, start, start + n_ch))
            start += n_ch
    return slices


def _collect_train_fit_windows(
    config: dict[str, Any],
    *,
    max_windows: int | None = None,
) -> tuple[np.ndarray, list[SensorChannelSlice]]:
    """Healthy **train-fold** windows only (v13 — no val/test leakage)."""
    from src.dataset.train_fit_mask import healthy_train_fit_mask
    from src.preprocessing.windowing import parse_window_spec, window_single_trial

    processed = Path(config["paths"]["processed_data"])
    signals_dir = processed / "signals_clean"
    meta = pd.read_csv(processed / "trial_metadata.csv")
    spec = parse_window_spec(config)
    sensor_positions = config["dataset"]["sensor_positions"]
    fit_mask = healthy_train_fit_mask(meta, config)

    windows: list[np.ndarray] = []
    ref_slices: list[SensorChannelSlice] = []

    for i, row in meta.iterrows():
        if not fit_mask[i]:
            continue
        tid = row["trial_id"]
        arr = trial_to_tensor(tid, signals_dir, sensor_positions, require_all_sensors=True)
        if arr is None or arr.shape[1] < spec.window_len:
            continue
        if not ref_slices:
            ref_slices = compute_sensor_slices(tid, signals_dir, sensor_positions)
        wins = window_single_trial(arr, spec)
        if len(wins):
            windows.append(wins)

    if not windows:
        raise RuntimeError("No Healthy TRAIN-fold windows for Phase 3 model fitting")

    X = np.concatenate(windows, axis=0).astype(np.float32)
    if max_windows and len(X) > max_windows:
        rng = np.random.default_rng(get_pipeline_seed(config))
        X = X[rng.choice(len(X), max_windows, replace=False)]
    return X, ref_slices


def _normalize_windows(X: np.ndarray, norm: PerChannelZNorm) -> np.ndarray:
    return apply_per_channel_znorm(X, norm)


class Phase3ModelBundle:
    """Fitted Phase 3 transforms (train-fold Healthy reference)."""

    def __init__(
        self,
        *,
        ae: BiLSTMAutoencoder | None,
        rocket: RocketTransform | None,
        minirocket: MiniRocketTransform | None,
        inception: InceptionMultiscaleExtractor | None,
        sensor_slices: list[SensorChannelSlice],
        channel_norm: PerChannelZNorm,
        ae_recon_threshold: float | None,
        latent_export_dims: int,
        device: torch.device,
    ):
        self.ae = ae
        self.rocket = rocket
        self.minirocket = minirocket
        self.inception = inception
        self.sensor_slices = sensor_slices
        self.channel_norm = channel_norm
        self.ae_recon_threshold = ae_recon_threshold
        self.latent_export_dims = latent_export_dims
        self.device = device


def fit_phase3_models(config: dict[str, Any]) -> Phase3ModelBundle:
    """Train / fit Phase 3 models on Healthy train-fold windows only."""
    p3 = _phase3_cfg(config)
    if not p3.get("enabled", True):
        raise RuntimeError("phase3_deep disabled in config")

    ckpt_dir = Path(config["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ae_path = ckpt_dir / "phase3_bilstm_ae.pt"
    rocket_path = ckpt_dir / "phase3_rocket_kernels.npz"
    mini_path = ckpt_dir / "phase3_minirocket_kernels.npz"

    max_fit = int(p3.get("max_fit_windows", 50_000))
    X_raw, sensor_slices = _collect_train_fit_windows(config, max_windows=max_fit)
    channel_norm = fit_per_channel_znorm(X_raw)
    X_norm = _normalize_windows(X_raw, channel_norm)
    device = resolve_torch_device(config)
    latent_export = int((p3.get("bilstm_autoencoder") or {}).get("latent_export_dims", 16))

    ae: BiLSTMAutoencoder | None = None
    ae_recon_threshold: float | None = None
    ae_cfg = p3.get("bilstm_autoencoder") or {}
    if ae_cfg.get("enabled", True):
        if ae_path.is_file() and not p3.get("force_retrain", False):
            ae = load_bilstm_autoencoder(ae_path, device=device)
            logger.info("Loaded BiLSTM-AE checkpoint → {}", ae_path)
        else:
            logger.info("Training BiLSTM-AE on {} train-fold Healthy windows", len(X_norm))
            ae = train_bilstm_autoencoder(
                X_norm,
                sensor_slices=sensor_slices,
                config=config,
                checkpoint_path=ae_path,
            )
            ae.to(device)
        if ae is not None:
            train_errors: list[float] = []
            with torch.no_grad():
                for start in range(0, len(X_norm), 64):
                    batch = torch.tensor(
                        X_norm[start : start + 64], dtype=torch.float32, device=device
                    )
                    recon, _ = ae(batch)
                    train_errors.extend(
                        torch.mean((recon - batch) ** 2, dim=(1, 2)).cpu().numpy().tolist()
                    )
            ae_recon_threshold = reconstruction_threshold_train_only(
                np.asarray(train_errors, dtype=float),
                percentile=float(ae_cfg.get("threshold_percentile", 90.0)),
            )

    rocket: RocketTransform | None = None
    rk_cfg = p3.get("rocket") or {}
    if rk_cfg.get("enabled", True):
        n_k = int(rk_cfg.get("n_kernels", 10_000))
        if rocket_path.is_file() and not p3.get("force_retrain", False):
            rocket = RocketTransform.load(rocket_path)
        else:
            logger.info("Fitting ROCKET ({} kernels)", n_k)
            rocket = RocketTransform(n_k, seed=42).fit(X_raw)
            rocket.save(rocket_path)

    minirocket: MiniRocketTransform | None = None
    mr_cfg = p3.get("minirocket") or {}
    if mr_cfg.get("enabled", True):
        n_k = int(mr_cfg.get("n_kernels", 10_000))
        if mini_path.is_file() and not p3.get("force_retrain", False):
            minirocket = MiniRocketTransform.load(mini_path)
        else:
            logger.info("Fitting MINIROCKET ({} kernels)", n_k)
            minirocket = MiniRocketTransform(n_k, seed=42).fit(X_raw)
            minirocket.save(mini_path)

    inception: InceptionMultiscaleExtractor | None = None
    it_cfg = p3.get("inception_multiscale") or {}
    if it_cfg.get("enabled", True):
        inception = InceptionMultiscaleExtractor(int(X_raw.shape[1])).to(device)
        inception.eval()

    return Phase3ModelBundle(
        ae=ae,
        rocket=rocket,
        minirocket=minirocket,
        inception=inception,
        sensor_slices=sensor_slices,
        channel_norm=channel_norm,
        ae_recon_threshold=ae_recon_threshold,
        latent_export_dims=latent_export,
        device=device,
    )


# --------------------------------------------------------------------------
# Legacy single-trial helpers — kept for on-demand / single-trial inference.
# Not used by the batched extraction path below (run_phase3_feature_extraction).
# --------------------------------------------------------------------------


def _ae_trial_features(
    windows: np.ndarray,
    bundle: Phase3ModelBundle,
) -> dict[str, float]:
    if bundle.ae is None or len(windows) == 0:
        return {}
    X = _normalize_windows(windows, bundle.channel_norm)
    feats: dict[str, float] = {}
    latent_chunks: list[np.ndarray] = []
    sensor_mse_acc: dict[str, list[float]] = {s.name: [] for s in bundle.sensor_slices}
    sensor_mse_acc["total"] = []

    with torch.no_grad():
        for start in range(0, len(X), 64):
            batch = torch.tensor(X[start : start + 64], dtype=torch.float32, device=bundle.device)
            recon, h = bundle.ae(batch)
            per_mse = bundle.ae.per_sensor_mse(batch, recon)
            for key, vals in per_mse.items():
                if key in sensor_mse_acc:
                    sensor_mse_acc[key].extend(vals.cpu().numpy().tolist())
            latent_chunks.append(h.mean(dim=1).cpu().numpy())

    latent = np.concatenate(latent_chunks, axis=0)
    k = min(bundle.latent_export_dims, latent.shape[1])
    for dim in range(k):
        feats[f"ae_latent_h{dim:02d}_mean"] = float(np.mean(latent[:, dim]))
        feats[f"ae_latent_h{dim:02d}_std"] = float(np.std(latent[:, dim]))

    for sl in bundle.sensor_slices:
        short = sl.name.replace("_", "")
        if sl.name in sensor_mse_acc and sensor_mse_acc[sl.name]:
            feats[f"ae_recon_mse_{short}"] = float(np.mean(sensor_mse_acc[sl.name]))
    if sensor_mse_acc["total"]:
        feats["ae_recon_mse_total"] = float(np.mean(sensor_mse_acc["total"]))

    lb_key = "ae_recon_mse_lowerback"
    if "lower_back" in sensor_mse_acc and sensor_mse_acc["lower_back"]:
        feats[lb_key] = float(np.mean(sensor_mse_acc["lower_back"]))
        feats["ae_lb_recon_error"] = feats[lb_key]
    if bundle.ae_recon_threshold is not None and np.isfinite(bundle.ae_recon_threshold):
        feats["ae_recon_threshold_train_p90"] = float(bundle.ae_recon_threshold)
    return feats


def _rocket_trial_features(
    windows: np.ndarray,
    transform: RocketTransform | MiniRocketTransform,
    prefix: str,
    *,
    max_export: int,
) -> dict[str, float]:
    if len(windows) == 0:
        return {}
    mat = transform.transform(windows)
    trial_vec = np.mean(mat, axis=0)
    feats: dict[str, float] = {}
    n_export = min(max_export, len(trial_vec))
    for i in range(n_export):
        feats[f"{prefix}_f{i:05d}"] = float(trial_vec[i])
    feats[f"{prefix}_max_mean"] = float(np.mean(mat[:, 0::2]))
    feats[f"{prefix}_ppv_mean"] = float(np.mean(mat[:, 1::2]))
    return feats


def _inception_trial_features(windows: np.ndarray, bundle: Phase3ModelBundle) -> dict[str, float]:
    if bundle.inception is None or len(windows) == 0:
        return {}
    X = _normalize_windows(windows, bundle.channel_norm)
    names = InceptionMultiscaleExtractor.feature_names()
    accum = np.zeros(len(names), dtype=float)
    with torch.no_grad():
        for start in range(0, len(X), 64):
            batch = torch.tensor(X[start : start + 64], dtype=torch.float32, device=bundle.device)
            out = bundle.inception(batch).cpu().numpy()
            accum += out.sum(axis=0)
    trial_vec = accum / len(X)
    return {name: float(val) for name, val in zip(names, trial_vec, strict=True)}


def extract_phase3_trial_features(
    trial_id: str,
    signals_dir: Path,
    sensor_positions: list[str],
    bundle: Phase3ModelBundle,
    config: dict[str, Any],
) -> dict[str, float]:
    """Single-trial extraction. Use for on-demand / serving-time scoring
    of one new trial. Batch jobs over many trials should use
    `run_phase3_feature_extraction()` instead, which is far faster for
    the same result."""
    p3 = _phase3_cfg(config)
    spec = parse_window_spec(config)

    arr = trial_to_tensor(
        trial_id, signals_dir, sensor_positions, require_all_sensors=False
    )
    if arr is None or arr.shape[1] < spec.window_len:
        return {}

    windows = window_single_trial(arr, spec)
    if len(windows) == 0:
        return {}

    feats: dict[str, float] = {}
    feats.update(_ae_trial_features(windows, bundle))

    rk_export = int((p3.get("rocket") or {}).get("export_dims", 128))
    if bundle.rocket is not None:
        feats.update(_rocket_trial_features(windows, bundle.rocket, "rk", max_export=rk_export))

    mr_export = int((p3.get("minirocket") or {}).get("export_dims", 128))
    if bundle.minirocket is not None:
        feats.update(_rocket_trial_features(windows, bundle.minirocket, "mr", max_export=mr_export))

    feats.update(_inception_trial_features(windows, bundle))
    return feats


# --------------------------------------------------------------------------
# Batched extraction path — same features, same aggregation, far fewer
# transform()/forward() calls. This is what run_phase3_feature_extraction
# uses for full-dataset extraction.
# --------------------------------------------------------------------------


def _expected_n_channels(sensor_slices: list[SensorChannelSlice]) -> int:
    return max((sl.end for sl in sensor_slices), default=0)


def _trial_to_padded_tensor(
    trial_id: str,
    signals_dir: Path,
    sensor_positions: list[str],
    sensor_slices: list[SensorChannelSlice],
    *,
    n_channels: int,
    channels: list[str] | None = None,
) -> np.ndarray | None:
    """Load trial IMU into a fixed (n_channels, T) layout using ``sensor_slices``.

    Missing sensors are zero-padded so batched window concatenation and the
  fitted AE/ROCKET transforms (trained on the full Voisard layout) see a
    consistent channel dimension — same convention as DAPHNET LB zero-shot eval.
    """
    channels = channels or CHANNEL_ORDER
    per_sensor: dict[str, np.ndarray] = {}
    min_len = float("inf")

    for pos in sensor_positions:
        path = signals_dir / f"{trial_id}_{pos}.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path)
        usable = [c for c in channels if c in df.columns]
        if not usable:
            continue
        arr = df[usable].values.T.astype(np.float32)
        per_sensor[pos] = arr
        min_len = min(min_len, arr.shape[1])

    if not per_sensor or min_len < 64:
        return None

    t_len = int(min_len)
    full = np.zeros((n_channels, t_len), dtype=np.float32)
    for sl in sensor_slices:
        arr = per_sensor.get(sl.name)
        if arr is None:
            continue
        n_ch = sl.end - sl.start
        full[sl.start : sl.start + min(n_ch, arr.shape[0]), :] = arr[:n_ch, :t_len]
    return full


def _load_trial_windows(
    meta: pd.DataFrame,
    signals_dir: Path,
    sensor_positions: list[str],
    spec,
    bundle: Phase3ModelBundle,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load raw (unnormalized) windows for every trial in `meta` that has
    enough signal, returning:
      - trial_ids: list of trial ids actually kept (in order encountered)
      - windows: concatenated (M, C, T) raw window array across all trials
      - window_trial_idx: (M,) int array mapping each window row to an
        index into trial_ids
    """
    trial_ids: list[str] = []
    window_chunks: list[np.ndarray] = []
    window_trial_idx_chunks: list[np.ndarray] = []

    n_channels = _expected_n_channels(bundle.sensor_slices)
    if n_channels <= 0:
        return trial_ids, np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    for _, row in meta.iterrows():
        tid = row["trial_id"]
        arr = _trial_to_padded_tensor(
            tid,
            signals_dir,
            sensor_positions,
            bundle.sensor_slices,
            n_channels=n_channels,
        )
        if arr is None or arr.shape[1] < spec.window_len:
            continue
        wins = window_single_trial(arr, spec)
        if len(wins) == 0:
            continue
        trial_idx = len(trial_ids)
        trial_ids.append(tid)
        window_chunks.append(wins)
        window_trial_idx_chunks.append(np.full(len(wins), trial_idx, dtype=np.int64))

    if not window_chunks:
        return trial_ids, np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    windows = np.concatenate(window_chunks, axis=0)
    window_trial_idx = np.concatenate(window_trial_idx_chunks, axis=0)
    return trial_ids, windows, window_trial_idx


def _ae_window_batch_features(
    windows_raw: np.ndarray,
    bundle: Phase3ModelBundle,
    *,
    forward_batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per-window AE features for a (possibly multi-trial) batch of windows.

    Returns:
      - latent: (M, K) per-window latent means (h.mean over time)
      - sensor_mse: dict sensor_name -> (M,) per-window MSE, including "total"
    """
    if bundle.ae is None or len(windows_raw) == 0:
        return np.empty((0, 0)), {}

    X = _normalize_windows(windows_raw, bundle.channel_norm)
    sensor_mse: dict[str, list[np.ndarray]] = {s.name: [] for s in bundle.sensor_slices}
    sensor_mse["total"] = []
    latent_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(X), forward_batch_size):
            batch = torch.tensor(
                X[start : start + forward_batch_size], dtype=torch.float32, device=bundle.device
            )
            recon, h = bundle.ae(batch)
            per_mse = bundle.ae.per_sensor_mse(batch, recon)
            for key, vals in per_mse.items():
                if key in sensor_mse:
                    sensor_mse[key].append(vals.cpu().numpy())
            latent_chunks.append(h.mean(dim=1).cpu().numpy())

    latent = np.concatenate(latent_chunks, axis=0)
    sensor_mse_arr = {
        k: (np.concatenate(v) if v else np.empty((0,))) for k, v in sensor_mse.items()
    }
    return latent, sensor_mse_arr


def _inception_window_batch_features(
    windows_raw: np.ndarray,
    bundle: Phase3ModelBundle,
    *,
    forward_batch_size: int,
) -> np.ndarray:
    """Per-window InceptionTime features, shape (M, F)."""
    if bundle.inception is None or len(windows_raw) == 0:
        return np.empty((0, 0))
    X = _normalize_windows(windows_raw, bundle.channel_norm)
    out_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), forward_batch_size):
            batch = torch.tensor(
                X[start : start + forward_batch_size], dtype=torch.float32, device=bundle.device
            )
            out_chunks.append(bundle.inception(batch).cpu().numpy())
    return np.concatenate(out_chunks, axis=0)


def _aggregate_batch_to_trial_rows(
    trial_ids_in_batch: list[str],
    window_trial_idx: np.ndarray,
    *,
    bundle: Phase3ModelBundle,
    p3: dict[str, Any],
    latent: np.ndarray,
    sensor_mse: dict[str, np.ndarray],
    rk_mat: np.ndarray | None,
    mr_mat: np.ndarray | None,
    inception_mat: np.ndarray | None,
) -> list[dict[str, float]]:
    """Collapse per-window arrays down to one feature row per trial,
    reproducing exactly the same aggregation the single-trial helpers did
    (mean over the trial's own windows; std for latent stats)."""
    n_trials = len(trial_ids_in_batch)
    rows: list[dict[str, float]] = [dict() for _ in range(n_trials)]

    # --- AE latent mean/std + per-sensor reconstruction MSE ---
    if latent.size:
        k = min(bundle.latent_export_dims, latent.shape[1])
        latent_df = pd.DataFrame(latent[:, :k], columns=[f"h{d:02d}" for d in range(k)])
        latent_df["trial_idx"] = window_trial_idx
        grouped = latent_df.groupby("trial_idx")
        means = grouped.mean()
        stds = grouped.std(ddof=0)
        for trial_idx in means.index:
            row = rows[trial_idx]
            for dim in range(k):
                col = f"h{dim:02d}"
                row[f"ae_latent_h{dim:02d}_mean"] = float(means.loc[trial_idx, col])
                row[f"ae_latent_h{dim:02d}_std"] = float(stds.loc[trial_idx, col])

    if sensor_mse:
        mse_df = pd.DataFrame({k: v for k, v in sensor_mse.items() if v.size})
        if not mse_df.empty:
            mse_df["trial_idx"] = window_trial_idx
            mse_means = mse_df.groupby("trial_idx").mean()
            for sl in bundle.sensor_slices:
                short = sl.name.replace("_", "")
                if sl.name in mse_means.columns:
                    for trial_idx in mse_means.index:
                        rows[trial_idx][f"ae_recon_mse_{short}"] = float(
                            mse_means.loc[trial_idx, sl.name]
                        )
            if "total" in mse_means.columns:
                for trial_idx in mse_means.index:
                    rows[trial_idx]["ae_recon_mse_total"] = float(mse_means.loc[trial_idx, "total"])
            if "lower_back" in mse_means.columns:
                for trial_idx in mse_means.index:
                    val = float(mse_means.loc[trial_idx, "lower_back"])
                    rows[trial_idx]["ae_recon_mse_lowerback"] = val
                    rows[trial_idx]["ae_lb_recon_error"] = val

    if bundle.ae_recon_threshold is not None and np.isfinite(bundle.ae_recon_threshold):
        for row in rows:
            row["ae_recon_threshold_train_p90"] = float(bundle.ae_recon_threshold)

    # --- ROCKET / MiniROCKET ---
    for mat, prefix, cfg_key in (
        (rk_mat, "rk", "rocket"),
        (mr_mat, "mr", "minirocket"),
    ):
        if mat is None or mat.size == 0:
            continue
        max_export = int((p3.get(cfg_key) or {}).get("export_dims", 128))
        mat_df = pd.DataFrame(mat)
        mat_df["trial_idx"] = window_trial_idx
        trial_vecs = mat_df.groupby("trial_idx").mean()
        n_export = min(max_export, mat.shape[1])
        for trial_idx in trial_vecs.index:
            vec = trial_vecs.loc[trial_idx].to_numpy()
            row = rows[trial_idx]
            for i in range(n_export):
                row[f"{prefix}_f{i:05d}"] = float(vec[i])
            row[f"{prefix}_max_mean"] = float(np.mean(vec[0::2]))
            row[f"{prefix}_ppv_mean"] = float(np.mean(vec[1::2]))

    # --- InceptionTime ---
    if inception_mat is not None and inception_mat.size:
        names = InceptionMultiscaleExtractor.feature_names()
        inc_df = pd.DataFrame(inception_mat, columns=names)
        inc_df["trial_idx"] = window_trial_idx
        inc_means = inc_df.groupby("trial_idx").mean()
        for trial_idx in inc_means.index:
            row = rows[trial_idx]
            for name in names:
                row[name] = float(inc_means.loc[trial_idx, name])

    return rows


def run_phase3_feature_extraction(config: dict[str, Any]) -> pd.DataFrame:
    """Extract Phase 3 features for all trials; merge into trial_features.parquet.

    Trials are processed in macro-batches of `extraction_batch_trials`
    trials (config: features.phase3_deep.extraction_batch_trials,
    default 200). Within each macro-batch, ROCKET/MiniROCKET/AE/
    InceptionTime each run ONCE over all windows in the batch rather
    than once per trial, then results are aggregated back to per-trial
    rows via groupby-mean (matching the original per-trial mean/std
    aggregation exactly). Larger macro-batches are faster but use more
    memory; tune `extraction_batch_trials` to your available RAM/VRAM.
    """
    p3 = _phase3_cfg(config)
    if not p3.get("enabled", True):
        logger.info("Phase 3 deep features disabled")
        return pd.DataFrame()

    processed = Path(config["paths"]["processed_data"])
    features_dir = Path(config["paths"]["features"])
    signals_dir = processed / "signals_clean"
    meta = pd.read_csv(processed / "trial_metadata.csv")
    sensor_positions = config["dataset"]["sensor_positions"]
    spec = parse_window_spec(config)

    bundle = fit_phase3_models(config)

    batch_trials = int(p3.get("extraction_batch_trials", 200))
    forward_batch_size = int(p3.get("forward_batch_size", 256))
    rocket_batch_windows = int(p3.get("rocket_transform_batch_windows", 8192))

    all_rows: list[dict[str, float]] = []
    meta_chunks = [meta.iloc[i : i + batch_trials] for i in range(0, len(meta), batch_trials)]

    for meta_chunk in progress_bar(
        meta_chunks,
        total=len(meta_chunks),
        desc="phase3_features_batched",
        colour="magenta",
        unit="trial_batch",
    ):
        trial_ids, windows_raw, window_trial_idx = _load_trial_windows(
            meta_chunk, signals_dir, sensor_positions, spec, bundle
        )
        if not trial_ids:
            continue

        latent, sensor_mse = _ae_window_batch_features(
            windows_raw, bundle, forward_batch_size=forward_batch_size
        )

        rk_mat = (
            bundle.rocket.transform(windows_raw, batch_size=rocket_batch_windows)
            if bundle.rocket is not None
            else None
        )
        mr_mat = (
            bundle.minirocket.transform(windows_raw, batch_size=rocket_batch_windows)
            if bundle.minirocket is not None
            else None
        )
        inception_mat = _inception_window_batch_features(
            windows_raw, bundle, forward_batch_size=forward_batch_size
        )

        batch_rows = _aggregate_batch_to_trial_rows(
            trial_ids,
            window_trial_idx,
            bundle=bundle,
            p3=p3,
            latent=latent,
            sensor_mse=sensor_mse,
            rk_mat=rk_mat,
            mr_mat=mr_mat,
            inception_mat=inception_mat,
        )
        for tid, row in zip(trial_ids, batch_rows, strict=True):
            row["trial_id"] = tid
            all_rows.append(row)

    if not all_rows:
        logger.warning("No Phase 3 features extracted")
        return pd.DataFrame()

    phase3_df = pd.DataFrame(all_rows)
    trial_path = features_dir / "trial_features.parquet"
    if trial_path.is_file():
        trial_df = pd.read_parquet(trial_path)
        phase3_cols = [c for c in phase3_df.columns if c != "trial_id"]
        trial_df = trial_df.drop(columns=[c for c in phase3_cols if c in trial_df.columns], errors="ignore")
        trial_df = trial_df.merge(phase3_df, on="trial_id", how="left")
        trial_df.to_parquet(trial_path, index=False)
        logger.info("Merged Phase 3 columns into {} ({} new cols)", trial_path, len(phase3_cols))

        patient_path = features_dir / "patient_features.parquet"
        if patient_path.is_file():
            from src.features.feature_extractor import FeatureExtractor

            patient_df = FeatureExtractor(config)._aggregate_to_patient(trial_df)
            patient_df.to_parquet(patient_path, index=False)
            logger.info("Re-aggregated patient features → {}", patient_path)
    else:
        phase3_path = features_dir / "trial_features_phase3.parquet"
        phase3_df.to_parquet(phase3_path, index=False)
        logger.info("Wrote standalone Phase 3 features → {}", phase3_path)

    if p3.get("run_baseline_eval", False):
        from src.evaluation.phase3_baseline_evaluator import run_phase3_rocket_baselines
        run_phase3_rocket_baselines(config)

    return phase3_df