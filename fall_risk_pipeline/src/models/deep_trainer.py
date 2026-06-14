"""
Deep-learning LOSO evaluation pipeline.

Loads cleaned per-trial signals, windows them, and runs Leave-One-Subject-Out
cross-validation for each architecture in the deep_learning config.

Trial-level window predictions are aggregated to participant-level via
mean class probability (soft vote across windows) to match the classical
ML evaluation granularity (N = 260 participants).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from src.evaluation.multiclass_metrics import build_multiclass_metric_payload
from src.models.deep_models import (
    DEEP_MODEL_REGISTRY,
    DeepModelTrainer,
    GaitTransformer,
    build_deep_model,
    create_windows,
    independent_stride_window_indices,
    trial_to_tensor,
)
from src.utils.progress import progress_bar, stderr_is_tty
from src.utils.reproducibility import get_pipeline_seed, set_global_seed

# Spread inner-val RNG seeds across LOSO fold indices (HIGH-002).
_INNER_VAL_SEED_STRIDE = 31337

optuna.logging.set_verbosity(optuna.logging.WARNING)


def extract_attention_weights(
    model: GaitTransformer,
    X: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Mean self-attention to each time step (last encoder layer, batch/query avg)."""
    model.eval().to(device)
    x = torch.tensor(X, dtype=torch.float32, device=device).transpose(1, 2)
    with torch.no_grad():
        x = model.input_proj(x)
        x = model.pos_enc(x)
        attn = None
        for layer in model.encoder.layers:
            attn_out, weights = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=True,
            )
            attn = weights
            x = layer.norm1(x + layer.dropout1(attn_out))
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
            x = layer.norm2(x + layer.dropout2(ff))
    return attn.mean(dim=(0, 1)).cpu().numpy()


class DeepLearningPipeline:
    """End-to-end DL training + LOSO evaluation on raw IMU windows."""

    def __init__(self, config: dict):
        self.config = config
        self.dl_cfg = config["deep_learning"]
        self.seed = get_pipeline_seed(config)

        self.signals_dir = Path(config["paths"]["processed_data"]) / "signals_clean"
        self.meta_path = Path(config["paths"]["processed_data"]) / "trial_metadata.csv"
        self.metrics_dir = Path(config["paths"]["metrics"])
        self.ckpt_dir = Path(config["paths"]["checkpoints"])
        self.figures_dir = Path(config["paths"]["figures_models"])
        for d in (self.metrics_dir, self.ckpt_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.sensor_positions = config["dataset"]["sensor_positions"]
        self.window_len = int(self.dl_cfg["sequence_length"])
        self.overlap = float(self.dl_cfg["overlap"])
        self.training_window_deduplication = bool(
            self.dl_cfg.get("training_window_deduplication", True)
        )
        self.enabled_models: list[str] = self.dl_cfg.get("models", list(DEEP_MODEL_REGISTRY.keys()))

    # ─────────────────────────────────────────────────────────────
    # Data preparation
    # ─────────────────────────────────────────────────────────────

    def _load_trial_data(self, bar=None) -> tuple[dict, pd.DataFrame]:
        meta = pd.read_csv(self.meta_path)
        trial_signals: dict[str, np.ndarray] = {}
        skipped = 0
        for _, row in meta.iterrows():
            tid = row["trial_id"]
            arr = trial_to_tensor(tid, self.signals_dir, self.sensor_positions)
            if arr is not None and arr.shape[1] >= self.window_len:
                trial_signals[tid] = arr
            else:
                skipped += 1
            if bar is not None:
                bar.update(1)
        logger.info(f"Loaded {len(trial_signals)} trials ({skipped} skipped, too short or missing)")
        meta = meta[meta["trial_id"].isin(trial_signals)].reset_index(drop=True)
        return trial_signals, meta

    def _build_participant_windows(
        self, trial_signals: dict, meta: pd.DataFrame, bar=None
    ) -> tuple[dict, int]:
        participants: dict = {}
        n_channels = None
        for _, row in meta.iterrows():
            tid = row["trial_id"]
            pid = row["participant_id"]
            label = int(row["multiclass_label"])
            signal = trial_signals[tid]
            if n_channels is None:
                n_channels = signal.shape[0]
            wins, starts = create_windows(
                signal, self.window_len, self.overlap, return_starts=True
            )
            if len(wins) == 0:
                continue
            if pid not in participants:
                participants[pid] = {
                    "windows": [],
                    "window_starts": [],
                    "window_trial_ids": [],
                    "label": label,
                    "trial_ids": [],
                }
            participants[pid]["windows"].append(wins)
            participants[pid]["window_starts"].append(starts)
            participants[pid]["window_trial_ids"].extend([tid] * len(wins))
            participants[pid]["trial_ids"].append(tid)
            if bar is not None:
                bar.update(1)
        for pid in participants:
            participants[pid]["windows"] = np.concatenate(participants[pid]["windows"], axis=0)
            participants[pid]["window_starts"] = np.concatenate(
                participants[pid]["window_starts"], axis=0
            )
            participants[pid]["window_trial_ids"] = np.asarray(
                participants[pid]["window_trial_ids"], dtype=object
            )
        logger.info(
            f"Windowed data: {len(participants)} participants, "
            f"{sum(p['windows'].shape[0] for p in participants.values())} total windows, "
            f"{n_channels} channels, window_len={self.window_len}, overlap={self.overlap}"
        )
        return participants, n_channels or 0

    @staticmethod
    def _inner_val_split_seed(base_seed: int, fold_idx: int) -> int:
        """Derive a fold-specific seed that diversifies inner-val participant draws."""
        return int(base_seed) + int(fold_idx) * _INNER_VAL_SEED_STRIDE

    @staticmethod
    def split_inner_train_val_participants(
        train_pids: list[str],
        participant_labels: dict[str, int],
        *,
        seed: int,
        val_fraction: float = 0.1,
    ) -> tuple[list[str], list[str]]:
        """
        Hold out entire participants for inner validation (no window-level leakage).

        Splits at participant granularity with stratification on pathology label.
        """
        if len(train_pids) <= 2:
            inner_val = [train_pids[-1]]
            inner_train = train_pids[:-1]
            return inner_train, inner_val

        labels = np.array([participant_labels[pid] for pid in train_pids], dtype=int)
        n_val = max(1, int(round(val_fraction * len(train_pids))))
        if n_val >= len(train_pids):
            n_val = max(1, len(train_pids) - 1)

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=n_val,
            random_state=seed,
        )
        try:
            train_idx, val_idx = next(
                sss.split(np.arange(len(train_pids)), labels)
            )
        except ValueError:
            logger.warning(
                "Participant-level stratified val split failed (n=%d); "
                "using random participant holdout",
                len(train_pids),
            )
            rng = np.random.default_rng(seed)
            val_idx = rng.choice(len(train_pids), n_val, replace=False)
            train_idx = np.setdiff1d(np.arange(len(train_pids)), val_idx)

        inner_train = [train_pids[int(i)] for i in train_idx]
        inner_val = [train_pids[int(i)] for i in val_idx]
        return inner_train, inner_val

    @staticmethod
    def _collect_participant_window_rows(
        participant_ids: list[str],
        participants: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return windows, labels, participant ids, trial ids, and start indices."""
        X_list: list[np.ndarray] = []
        y_list: list[int] = []
        pid_list: list[str] = []
        trial_list: list[str] = []
        start_list: list[np.ndarray] = []
        for pid in participant_ids:
            wins = participants[pid]["windows"]
            n = len(wins)
            X_list.append(wins)
            y_list.extend([int(participants[pid]["label"])] * n)
            pid_list.extend([pid] * n)
            if "window_trial_ids" in participants[pid]:
                trial_list.extend(participants[pid]["window_trial_ids"].tolist())
            else:
                fallback_tid = participants[pid].get("trial_ids", [pid])[0]
                trial_list.extend([fallback_tid] * n)
            if "window_starts" in participants[pid]:
                start_list.append(participants[pid]["window_starts"])
            else:
                start_list.append(np.arange(n, dtype=int))
        if not X_list:
            empty_x = np.zeros((0,), dtype=np.float32)
            empty_i = np.array([], dtype=int)
            empty_o = np.array([], dtype=object)
            return empty_x, empty_i, empty_o, empty_o, empty_i
        return (
            np.concatenate(X_list, axis=0),
            np.array(y_list, dtype=int),
            np.asarray(pid_list, dtype=object),
            np.asarray(trial_list, dtype=object),
            np.concatenate(start_list, axis=0),
        )

    @staticmethod
    def concat_participant_windows(
        participant_ids: list[str],
        participants: dict,
        *,
        independent_stride_only: bool = False,
        window_len: int = 256,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate windows for the given participants."""
        windows, y, _, trial_ids, starts = DeepLearningPipeline._collect_participant_window_rows(
            participant_ids, participants
        )
        if independent_stride_only and len(windows):
            keep = independent_stride_window_indices(
                starts, trial_ids, window_len, seed=seed
            )
            windows = windows[keep]
            y = y[keep]
        if len(windows) == 0:
            return np.zeros((0,), dtype=np.float32), np.array([], dtype=int)
        return windows, y

    @staticmethod
    def window_participant_ids(
        participant_ids: list[str],
        participants: dict,
        *,
        independent_stride_only: bool = False,
        window_len: int = 256,
        seed: int = 0,
    ) -> np.ndarray:
        """One participant id per window row (parallel to concat_participant_windows)."""
        _, _, pids, trial_ids, starts = DeepLearningPipeline._collect_participant_window_rows(
            participant_ids, participants
        )
        if independent_stride_only and len(pids):
            keep = independent_stride_window_indices(
                starts, trial_ids, window_len, seed=seed
            )
            pids = pids[keep]
        return pids

    # ─────────────────────────────────────────────────────────────
    # LOSO evaluation
    # ─────────────────────────────────────────────────────────────

    def _loso_tune_cfg(self) -> dict:
        return self.dl_cfg.get("loso_hyperparameter_tuning") or {}

    def _loso_tune_enabled(self) -> bool:
        return bool(self._loso_tune_cfg().get("enabled", False))

    def _make_trainer(
        self,
        *,
        learning_rate: float | None = None,
        max_epochs: int | None = None,
        early_stopping_patience: int | None = None,
    ) -> DeepModelTrainer:
        """DeepModelTrainer with per-fold hyperparameter overrides (ML-042)."""
        cfg = copy.deepcopy(self.config)
        dl = dict(self.dl_cfg)
        if learning_rate is not None:
            dl["learning_rate"] = learning_rate
        if max_epochs is not None:
            dl["max_epochs"] = max_epochs
        if early_stopping_patience is not None:
            dl["early_stopping_patience"] = early_stopping_patience
        cfg["deep_learning"] = dl
        return DeepModelTrainer(cfg)

    @staticmethod
    def _participant_level_labels_proba(
        y_windows: np.ndarray,
        proba_windows: np.ndarray,
        participant_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate window labels/probabilities to one row per participant."""
        unique_pids = np.unique(participant_ids)
        agg_y: list[int] = []
        agg_prob: list[np.ndarray] = []
        for pid in unique_pids:
            mask = participant_ids == pid
            agg_y.append(int(y_windows[mask][0]))
            agg_prob.append(proba_windows[mask].mean(axis=0))
        return np.asarray(agg_y, dtype=int), np.vstack(agg_prob)

    def _tune_loso_fold_learning_rate(
        self,
        model_name: str,
        n_channels: int,
        n_classes: int,
        X_train_norm: np.ndarray,
        y_train: np.ndarray,
        X_val_norm: np.ndarray,
        y_val: np.ndarray,
        train_window_pids: np.ndarray,
        val_window_pids: np.ndarray,
        fold_seed: int,
    ) -> float:
        """Optuna search for learning rate on inner train/val (participant-level AUC)."""
        tune = self._loso_tune_cfg()
        base_lr = float(self.dl_cfg["learning_rate"])
        n_trials = int(tune.get("n_trials", 5))
        timeout = int(tune.get("timeout_seconds", 120))
        search_epochs = int(tune.get("search_epochs", 12))
        search_patience = int(tune.get("search_patience", 4))
        lr_min = float(tune.get("lr_min_factor", 0.1)) * base_lr
        lr_max = float(tune.get("lr_max_factor", 10.0)) * base_lr

        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("learning_rate", lr_min, lr_max, log=True)
            trainer = self._make_trainer(
                learning_rate=lr,
                max_epochs=search_epochs,
                early_stopping_patience=search_patience,
            )
            model = build_deep_model(model_name, n_channels, self.window_len, n_classes)
            model = trainer.train(
                model,
                X_train_norm,
                y_train,
                X_val_norm,
                y_val,
                participant_ids_train=train_window_pids,
                participant_ids_val=val_window_pids,
                shuffle_seed=fold_seed + trial.number + 1,
            )
            proba = trainer.predict_proba(model, X_val_norm)
            y_part, p_part = self._participant_level_labels_proba(
                y_val, proba, val_window_pids
            )
            if len(np.unique(y_part)) < 2:
                return 0.5
            try:
                if n_classes == 2:
                    return float(roc_auc_score(y_part, p_part[:, 1]))
                return float(
                    roc_auc_score(y_part, p_part, multi_class="ovr", average="macro")
                )
            except ValueError:
                return 0.5

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=fold_seed),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        if study.best_trial is not None:
            return float(study.best_params.get("learning_rate", base_lr))
        return base_lr

    def _loso_evaluate(
        self,
        model_name: str,
        participants: dict,
        n_channels: int,
        bar,
    ) -> dict:
        pids = sorted(participants.keys())
        n_classes = len(set(p["label"] for p in participants.values()))
        base_lr = float(self.dl_cfg["learning_rate"])
        fold_lrs: list[float] = []

        oof_y_true = []
        oof_y_proba = []
        oof_pids = []

        for fold_idx, test_pid in enumerate(pids):
            set_global_seed(self.seed + fold_idx, deterministic_torch=True)

            test_data = participants[test_pid]
            X_test_raw = test_data["windows"]
            y_test_label = test_data["label"]

            train_pids = [p for p in pids if p != test_pid]
            participant_labels = {
                pid: int(participants[pid]["label"]) for pid in train_pids
            }
            inner_train_pids, inner_val_pids = self.split_inner_train_val_participants(
                train_pids,
                participant_labels,
                seed=self._inner_val_split_seed(self.seed, fold_idx),
                val_fraction=0.1,
            )
            dedupe_kw = {}
            if self.training_window_deduplication:
                dedupe_kw = {
                    "independent_stride_only": True,
                    "window_len": self.window_len,
                    "seed": self.seed + fold_idx,
                }
            X_train_f_raw, y_train_f = self.concat_participant_windows(
                inner_train_pids, participants, **dedupe_kw
            )
            X_val_raw, y_val = self.concat_participant_windows(
                inner_val_pids, participants, **dedupe_kw
            )
            train_window_pids = self.window_participant_ids(
                inner_train_pids, participants, **dedupe_kw
            )
            val_window_pids = self.window_participant_ids(
                inner_val_pids, participants, **dedupe_kw
            )

            if len(X_train_f_raw) == 0 or len(X_val_raw) == 0:
                logger.warning(
                    f"Skipping DL fold {fold_idx + 1}: empty inner train/val after "
                    f"participant split (train_pids={len(inner_train_pids)}, "
                    f"val_pids={len(inner_val_pids)})"
                )
                continue

            # Fit normalization on inner-train only, then apply to val/test.
            X_train_norm, X_test_norm = self._normalize(X_train_f_raw, X_test_raw)
            _, X_val_norm = self._normalize(X_train_f_raw, X_val_raw)

            fold_label = f"{fold_idx + 1}/{len(pids)}"
            bar.set_postfix(
                model=model_name,
                fold=fold_label,
                pid=str(test_pid)[:12],
                epoch="-",
                loss="-",
                val_auc="-",
            )
            if not stderr_is_tty():
                logger.info(f"DL {model_name} LOSO fold {fold_label} ({test_pid})")

            def on_epoch(epoch, mean_loss, val_auc, best_auc):
                bar.set_postfix(
                    model=model_name,
                    fold=fold_label,
                    epoch=epoch,
                    loss=f"{mean_loss:.3f}",
                    val_auc=f"{val_auc:.3f}",
                    best=f"{best_auc:.3f}",
                )

            if self._loso_tune_enabled():
                fold_lr = self._tune_loso_fold_learning_rate(
                    model_name,
                    n_channels,
                    n_classes,
                    X_train_norm,
                    y_train_f,
                    X_val_norm,
                    y_val,
                    train_window_pids,
                    val_window_pids,
                    self.seed + fold_idx,
                )
            else:
                fold_lr = base_lr
            fold_lrs.append(fold_lr)

            trainer = self._make_trainer(learning_rate=fold_lr)
            model = build_deep_model(model_name, n_channels, self.window_len, n_classes)
            model = trainer.train(
                model,
                X_train_norm,
                y_train_f,
                X_val_norm,
                y_val,
                participant_ids_train=train_window_pids,
                participant_ids_val=val_window_pids,
                shuffle_seed=self.seed + fold_idx,
                on_epoch=on_epoch,
            )

            if model_name == "gait_transformer":
                self._gait_transformer_attn = {
                    "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                    "X_norm": X_test_norm[: min(128, len(X_test_norm))].copy(),
                    "n_channels": n_channels,
                    "n_classes": n_classes,
                }

            proba_windows = trainer.predict_proba(model, X_test_norm)
            # Participant-level soft vote: mean window probabilities (ML-030).
            participant_proba = proba_windows.mean(axis=0)

            oof_y_true.append(y_test_label)
            oof_y_proba.append(participant_proba)
            oof_pids.append(test_pid)

            bar.update(1)

            del model, X_train_f_raw, X_val_raw, X_test_raw, X_train_norm, X_test_norm, X_val_norm
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        y_true = np.array(oof_y_true)
        y_proba = np.vstack(oof_y_proba).astype(np.float32)
        # Guard against tiny AMP-induced drift (nan/inf/rows not summing to 1)
        # so multiclass roc_auc_score can validate probabilities reliably.
        y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=0.0, neginf=0.0)
        y_proba = np.clip(y_proba, 0.0, 1.0)
        row_sums = y_proba.sum(axis=1, keepdims=True)
        invalid_rows = ~np.isfinite(row_sums) | (row_sums <= 0.0)
        n_invalid = int(invalid_rows.sum())
        if n_invalid:
            logger.warning(
                "dl_%s: repaired %d/%d participant probability rows (nan/inf/non-normalized)",
                model_name,
                n_invalid,
                len(y_proba),
            )
            y_proba[invalid_rows.ravel()] = 1.0 / max(n_classes, 1)
            row_sums = y_proba.sum(axis=1, keepdims=True)
        y_proba = y_proba / np.clip(row_sums, 1e-12, None)
        y_pred = np.argmax(y_proba, axis=1)

        payload = build_multiclass_metric_payload(
            f"dl_{model_name}", y_true, y_proba, y_pred, seed=self.seed
        )
        if not np.isfinite(float(payload.get("auc", float("nan")))):
            try:
                payload["auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
            except ValueError:
                payload["auc"] = float("nan")
        payload["participant_ids"] = oof_pids
        fold_lrs_arr = np.asarray(fold_lrs, dtype=float)
        payload["hyperparameter_protocol"] = (
            "loso_inner_participant_optuna_lr"
            if self._loso_tune_enabled()
            else "fixed_global_config"
        )
        payload["learning_rate_base"] = base_lr
        if len(fold_lrs_arr):
            payload["learning_rate_tuned_median"] = float(np.median(fold_lrs_arr))
            payload["learning_rate_tuned_std"] = (
                float(np.std(fold_lrs_arr)) if len(fold_lrs_arr) > 1 else 0.0
            )
        else:
            payload["learning_rate_tuned_median"] = base_lr
            payload["learning_rate_tuned_std"] = 0.0
        payload["window_overlap"] = self.overlap
        payload["training_window_protocol"] = (
            "independent_stride_blocks_per_trial"
            if self.training_window_deduplication
            else "all_overlapping_windows"
        )
        payload["inference_window_protocol"] = "all_overlapping_windows"
        return payload

    @staticmethod
    def _normalize(
        X_train: np.ndarray, X_apply: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-channel z-normalization fitted on training windows, applied to X_apply."""
        B, C, T = X_train.shape
        flat_train = X_train.transpose(1, 0, 2).reshape(C, -1)
        mean = flat_train.mean(axis=1, keepdims=True)
        std = flat_train.std(axis=1, keepdims=True) + 1e-8

        def _apply(x: np.ndarray) -> np.ndarray:
            n = x.shape[0]
            flat = x.transpose(1, 0, 2).reshape(C, -1)
            normed = ((flat - mean) / std).reshape(C, n, T).transpose(1, 0, 2)
            return normed.astype(np.float32)

        X_train_norm = _apply(X_train)
        X_apply_norm = _apply(X_apply)

        train_flat = X_train_norm.transpose(1, 0, 2).reshape(C, -1)
        assert np.allclose(train_flat.mean(axis=1), 0.0, atol=1e-4), (
            "normalized train windows should have ~zero per-channel mean"
        )
        assert np.allclose(train_flat.std(axis=1), 1.0, atol=1e-4), (
            "normalized train windows should have ~unit per-channel std"
        )
        assert X_apply_norm.shape == X_apply.shape, "normalization must preserve window shape"
        assert np.isfinite(X_apply_norm).all(), "normalized apply array contains non-finite values"

        return X_train_norm, X_apply_norm

    def _plot_gait_transformer_attention(self) -> None:
        """Save averaged temporal attention map from last LOSO fold GaitTransformer."""
        info = getattr(self, "_gait_transformer_attn", None)
        if not info:
            return
        model = build_deep_model(
            "gait_transformer", info["n_channels"], self.window_len, info["n_classes"]
        )
        model.load_state_dict(info["state_dict"])
        trainer = DeepModelTrainer(self.config)
        weights = extract_attention_weights(model, info["X_norm"], trainer.device)
        if weights.size == 0:
            return

        reporting = self.config.get("reporting", {})
        ext = str(reporting.get("figure_format", "png"))
        dpi = int(reporting.get("figure_dpi", 150))
        stem = "dl_gait_transformer_attention"
        fs = float(self.config.get("dataset", {}).get("sampling_rate", 100))
        t_axis = np.arange(len(weights)) / fs

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(t_axis, weights, color="#0a84ff", linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mean attention weight")
        ax.set_title("GaitTransformer — averaged self-attention vs time (last LOSO fold)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.figures_dir / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        pd.DataFrame({"time_s": t_axis, "attention": weights}).to_csv(
            self.metrics_dir / f"{stem}.csv", index=False
        )
        logger.info(f"GaitTransformer attention map saved → {self.figures_dir / f'{stem}.{ext}'}")

    def _merge_dl_significance_into_metrics(self, pval_df: pd.DataFrame) -> None:
        """Write DL vs classical bootstrap p-values into metrics.csv for paper tables."""
        metrics_path = self.metrics_dir / "metrics.csv"
        if not metrics_path.exists() or pval_df.empty:
            return
        mdf = pd.read_csv(metrics_path)
        pv = pval_df.set_index("dl_model")
        for i, row in mdf.iterrows():
            model = str(row["model"])
            if model not in pv.index:
                continue
            rec = pv.loc[model]
            mdf.at[i, "p_delong_vs_best"] = float("nan")
            mdf.at[i, "p_delong_fmt"] = "n/a (multiclass)"
            mdf.at[i, "p_bootstrap_delta_vs_best"] = rec.get("p_bootstrap_delta", float("nan"))
            mdf.at[i, "p_bootstrap_delta_fmt"] = rec.get("p_bootstrap_delta_fmt", "")
            mdf.at[i, "fdr_q_bootstrap_delta_vs_best"] = rec.get(
                "fdr_q_bootstrap_delta", float("nan")
            )
            mdf.at[i, "p_bootstrap_mwu_vs_best"] = rec.get("p_bootstrap_mwu", float("nan"))
            mdf.at[i, "auc_reference_model"] = rec.get("classical_reference", "")
        mdf.to_csv(metrics_path, index=False)
        dl_metrics_path = self.metrics_dir / "deep_learning_metrics.csv"
        if dl_metrics_path.exists():
            ddf = pd.read_csv(dl_metrics_path)
            for i, row in ddf.iterrows():
                model = str(row["model"])
                if model not in pv.index:
                    continue
                rec = pv.loc[model]
                ddf.at[i, "p_bootstrap_delta_vs_best"] = rec.get("p_bootstrap_delta", float("nan"))
                ddf.at[i, "p_bootstrap_mwu_vs_best"] = rec.get("p_bootstrap_mwu", float("nan"))
                ddf.at[i, "classical_reference"] = rec.get("classical_reference", "")
            ddf.to_csv(dl_metrics_path, index=False)

    # ─────────────────────────────────────────────────────────────
    # Main entry
    # ─────────────────────────────────────────────────────────────

    def run(self) -> dict[str, dict]:
        """Run LOSO evaluation for all enabled DL models and save metrics."""
        logger.info("=== Deep Learning LOSO Pipeline ===")
        t0 = time.time()

        meta = pd.read_csv(self.meta_path)
        n_trials = len(meta)
        models = [m for m in self.enabled_models if m in DEEP_MODEL_REGISTRY]
        n_folds = 260  # expected participants; refined after load
        prep_steps = n_trials * 2  # load + window
        total_steps = prep_steps + len(models) * n_folds

        bar = progress_bar(total=total_steps, desc="train_deep", unit="step")

        trial_signals, meta = self._load_trial_data(bar)
        participants, n_channels = self._build_participant_windows(trial_signals, meta, bar)

        if n_channels == 0:
            bar.close()
            logger.error("No valid trial signals found; aborting DL pipeline.")
            return {}

        n_folds = len(participants)
        # Adjust total if participant count differs from estimate
        bar.total = n_trials * 2 + len(models) * n_folds
        bar.refresh()

        all_results: dict[str, dict] = {}
        rows = []
        oof_rows: list[dict] = []

        for model_name in models:
            logger.info(f"Evaluating DL model: {model_name}")
            try:
                result = self._loso_evaluate(model_name, participants, n_channels, bar)
                all_results[model_name] = result
                if model_name == "gait_transformer":
                    self._plot_gait_transformer_attention()
                logger.info(
                    f"  {model_name}: AUC={result['auc']:.4f}  "
                    f"macro-F1={result['f1']:.4f}  acc={result['accuracy']:.4f}"
                )
                rows.append({
                    "model": f"dl_{model_name}",
                    "label_mode": "multiclass",
                    "auc": result["auc"],
                    "auc_ci_low": result["auc_ci_low"],
                    "auc_ci_high": result["auc_ci_high"],
                    "macro_f1": result["f1"],
                    "accuracy": result["accuracy"],
                    "evaluation_mode": "loso_dl",
                    "validation_strategy": "LOSO",
                    "participants": len(participants),
                    "hyperparameter_protocol": result.get(
                        "hyperparameter_protocol", "fixed_global_config"
                    ),
                    "learning_rate_base": result.get("learning_rate_base"),
                    "learning_rate_tuned_median": result.get("learning_rate_tuned_median"),
                    "window_overlap": result.get("window_overlap"),
                    "training_window_protocol": result.get("training_window_protocol"),
                    "inference_window_protocol": result.get("inference_window_protocol"),
                })
                y_true = np.asarray(result.get("y_true", []), dtype=int)
                y_pred = np.asarray(result.get("y_pred", []), dtype=int)
                y_proba_full = np.asarray(result.get("y_proba_full", []), dtype=float)
                pids = result.get("participant_ids", [])
                if (
                    y_proba_full.ndim == 2
                    and len(y_true) == len(y_pred) == len(pids) == y_proba_full.shape[0]
                    and y_proba_full.shape[1] > 0
                ):
                    for i in range(y_proba_full.shape[0]):
                        row = {
                            "model": f"dl_{model_name}",
                            "participant_id": pids[i],
                            "y_true": int(y_true[i]),
                            "y_pred": int(y_pred[i]),
                        }
                        for c in range(y_proba_full.shape[1]):
                            row[f"y_prob_class_{c}"] = float(y_proba_full[i, c])
                        oof_rows.append(row)
            except Exception as exc:
                logger.error(f"DL model {model_name} failed: {exc}")

        bar.close()

        if rows:
            dl_df = pd.DataFrame(rows)
            out_path = self.metrics_dir / "deep_learning_metrics.csv"
            dl_df.to_csv(out_path, index=False)
            logger.info(f"DL metrics saved → {out_path}")

            existing_path = self.metrics_dir / "metrics.csv"
            if existing_path.exists():
                existing = pd.read_csv(existing_path)
                existing = existing[~existing["model"].str.startswith("dl_")]
                combined = pd.concat([existing, dl_df], ignore_index=True)
                combined.to_csv(existing_path, index=False)
                logger.info(f"DL metrics appended to {existing_path}")

        if oof_rows:
            oof_df = pd.DataFrame(oof_rows)
            oof_path = self.metrics_dir / "deep_learning_oof_predictions.parquet"
            oof_df.to_parquet(oof_path, index=False)
            logger.info(f"DL OOF predictions saved → {oof_path}")

            from src.evaluation.auc_significance import dl_vs_classical_auc_significance

            eval_cfg = self.config.get("models", {}).get("evaluation", {})
            n_boot = int(eval_cfg.get("delong_bootstrap_n", 1000))
            seed = int(eval_cfg.get("random_state", self.seed))
            pval_df = dl_vs_classical_auc_significance(
                self.metrics_dir, n_bootstrap=n_boot, random_state=seed
            )
            if not pval_df.empty:
                self._merge_dl_significance_into_metrics(pval_df)

        elapsed = time.time() - t0
        logger.info(f"=== Deep Learning Pipeline complete in {elapsed:.1f}s ===")
        return all_results

