"""
Deep-learning LOSO evaluation pipeline.

Loads cleaned per-trial signals, windows them, and runs Leave-One-Subject-Out
cross-validation for each architecture in the deep_learning config.

Trial-level window predictions are aggregated to participant-level via
majority-vote (hard) and mean-probability (soft) to match the classical
ML evaluation granularity (N = 260 participants).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import roc_auc_score

from src.evaluation.multiclass_metrics import build_multiclass_metric_payload
from src.models.deep_models import (
    DEEP_MODEL_REGISTRY,
    DeepModelTrainer,
    build_deep_model,
    create_windows,
    trial_to_tensor,
)
from src.utils.progress import progress_bar, stderr_is_tty
from src.utils.reproducibility import get_pipeline_seed, set_global_seed


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
            wins = create_windows(signal, self.window_len, self.overlap)
            if len(wins) == 0:
                continue
            if pid not in participants:
                participants[pid] = {"windows": [], "label": label, "trial_ids": []}
            participants[pid]["windows"].append(wins)
            participants[pid]["trial_ids"].append(tid)
            if bar is not None:
                bar.update(1)
        for pid in participants:
            participants[pid]["windows"] = np.concatenate(participants[pid]["windows"], axis=0)
        logger.info(
            f"Windowed data: {len(participants)} participants, "
            f"{sum(p['windows'].shape[0] for p in participants.values())} total windows, "
            f"{n_channels} channels, window_len={self.window_len}"
        )
        return participants, n_channels or 0

    # ─────────────────────────────────────────────────────────────
    # LOSO evaluation
    # ─────────────────────────────────────────────────────────────

    def _loso_evaluate(
        self,
        model_name: str,
        participants: dict,
        n_channels: int,
        bar,
    ) -> dict:
        pids = sorted(participants.keys())
        n_classes = len(set(p["label"] for p in participants.values()))
        trainer = DeepModelTrainer(self.config)

        oof_y_true = []
        oof_y_proba = []
        oof_pids = []

        for fold_idx, test_pid in enumerate(pids):
            set_global_seed(self.seed + fold_idx, deterministic_torch=True)

            test_data = participants[test_pid]
            X_test = test_data["windows"]
            y_test_label = test_data["label"]

            train_pids = [p for p in pids if p != test_pid]
            X_train_list, y_train_list = [], []
            for tp in train_pids:
                X_train_list.append(participants[tp]["windows"])
                y_train_list.extend([participants[tp]["label"]] * len(participants[tp]["windows"]))
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.array(y_train_list, dtype=int)

            X_train, X_test = self._normalize(X_train, X_test)

            val_size = max(1, int(0.1 * len(X_train)))
            rng = np.random.default_rng(self.seed + fold_idx)
            val_idx = rng.choice(len(X_train), val_size, replace=False)
            train_idx = np.setdiff1d(np.arange(len(X_train)), val_idx)
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train_f, y_train_f = X_train[train_idx], y_train[train_idx]

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

            model = build_deep_model(model_name, n_channels, self.window_len, n_classes)
            model = trainer.train(
                model,
                X_train_f,
                y_train_f,
                X_val,
                y_val,
                on_epoch=on_epoch,
            )

            proba_windows = trainer.predict_proba(model, X_test)
            participant_proba = proba_windows.mean(axis=0)

            oof_y_true.append(y_test_label)
            oof_y_proba.append(participant_proba)
            oof_pids.append(test_pid)

            bar.update(1)

            del model, X_train, X_test, X_train_f, X_val
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        y_true = np.array(oof_y_true)
        y_proba = np.vstack(oof_y_proba).astype(np.float32)
        # Guard against tiny AMP-induced drift (nan/inf/rows not summing to 1)
        # so multiclass roc_auc_score can validate probabilities reliably.
        y_proba = np.nan_to_num(y_proba, nan=0.0, posinf=0.0, neginf=0.0)
        y_proba = np.clip(y_proba, 0.0, 1.0)
        row_sums = y_proba.sum(axis=1, keepdims=True)
        invalid_rows = ~np.isfinite(row_sums) | (row_sums <= 0.0)
        if np.any(invalid_rows):
            y_proba[invalid_rows.ravel()] = 1.0 / max(n_classes, 1)
            row_sums = y_proba.sum(axis=1, keepdims=True)
        y_proba = y_proba / np.clip(row_sums, 1e-12, None)
        y_pred = np.argmax(y_proba, axis=1)

        payload = build_multiclass_metric_payload(
            f"dl_{model_name}", y_true, y_proba, y_pred
        )
        if not np.isfinite(float(payload.get("auc", float("nan")))):
            try:
                payload["auc"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
            except ValueError:
                payload["auc"] = float("nan")
        payload["participant_ids"] = oof_pids
        return payload

    @staticmethod
    def _normalize(X_train: np.ndarray, X_test: np.ndarray):
        """Per-channel z-normalization fitted on training windows."""
        B, C, T = X_train.shape
        flat_train = X_train.transpose(1, 0, 2).reshape(C, -1)
        mean = flat_train.mean(axis=1, keepdims=True)
        std = flat_train.std(axis=1, keepdims=True) + 1e-8
        X_train = ((X_train.transpose(1, 0, 2).reshape(C, -1) - mean) / std).reshape(C, B, T).transpose(1, 0, 2)
        Bt = X_test.shape[0]
        X_test = ((X_test.transpose(1, 0, 2).reshape(C, -1) - mean) / std).reshape(C, Bt, T).transpose(1, 0, 2)
        return X_train.astype(np.float32), X_test.astype(np.float32)

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

        elapsed = time.time() - t0
        logger.info(f"=== Deep Learning Pipeline complete in {elapsed:.1f}s ===")
        return all_results

