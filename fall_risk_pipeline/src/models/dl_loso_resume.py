"""Disk resume cache for deep-learning LOSO folds.

Each completed fold writes ``fold_<participant_id>.json`` under
``metrics/dl_loso_cache/<fingerprint>/<model_name>/``. A killed
``train_deep`` run reloads finished folds and only retrains the rest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.reproducibility import get_pipeline_seed

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Orchestration-only knobs — changing them must not invalidate fold OOF.
_FINGERPRINT_IGNORE_DL_KEYS = frozenset(
    {
        "device",  # cuda vs cpu should not change math if seeds fixed; keep out of key
    }
)


def _safe_pid(participant_id: str) -> str:
    return _SAFE.sub("_", str(participant_id))[:180]


def build_dl_loso_fingerprint(
    config: dict[str, Any],
    *,
    model_name: str,
    participant_ids: list[str],
    n_channels: int,
    n_windows_total: int,
) -> str:
    """Stable key for a DL LOSO fold-cache directory."""
    dl = dict(config.get("deep_learning") or {})
    for key in list(dl.keys()):
        if key in _FINGERPRINT_IGNORE_DL_KEYS:
            dl.pop(key, None)
    # Model list does not affect a single model's fold OOF.
    dl.pop("models", None)
    payload = {
        "seed": int(get_pipeline_seed(config)),
        "model_name": str(model_name),
        "deep_learning": dl,
        "participant_ids": [str(p) for p in participant_ids],
        "n_channels": int(n_channels),
        "n_windows_total": int(n_windows_total),
        "sensor_positions": list((config.get("dataset") or {}).get("sensor_positions") or []),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def dl_loso_model_cache_dir(
    metrics_dir: Path,
    fingerprint: str,
    model_name: str,
) -> Path:
    path = Path(metrics_dir) / "dl_loso_cache" / fingerprint / str(model_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def fold_cache_path(cache_dir: Path, participant_id: str) -> Path:
    return cache_dir / f"fold_{_safe_pid(participant_id)}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def save_dl_fold_result(
    cache_dir: Path,
    *,
    participant_id: str,
    fold_idx: int,
    y_true: int,
    y_proba: list[float],
    learning_rate: float,
) -> None:
    """Persist one completed LOSO fold's participant-level OOF result."""
    path = fold_cache_path(cache_dir, participant_id)
    try:
        _atomic_write_json(
            path,
            {
                "participant_id": str(participant_id),
                "fold_idx": int(fold_idx),
                "y_true": int(y_true),
                "y_proba": [float(x) for x in y_proba],
                "learning_rate": float(learning_rate),
            },
        )
    except Exception as exc:
        logger.warning("DL fold resume save failed ({}): {}", participant_id, exc)


def load_dl_fold_result(cache_dir: Path, participant_id: str) -> dict[str, Any] | None:
    path = fold_cache_path(cache_dir, participant_id)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("participant_id") is None or payload.get("y_proba") is None:
        return None
    return payload


def count_cached_folds(cache_dir: Path | None) -> int:
    if cache_dir is None or not cache_dir.exists():
        return 0
    return sum(1 for p in cache_dir.glob("fold_*.json") if p.is_file())


def write_cache_manifest(cache_dir: Path, fingerprint: str, model_name: str) -> None:
    manifest = cache_dir / "manifest.json"
    if manifest.exists():
        return
    try:
        _atomic_write_json(
            manifest,
            {
                "fingerprint": fingerprint,
                "model_name": model_name,
                "n_cached_folds": count_cached_folds(cache_dir),
            },
        )
    except Exception as exc:
        logger.warning("DL fold cache manifest write failed: {}", exc)
