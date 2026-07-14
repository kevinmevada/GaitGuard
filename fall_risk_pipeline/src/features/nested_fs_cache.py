"""Disk cache for per-LOSO-fold nested RFECV feature masks.

Each fold is independent: write ``fold_<subject>.json`` as soon as RFECV
finishes so a killed evaluate/ablation run can resume without redoing
completed folds. Ablation / sensor / leakage paths that call
``nested_rfecv_column_indices`` for the same held-out subject reuse the
same cache (identical RFECV → identical metrics).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.utils.reproducibility import get_pipeline_seed

_SAFE_SUBJ = re.compile(r"[^A-Za-z0-9._-]+")


def build_nested_fs_fingerprint(
    config: dict[str, Any],
    feat_cols: list[str],
    *,
    n_samples: int,
    n_groups: int,
) -> str:
    """Stable key for a Nested FS cache directory (config + feature schema)."""
    payload = {
        "seed": get_pipeline_seed(config),
        "feature_selection": config.get("feature_selection") or {},
        "feat_cols": list(feat_cols),
        "n_samples": int(n_samples),
        "n_groups": int(n_groups),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def nested_fs_cache_dir(
    config: dict[str, Any],
    feat_cols: list[str],
    *,
    n_samples: int,
    n_groups: int,
) -> Path:
    """Return (and create) the cache directory for this Nested FS fingerprint."""
    metrics = Path(config["paths"]["metrics"])
    fp = build_nested_fs_fingerprint(
        config, feat_cols, n_samples=n_samples, n_groups=n_groups
    )
    cache_dir = metrics / "nested_fs_cache" / fp
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache_dir / "manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps(
                {
                    "fingerprint": fp,
                    "seed": get_pipeline_seed(config),
                    "n_features": len(feat_cols),
                    "n_samples": int(n_samples),
                    "n_groups": int(n_groups),
                    "feature_selection": config.get("feature_selection") or {},
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        logger.info("Nested FS disk cache: {}", cache_dir)
    return cache_dir


def _safe_subject_key(subject_key: str) -> str:
    return _SAFE_SUBJ.sub("_", str(subject_key))[:180]


def fold_cache_path(cache_dir: Path, subject_key: str) -> Path:
    return cache_dir / f"fold_{_safe_subject_key(subject_key)}.json"


def load_fold_selected(cache_dir: Path | None, subject_key: str) -> list[str] | None:
    if cache_dir is None:
        return None
    path = fold_cache_path(cache_dir, subject_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    feats = payload.get("selected") if isinstance(payload, dict) else None
    if not isinstance(feats, list) or not feats:
        return None
    return [str(x) for x in feats]


def save_fold_selected(
    cache_dir: Path | None, subject_key: str, selected: list[str]
) -> None:
    """Atomically persist selected feature names for one held-out subject."""
    if cache_dir is None or not selected:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = fold_cache_path(cache_dir, subject_key)
    payload = {
        "held_out_subject": str(subject_key),
        "selected": list(selected),
        "n_selected": len(selected),
    }
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(cache_dir)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def infer_held_out_subject(
    groups: np.ndarray, train_rows: np.ndarray
) -> str | None:
    """If ``train_rows`` is a LOSO train set (exactly one subject held out), return it."""
    train_rows = np.asarray(train_rows)
    groups = np.asarray(groups)
    if train_rows.dtype == bool:
        if train_rows.shape[0] != groups.shape[0]:
            return None
        train_groups = set(map(str, groups[train_rows]))
    else:
        train_groups = set(map(str, groups[np.asarray(train_rows, dtype=int)]))
    all_groups = set(map(str, groups))
    held = all_groups - train_groups
    if len(held) == 1:
        return next(iter(held))
    return None


def count_cached_folds(cache_dir: Path | None) -> int:
    if cache_dir is None or not cache_dir.exists():
        return 0
    return sum(1 for p in cache_dir.glob("fold_*.json") if p.is_file())
