"""Disk resume cache for the evaluate-stage leakage / split-protocol comparison.

Saves after each KFold seed and after each model so a killed run can skip
completed work. Nested RFECV on ungrouped folds is expensive (~hours/model);
without this cache a crash on the last model redoes everything.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from src.features.nested_fs_cache import build_nested_fs_fingerprint


def build_leakage_resume_key(
    config: dict[str, Any],
    feat_cols: list[str] | None,
    *,
    n_samples: int,
    n_groups: int,
) -> str:
    """Stable key: Nested FS fingerprint + leakage knobs that change results."""
    eval_cfg = (config.get("models") or {}).get("evaluation") or {}
    base = build_nested_fs_fingerprint(
        config,
        list(feat_cols or []),
        n_samples=int(n_samples),
        n_groups=int(n_groups),
    )
    payload = {
        "nested_fs_fp": base,
        "leakage_kfold_splits": int(eval_cfg.get("leakage_kfold_splits", 10)),
        "random_state": int(eval_cfg.get("random_state", 42)),
        "leakage_kfold_seed_repeats": int(eval_cfg.get("leakage_kfold_seed_repeats", 1)),
        "leakage_kfold_seed_repeats_by_model": dict(
            eval_cfg.get("leakage_kfold_seed_repeats_by_model") or {}
        ),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def leakage_resume_path(metrics_dir: Path) -> Path:
    return Path(metrics_dir) / "leakage_comparison_resume.json"


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


def load_leakage_resume(
    metrics_dir: Path, resume_key: str
) -> dict[str, Any]:
    """Return ``{"seed_aucs": {model: {rep: auc}}, "rows": [...]}`` or empty."""
    path = leakage_resume_path(metrics_dir)
    empty: dict[str, Any] = {"seed_aucs": {}, "rows": []}
    if not path.is_file():
        return empty
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Leakage resume load failed (will recompute): {}", exc)
        return empty
    if not isinstance(payload, dict) or payload.get("resume_key") != resume_key:
        logger.info(
            "Leakage resume key mismatch — ignoring {}",
            path,
        )
        return empty
    seed_aucs = payload.get("seed_aucs") or {}
    rows = payload.get("rows") or []
    if not isinstance(seed_aucs, dict):
        seed_aucs = {}
    if not isinstance(rows, list):
        rows = []
    # Normalize rep keys to int-string for stable lookup.
    norm_seeds: dict[str, dict[str, float]] = {}
    for model, reps in seed_aucs.items():
        if not isinstance(reps, dict):
            continue
        norm_seeds[str(model)] = {
            str(int(k)): float(v) for k, v in reps.items() if v is not None
        }
    return {"seed_aucs": norm_seeds, "rows": list(rows)}


def save_leakage_resume(
    metrics_dir: Path,
    resume_key: str,
    *,
    seed_aucs: dict[str, dict[str, float]],
    rows: list[dict[str, Any]],
) -> None:
    path = leakage_resume_path(metrics_dir)
    try:
        _atomic_write_json(
            path,
            {
                "resume_key": resume_key,
                "seed_aucs": seed_aucs,
                "rows": rows,
            },
        )
    except Exception as exc:
        logger.warning("Leakage resume save failed: {}", exc)


def completed_model_names(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(r["model"])
        for r in rows
        if isinstance(r, dict) and r.get("model") is not None
    }


def get_cached_seed_auc(
    seed_aucs: dict[str, dict[str, float]], model: str, rep: int
) -> float | None:
    by_rep = seed_aucs.get(model) or {}
    key = str(int(rep))
    if key not in by_rep:
        return None
    return float(by_rep[key])


def set_cached_seed_auc(
    seed_aucs: dict[str, dict[str, float]], model: str, rep: int, auc: float
) -> None:
    seed_aucs.setdefault(model, {})[str(int(rep))] = float(auc)
