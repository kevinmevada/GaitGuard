"""
Write ``PIPELINE_VERSION.json`` so generated metrics can be matched to code + config.

Emitted at the start of the ``report`` stage (CRIT-02 / reproducibility).
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.utils.reproducibility import get_pipeline_seed

PIPELINE_VERSION_FILENAME = "PIPELINE_VERSION.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_revision(repo_root: Path | None = None) -> dict[str, Any]:
    """Return ``{commit, dirty}``; null fields when git is unavailable."""
    root = repo_root or _repo_root()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def config_dict_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 of the loaded config dict (excludes ephemeral ``_pipeline_meta``)."""
    payload = {k: v for k, v in config.items() if not str(k).startswith("_")}
    canonical = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_file_hash(config_path: Path) -> str | None:
    if not config_path.is_file():
        return None
    return hashlib.sha256(config_path.read_bytes()).hexdigest()


def build_pipeline_version_record(
    config: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    meta = config.get("_pipeline_meta") or {}
    config_path = meta.get("config_path")
    config_path_obj = Path(config_path) if config_path else None

    fs = config.get("feature_selection") or {}
    ev = (config.get("models") or {}).get("evaluation") or {}

    record: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_revision(repo_root),
        "config_sha256": config_dict_hash(config),
        "config_path": str(config_path_obj) if config_path_obj else None,
        "config_file_sha256": (
            config_file_hash(config_path_obj) if config_path_obj else None
        ),
        "pipeline_seed": get_pipeline_seed(config),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "primary_endpoint": ev.get("primary_endpoint"),
        "feature_selection": {
            "enabled": fs.get("enabled"),
            "primary_method": fs.get("primary_method"),
            "rfecv_importance_method": fs.get("rfecv_importance_method", "permutation"),
            "max_features": fs.get("max_features"),
            "required_feature_substrings": fs.get("required_feature_substrings"),
            "nested_in_evaluation": fs.get("nested_in_evaluation"),
        },
        "anomaly_detection_enabled": bool(
            (config.get("anomaly_detection") or {}).get("enabled")
        ),
    }
    return record


def write_pipeline_version(
    metrics_dir: Path,
    config: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    record = build_pipeline_version_record(config, repo_root=repo_root)
    out_path = metrics_dir / PIPELINE_VERSION_FILENAME
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return out_path
