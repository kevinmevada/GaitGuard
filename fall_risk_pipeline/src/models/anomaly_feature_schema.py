"""
Trial-level feature schema for anomaly detection scalers.

StandardScaler stores per-column mean/std in column order. The API must use the
same feature names and order as when scalers were fit on Healthy trials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_FILENAME = "trial_feature_schema.json"


def save_trial_feature_schema(
    results_dir: Path,
    feature_columns: list[str],
    *,
    healthy_n_samples: int,
) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_columns": [str(c) for c in feature_columns],
        "n_features": len(feature_columns),
        "scaler_fit_cohort": "Healthy",
        "healthy_n_samples": int(healthy_n_samples),
    }
    path = results_dir / SCHEMA_FILENAME
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def load_trial_feature_schema(results_dir: Path) -> dict[str, Any] | None:
    path = results_dir / SCHEMA_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        return None
    cols = payload.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        return None
    payload["feature_columns"] = [str(c) for c in cols]
    payload["n_features"] = int(payload.get("n_features", len(cols)))
    return payload


def scaler_expected_n_features(scalers: dict[str, Any]) -> int | None:
    """Return n_features_in_ if all scalers agree; None if empty or inconsistent."""
    expected: int | None = None
    for name, scaler in scalers.items():
        n = getattr(scaler, "n_features_in_", None)
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"Anomaly scaler '{name}' is missing n_features_in_.")
        if expected is None:
            expected = n
        elif n != expected:
            raise ValueError(
                f"Anomaly scaler dimension mismatch: '{name}' has n_features_in_={n}, "
                f"expected {expected} from other methods."
            )
    return expected


def validate_trial_columns_for_anomaly_scalers(
    runtime_columns: list[str],
    scalers: dict[str, Any],
    saved_schema: dict[str, Any] | None,
) -> tuple[bool, str]:
    """
    Verify runtime trial columns match scaler width and saved training schema.

    Returns (ok, message).
    """
    if not scalers:
        return True, "no anomaly scalers loaded"

    try:
        n_scaler = scaler_expected_n_features(scalers)
    except ValueError as exc:
        return False, str(exc)

    if n_scaler is None:
        return True, "no scalers"

    if saved_schema is not None:
        saved_cols = saved_schema["feature_columns"]
        if runtime_columns != saved_cols:
            missing = [c for c in saved_cols if c not in runtime_columns]
            extra = [c for c in runtime_columns if c not in saved_cols]
            return False, (
                f"trial feature columns do not match anomaly training schema "
                f"(saved n={len(saved_cols)}, runtime n={len(runtime_columns)}, "
                f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"extra={extra[:5]}{'...' if len(extra) > 5 else ''})"
            )
        if n_scaler != len(saved_cols):
            return False, (
                f"scaler n_features_in_={n_scaler} != saved schema n={len(saved_cols)}"
            )
        return True, "ok"

    if len(runtime_columns) != n_scaler:
        return False, (
            f"trial feature count {len(runtime_columns)} != scaler n_features_in_={n_scaler} "
            "(no trial_feature_schema.json — retrain anomaly stage to export schema)"
        )
    return True, "ok"
