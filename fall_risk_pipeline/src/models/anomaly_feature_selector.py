"""
LOSO-safe trial-level feature selection for Healthy-reference anomaly screening.

Ranks features by permutation importance on the train fold (non-Healthy = positive)
and caps dimensionality before one-class fitting — mirroring the supervised RFECV
discipline without leaking held-out participants.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _anomaly_fscfg(config: dict) -> dict:
    return config.get("anomaly", {}).get("feature_selection", {})


def anomaly_feature_selection_enabled(config: dict) -> bool:
    return bool(_anomaly_fscfg(config).get("enabled", True))


def select_anomaly_feature_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    config: dict,
    *,
    random_state: int = 42,
) -> np.ndarray:
    """
    Return sorted column indices for anomaly modeling on a LOSO train fold.

    When disabled, returns all columns. Otherwise ranks by permutation importance
    on pseudo-labels (non-Healthy = 1) and enforces required nonlinear families.
    """
    n_features = X_train.shape[1]
    all_idx = np.arange(n_features, dtype=int)
    fscfg = _anomaly_fscfg(config)
    if not fscfg.get("enabled", True):
        return all_idx

    max_features = int(fscfg.get("max_features", 30))
    min_features = int(fscfg.get("min_features", 8))
    required_substrings = [
        str(s).strip().lower()
        for s in fscfg.get("required_feature_substrings", [])
        if str(s).strip()
    ]
    max_features = max(min_features, max_features)

    y_train = np.asarray(y_train).astype(int)
    if len(np.unique(y_train)) < 2 or X_train.shape[0] < 10:
        logger.warning(
            "Anomaly feature selection skipped — insufficient train-fold diversity"
        )
        return all_idx

    clf = RandomForestClassifier(
        n_estimators=int(fscfg.get("n_estimators", 120)),
        max_depth=int(fscfg.get("max_depth", 6)),
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
    pipe.fit(X_train, y_train)
    n_repeats = int(fscfg.get("permutation_n_repeats", 5))
    result = permutation_importance(
        pipe,
        X_train,
        y_train,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    importances = np.asarray(result.importances_mean, dtype=float)
    ranked = np.argsort(importances)[::-1]

    selected: set[int] = set()
    for sub in required_substrings:
        for i, name in enumerate(feature_names):
            if sub in str(name).lower():
                selected.add(i)

    for idx in ranked:
        if len(selected) >= max_features:
            break
        selected.add(int(idx))

    if len(selected) < min_features:
        for idx in ranked:
            selected.add(int(idx))
            if len(selected) >= min_features:
                break

    out = np.array(sorted(selected), dtype=int)
    logger.debug(
        "Anomaly feature selection: {} / {} features (required slots={})",
        len(out),
        n_features,
        len(required_substrings),
    )
    return out


def subset_matrix(X: np.ndarray, col_idx: np.ndarray) -> np.ndarray:
    return np.asarray(X)[:, col_idx]


def save_anomaly_selected_features(
    metrics_dir: Path,
    feature_names: list[str],
    col_idx: np.ndarray,
    *,
    protocol: str,
) -> Path:
    """Persist deploy-time anomaly feature subset."""
    metrics_dir.mkdir(parents=True, exist_ok=True)
    names = [feature_names[int(i)] for i in col_idx]
    payload = {
        "feature_columns": names,
        "feature_indices": [int(i) for i in col_idx],
        "n_features": len(names),
        "protocol": protocol,
    }
    path = metrics_dir / "anomaly_selected_features.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_anomaly_selected_features(metrics_dir: Path) -> dict | None:
    path = metrics_dir / "anomaly_selected_features.json"
    if not path.is_file():
        return None
    import json as _json

    payload = _json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    cols = payload.get("feature_columns")
    if not isinstance(cols, list) or not cols:
        return None
    payload["feature_columns"] = [str(c) for c in cols]
    return payload
