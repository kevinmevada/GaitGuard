"""Per-feature missingness reporting and training warnings (MED-006)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

DEFAULT_HIGH_MISSINGNESS_THRESHOLD = 0.15


def _missing_fractions(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.array([], dtype=float)
    return np.mean(~np.isfinite(X), axis=0)


def compute_feature_missingness_rows(
    X: np.ndarray,
    feat_cols: list[str],
    *,
    threshold: float = DEFAULT_HIGH_MISSINGNESS_THRESHOLD,
) -> list[dict]:
    fracs = _missing_fractions(X)
    rows: list[dict] = []
    for idx, name in enumerate(feat_cols):
        frac = float(fracs[idx]) if idx < len(fracs) else 0.0
        rows.append({
            "feature": name,
            "missing_fraction": round(frac, 6),
            "missing_pct": round(100.0 * frac, 2),
            "exceeds_threshold": bool(frac > threshold),
            "threshold": threshold,
        })
    rows.sort(key=lambda r: (-r["missing_fraction"], r["feature"]))
    return rows


def warn_high_missingness_features(
    X: np.ndarray,
    feat_names: list[str] | None = None,
    *,
    threshold: float = DEFAULT_HIGH_MISSINGNESS_THRESHOLD,
    context: str = "training fold",
) -> list[str]:
    """Log WARNING for features with missing fraction above ``threshold``."""
    fracs = _missing_fractions(X)
    flagged: list[str] = []
    for idx, frac in enumerate(fracs):
        if frac <= threshold:
            continue
        label = feat_names[idx] if feat_names and idx < len(feat_names) else f"column_{idx}"
        flagged.append(label)
        logger.warning(
            "High missingness in {ctx}: feature '{feat}' has {pct:.1f}% non-finite values "
            "(>{lim:.0%} threshold). Median imputation may be unstable; consider dropping, "
            "a missingness indicator, or group-specific imputation.",
            ctx=context,
            feat=label,
            pct=100.0 * frac,
            lim=threshold,
        )
    return flagged


def write_feature_missingness_report(
    X: np.ndarray,
    feat_cols: list[str],
    metrics_dir: Path,
    *,
    threshold: float = DEFAULT_HIGH_MISSINGNESS_THRESHOLD,
) -> pd.DataFrame:
    rows = compute_feature_missingness_rows(
        X, feat_cols, threshold=threshold
    )
    df = pd.DataFrame(rows)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    path = metrics_dir / "feature_missingness_report.csv"
    df.to_csv(path, index=False)
    n_high = int(df["exceeds_threshold"].sum()) if not df.empty else 0
    if n_high:
        logger.warning(
            "Feature missingness report: {n} feature(s) exceed {lim:.0%} non-finite rate → {path}",
            n=n_high,
            lim=threshold,
            path=path,
        )
    else:
        logger.info("Feature missingness report saved → {}", path)
    return df
