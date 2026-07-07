"""
Post-hoc probability calibration and split-conformal prediction sets.

Both techniques in this module are *wrappers around already-computed model
outputs* — they consume the LOSO out-of-fold (OOF) predictions the pipeline
already produces (``y_true``, ``y_prob`` arrays) and never retrain, refit,
or rerun any upstream model. This closes two gaps identified in review:

1. Every probability the pipeline reports today is a bare, uncalibrated
   point estimate. :func:`fit_isotonic_calibrator` / :func:`apply_calibrator`
   map raw model probabilities onto empirically-observed frequencies using
   the model's own OOF predictions, so "70% confidence" is closer to
   meaning "correct about 70% of the time" than an unadjusted score is.
2. No prediction carries a distribution-free coverage guarantee.
   :func:`fit_conformal_threshold` / :func:`conformal_prediction_set`
   implement split-conformal classification (Vovk/Papadopoulos-style
   nonconformity scoring): given a desired miscoverage rate ``alpha``, the
   returned prediction *set* contains the true label with probability at
   least ``1 - alpha`` on new data drawn from the same distribution as the
   OOF calibration set — a guarantee that holds regardless of whether the
   underlying model is well-calibrated, and that a single point probability
   can never provide on its own.

Both binary and multiclass (one-vs-rest per class) probability layouts are
supported, matching the two label modes already used elsewhere in this
pipeline (``dataset.label_mode: binary`` / ``multiclass``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

@dataclass
class CalibrationArtifact:
    """A fitted calibration mapping, serializable to/from JSON.

    For binary: a single isotonic map from raw P(positive) to calibrated
    P(positive). For multiclass: one isotonic map per class, applied
    one-vs-rest and renormalized to sum to 1 (a standard, simple approach
    that trades a small amount of theoretical elegance for something that's
    easy to audit and serialize without a pickle/joblib dependency).
    """

    label_mode: str  # "binary" | "multiclass"
    n_classes: int
    # For binary: {"_binary": [x_thresholds, y_values]}
    # For multiclass: {"0": [...], "1": [...], ...} one entry per class
    isotonic_maps: dict[str, tuple[list[float], list[float]]]

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "label_mode": self.label_mode,
                    "n_classes": self.n_classes,
                    "isotonic_maps": self.isotonic_maps,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def from_json(cls, path: Path) -> "CalibrationArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            label_mode=data["label_mode"],
            n_classes=data["n_classes"],
            isotonic_maps={k: (v[0], v[1]) for k, v in data["isotonic_maps"].items()},
        )


def _fit_one_isotonic(raw: np.ndarray, target: np.ndarray) -> tuple[list[float], list[float]]:
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(raw, target)
    # IsotonicRegression exposes the fitted step function via X_thresholds_/y_thresholds_
    return iso.X_thresholds_.tolist(), iso.y_thresholds_.tolist()


def _apply_one_isotonic(x_thresholds: list[float], y_values: list[float], raw: np.ndarray) -> np.ndarray:
    return np.interp(raw, x_thresholds, y_values, left=y_values[0], right=y_values[-1])


def fit_isotonic_calibrator(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    label_mode: str = "binary",
) -> CalibrationArtifact:
    """Fit isotonic calibration on existing OOF predictions.

    Parameters
    ----------
    y_true : (n,) int array of true labels (0/1 for binary; 0..K-1 for multiclass)
    y_prob : (n,) array of raw P(positive) for binary, or (n, K) array of
        raw per-class probabilities for multiclass.
    label_mode : "binary" or "multiclass"
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if label_mode == "binary":
        if y_prob.ndim != 1:
            raise ValueError("Binary calibration expects a 1-D y_prob array of P(positive).")
        x_thr, y_val = _fit_one_isotonic(y_prob, y_true.astype(float))
        return CalibrationArtifact(
            label_mode="binary", n_classes=2, isotonic_maps={"_binary": (x_thr, y_val)}
        )

    if y_prob.ndim != 2:
        raise ValueError("Multiclass calibration expects a 2-D (n, K) y_prob array.")
    n_classes = y_prob.shape[1]
    maps: dict[str, tuple[list[float], list[float]]] = {}
    for k in range(n_classes):
        target_k = (y_true == k).astype(float)
        maps[str(k)] = _fit_one_isotonic(y_prob[:, k], target_k)
    return CalibrationArtifact(label_mode="multiclass", n_classes=n_classes, isotonic_maps=maps)


def apply_calibrator(artifact: CalibrationArtifact, y_prob: np.ndarray) -> np.ndarray:
    """Apply a fitted :class:`CalibrationArtifact` to new raw probabilities."""
    y_prob = np.asarray(y_prob, dtype=float)

    if artifact.label_mode == "binary":
        x_thr, y_val = artifact.isotonic_maps["_binary"]
        return _apply_one_isotonic(x_thr, y_val, y_prob)

    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(1, -1)
    calibrated = np.zeros_like(y_prob, dtype=float)
    for k in range(artifact.n_classes):
        x_thr, y_val = artifact.isotonic_maps[str(k)]
        calibrated[:, k] = _apply_one_isotonic(x_thr, y_val, y_prob[:, k])
    # Renormalize so calibrated one-vs-rest scores sum to 1 per row.
    row_sums = calibrated.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return calibrated / row_sums


# ---------------------------------------------------------------------------
# Split-conformal prediction sets
# ---------------------------------------------------------------------------

@dataclass
class ConformalArtifact:
    """A fitted split-conformal threshold, serializable to/from JSON."""

    label_mode: str
    alpha: float
    q_hat: float
    n_calibration: int
    n_classes: int

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "ConformalArtifact":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.setdefault("n_classes", 2)  # backward-compat with artifacts fit before this field existed
        return cls(**data)


def fit_conformal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    alpha: float = 0.1,
    label_mode: str = "multiclass",
) -> ConformalArtifact:
    """Fit a split-conformal nonconformity threshold on existing OOF predictions.

    Uses the standard "1 - probability of the true class" nonconformity
    score (Sadinle et al. 2019 / Angelopoulos & Bates 2021 style). The
    OOF set doubles as the conformal calibration set — this is standard
    practice for LOSO/cross-validated predictions where a separate held-out
    calibration split isn't available; it is a slight optimism relative to
    a fully disjoint calibration set, and should be treated as approximate
    coverage rather than an exact guarantee for that reason.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        # Binary: build a (n, 2) matrix [P(neg), P(pos)] for a uniform
        # nonconformity-score computation below.
        y_prob = np.stack([1.0 - y_prob, y_prob], axis=1)

    n = len(y_true)
    true_class_prob = y_prob[np.arange(n), y_true]
    nonconformity = 1.0 - true_class_prob

    # Finite-sample-corrected quantile level (Angelopoulos & Bates 2021, eq. 1).
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(nonconformity, q_level, method="higher"))

    return ConformalArtifact(
        label_mode=label_mode,
        alpha=alpha,
        q_hat=q_hat,
        n_calibration=n,
        n_classes=y_prob.shape[1],
    )


def conformal_prediction_set(artifact: ConformalArtifact, y_prob: np.ndarray) -> list[list[int]]:
    """Return, for each row, the set of class indices whose probability is
    high enough to include under the fitted conformal threshold.

    A class ``k`` is included if ``1 - y_prob[:, k] <= q_hat``, i.e. its
    probability is at least ``1 - q_hat``. The returned set is guaranteed
    non-empty (if every class would otherwise be excluded, the single
    highest-probability class is kept, matching standard conformal
    practice of never returning an empty prediction set).
    """
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim == 1:
        y_prob = np.stack([1.0 - y_prob, y_prob], axis=1)

    if y_prob.shape[1] != artifact.n_classes:
        raise ValueError(
            f"conformal_prediction_set: probability vector has {y_prob.shape[1]} "
            f"classes but this artifact was fit on {artifact.n_classes} classes — "
            "these are not comparable. Refusing to guess."
        )

    sets: list[list[int]] = []
    for row in y_prob:
        included = [k for k, p in enumerate(row) if (1.0 - p) <= artifact.q_hat]
        if not included:
            included = [int(np.argmax(row))]
        sets.append(included)
    return sets


def coverage_report(
    artifact: ConformalArtifact,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, Any]:
    """Empirical coverage/set-size diagnostic on a held-out (or OOF) set —
    useful as a sanity check that the fitted threshold behaves as expected
    before trusting it in the API.
    """
    y_true = np.asarray(y_true, dtype=int)
    sets = conformal_prediction_set(artifact, y_prob)
    covered = [y_true[i] in s for i, s in enumerate(sets)]
    return {
        "target_coverage": 1.0 - artifact.alpha,
        "empirical_coverage": float(np.mean(covered)),
        "mean_set_size": float(np.mean([len(s) for s in sets])),
        "n_evaluated": len(y_true),
    }
