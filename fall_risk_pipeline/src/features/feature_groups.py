"""
Map config feature groups → patient-level column names for ablation.

Patient columns are named ``{sensor_prefix}_{trial_base}_{agg_suffix}``
where agg_suffix ∈ {mean, std, range, trend}.  A trial base like ``lyapunov``
produces ``lb_lyapunov_mean``, ``lb_lyapunov_std``, etc.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_AGG_SUFFIXES = ("_mean", "_std", "_range", "_trend")
_CROSS_SITE_PREFIX = "head_lb_"


def trial_feature_groups(config: dict[str, Any]) -> dict[str, list[str]]:
    """Config feature groups with their trial-level base names (non-empty only)."""
    feat_cfg = config.get("features", {})
    groups: dict[str, list[str]] = {}
    for group_name in ("temporal", "spectral", "wavelet", "trunk_dynamics",
                       "orientation", "asymmetry", "turning", "spatial",
                       "phase2_kinematic", "phase3_deep_features"):
        bases = feat_cfg.get(group_name, [])
        if bases:
            groups[group_name] = list(bases)
    return groups


def count_trial_features(config: dict[str, Any]) -> int:
    return sum(len(v) for v in trial_feature_groups(config).values())


def patient_columns_for_trial_base(column: str, base: str) -> bool:
    """True if *column* (patient-level) derives from a trial-level *base* feature.

    Handles sensor-prefixed naming like ``lb_lyapunov_mean`` matching base ``lyapunov``,
    and un-prefixed names like ``cadence_mean`` matching base ``cadence``.

    Cross-site ``head_lb_*`` features (e.g. ``head_lb_rms_ratio_mean``) require an
    exact stem match so shorter trunk bases like ``lyapunov`` do not subsume
    ``head_lb_lyapunov_ratio`` during ablation.
    """
    for suffix in _AGG_SUFFIXES:
        if column.endswith(suffix):
            stem = column[: -len(suffix)]
            if stem.startswith(_CROSS_SITE_PREFIX) or base.startswith(_CROSS_SITE_PREFIX):
                return stem == base
            if stem == base or stem.endswith(f"_{base}"):
                return True
    return False


def patient_columns_for_trial_bases(
    all_columns: list[str],
    trial_bases: list[str],
) -> list[str]:
    """Return patient columns matching any of the given trial-level bases."""
    return [
        col for col in all_columns
        if any(patient_columns_for_trial_base(col, base) for base in trial_bases)
    ]


def patient_columns_for_group(
    all_columns: list[str],
    group_name: str,
    config: dict[str, Any],
) -> list[str]:
    """Patient columns belonging to a feature group (e.g. 'spectral')."""
    groups = trial_feature_groups(config)
    bases = groups.get(group_name, [])
    return patient_columns_for_trial_bases(all_columns, bases)


def patient_columns_minus_group(
    all_columns: list[str],
    group_name: str,
    config: dict[str, Any],
) -> list[str]:
    """All patient columns *except* those in the named group."""
    drop = set(patient_columns_for_group(all_columns, group_name, config))
    return [c for c in all_columns if c not in drop]


def patient_columns_minus_trial_bases(
    all_columns: list[str],
    trial_bases: list[str],
) -> list[str]:
    drop = set(patient_columns_for_trial_bases(all_columns, trial_bases))
    return [c for c in all_columns if c not in drop]


def summarize_ablation_groups(
    all_columns: list[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    """DataFrame summarizing how many patient columns each group produces."""
    rows = []
    for group_name in trial_feature_groups(config):
        cols = patient_columns_for_group(all_columns, group_name, config)
        rows.append({
            "group": group_name,
            "n_trial_features": len(trial_feature_groups(config)[group_name]),
            "n_patient_columns": len(cols),
            "columns": ", ".join(sorted(cols)),
        })
    return pd.DataFrame(rows)
