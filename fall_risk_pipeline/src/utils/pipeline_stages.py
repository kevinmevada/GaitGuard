"""
Canonical pipeline stage list and dependency order.

``python main.py`` (or ``--stage all``) runs every stage in this sequence.
Downstream stages assume upstream artifacts exist — do not reorder without
checking dependencies.
"""

from __future__ import annotations

# Ingest → features → supervised → baselines → primary anomaly → analytics → report
PIPELINE_STAGES: tuple[str, ...] = (
    "discover",
    "validate_raw",
    "ingest",
    "validate_gait_events",
    "preprocess",
    "eda",
    "features",
    "phase3_features",
    "select_features",
    "train",
    "evaluate",
    "train_deep",
    "ablation",
    "sensor_ablation",
    "classical_baselines",
    "anomaly",
    "dl_baselines",
    "competitor_metrics",
    "severity_regression",
    "statistical_benchmark",
    "compute_overhead",
    "novelty_table",
    "per_cohort_loso",
    "fall_risk_spearman",
    "cross_cohort",
    "predict",
    "report",
)

# Stages that require ``anomaly`` (BiLSTM-AE LOSO OOF scores / metrics).
POST_ANOMALY_STAGES: frozenset[str] = frozenset(
    {
        "dl_baselines",
        "competitor_metrics",
        "statistical_benchmark",
        "per_cohort_loso",
        "fall_risk_spearman",
        "report",
    }
)


def resolve_stages(request: str) -> list[str]:
    """Resolve CLI ``--stage`` argument to an ordered stage list."""
    if request == "all":
        return list(PIPELINE_STAGES)
    if request in PIPELINE_STAGES:
        return [request]
    if "," in request:
        names = [s.strip() for s in request.split(",") if s.strip()]
        unknown = [s for s in names if s not in PIPELINE_STAGES]
        if unknown:
            raise ValueError(f"Unknown stage(s): {unknown}. Valid: {list(PIPELINE_STAGES)}")
        return [s for s in PIPELINE_STAGES if s in names]
    raise ValueError(f"Unknown stage: {request!r}. Use 'all', a stage name, or comma-separated list.")


def validate_stage_order(stages: list[str]) -> list[str]:
    """Warn when a stage runs before its dependencies (partial runs)."""
    warnings: list[str] = []
    seen: set[str] = set()
    for stage in stages:
        if stage in POST_ANOMALY_STAGES and "anomaly" not in seen and "anomaly" in stages:
            idx = stages.index(stage)
            if "anomaly" not in stages[:idx]:
                warnings.append(
                    f"Stage '{stage}' is scheduled before 'anomaly' — "
                    "BiLSTM-AE artifacts may be missing."
                )
        seen.add(stage)
    return warnings
