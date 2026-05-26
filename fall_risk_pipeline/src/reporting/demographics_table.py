"""
Table 1 — participant demographics by cohort from trial_metadata.csv.

Uses one row per participant (deduplicated on participant_id within cohort).
Trial counts are reported separately per cohort.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

COHORT_ORDER = [
    "Healthy",
    "HipOA",
    "KneeOA",
    "ACL",
    "PD",
    "CVA",
    "CIPN",
    "RIL",
    "Unknown",
]

TABLE1_LATEX_TEMPLATE = r"""
\begin{table}[!t]
\renewcommand{\arraystretch}{1.15}
\caption{Participant demographics by cohort (Table~1). Age: mean $\pm$ SD (years). Sex: female/male counts with female proportion among participants with known sex. Laterality: affected or dominant side when recorded in trial metadata.}
\label{tab:demographics}
\centering
\small
\begin{tabular}{lrrlll}
\hline\hline
\textbf{Cohort} & \textbf{$n$} & \textbf{Trials} & \textbf{Age (years)} & \textbf{Sex (F/M)} & \textbf{Laterality} \\
\hline
ROWS
\hline\hline
\end{tabular}
\end{table}
"""


def trial_metadata_path(config: dict) -> Path:
    return Path(config["paths"]["processed_data"]) / "trial_metadata.csv"


def load_trial_metadata(config: dict) -> pd.DataFrame:
    path = trial_metadata_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"trial_metadata.csv not found at {path}. Run ingest first."
        )
    return pd.read_csv(path)


def normalize_sex(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "unknown", "na", "n/a", ""}:
        return None
    if text in {"f", "female", "woman", "w", "1"}:
        return "F"
    if text in {"m", "male", "man", "2"}:
        return "M"
    if text in {"other", "o", "non-binary", "nb"}:
        return "Other"
    return text[:1].upper() if len(text) == 1 else text.title()


def normalize_laterality(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "unknown", "na", "n/a"}:
        return None
    low = text.lower().replace("_", " ").replace("-", " ")
    if "left" in low and "right" not in low:
        return "Left"
    if "right" in low and "left" not in low:
        return "Right"
    if "bilateral" in low or "both" in low:
        return "Bilateral"
    if "dominant" in low:
        return "Dominant"
    return text.title()


def participant_level_metadata(trial_df: pd.DataFrame) -> pd.DataFrame:
    """One row per participant with stable demographics within cohort."""
    work = trial_df.copy()
    if "participant_id" not in work.columns:
        work["participant_id"] = work.index.astype(str)

    for col in ("age", "sex", "laterality"):
        if col not in work.columns:
            work[col] = np.nan
        if col == "sex" and "gender" in work.columns:
            work[col] = work[col].fillna(work["gender"])

    work["sex_norm"] = work["sex"].map(normalize_sex)
    work["laterality_norm"] = work["laterality"].map(normalize_laterality)
    work["age"] = pd.to_numeric(work["age"], errors="coerce")

    rows = []
    for pid, grp in work.groupby("participant_id", dropna=False):
        cohort = str(grp["cohort"].iloc[0]) if "cohort" in grp.columns else "Unknown"
        age_vals = grp["age"].dropna()
        age = float(age_vals.iloc[0]) if len(age_vals) else np.nan

        sex_vals = grp["sex_norm"].dropna()
        sex = sex_vals.mode().iloc[0] if len(sex_vals) else None

        lat_vals = grp["laterality_norm"].dropna()
        laterality = lat_vals.mode().iloc[0] if len(lat_vals) else None

        rows.append({
            "participant_id": pid,
            "cohort": cohort,
            "age": age,
            "sex": sex,
            "laterality": laterality,
            "n_trials": int(len(grp)),
        })
    return pd.DataFrame(rows)


def _format_age_mean_sd(ages: pd.Series) -> str:
    ages = pd.to_numeric(ages, errors="coerce").dropna()
    if len(ages) == 0:
        return "—"
    if len(ages) == 1:
        return f"{ages.iloc[0]:.1f}"
    return f"{ages.mean():.1f} ± {ages.std(ddof=1):.1f}"


def _format_sex_ratio(sex: pd.Series) -> tuple[str, int, int, int]:
    """Return display string and counts (n_female, n_male, n_known)."""
    known = sex.dropna()
    known = known[known.isin(["F", "M", "Other"])]
    n_f = int((known == "F").sum())
    n_m = int((known == "M").sum())
    n_o = int((known == "Other").sum())
    n_known = n_f + n_m + n_o
    if n_known == 0:
        return "—", 0, 0, 0
    pct_f = 100.0 * n_f / n_known if n_known else 0.0
    parts = f"{n_f} F / {n_m} M"
    if n_o:
        parts += f" / {n_o} other"
    parts += f" ({pct_f:.0f}% F)"
    return parts, n_f, n_m, n_known


def _format_laterality(lateralities: pd.Series) -> str:
    known = lateralities.dropna()
    if len(known) == 0:
        return "—"
    counts = known.value_counts()
    parts = [f"{label} ({int(n)})" for label, n in counts.items()]
    return "; ".join(parts)


def build_demographics_by_cohort(trial_df: pd.DataFrame) -> pd.DataFrame:
    participants = participant_level_metadata(trial_df)
    rows: list[dict[str, Any]] = []

    cohorts = list(COHORT_ORDER)
    extra = sorted(
        c for c in participants["cohort"].unique() if c not in cohorts
    )
    cohorts.extend(extra)

    for cohort in cohorts:
        psub = participants[participants["cohort"] == cohort]
        if psub.empty:
            continue
        tsub = trial_df[trial_df["cohort"] == cohort] if "cohort" in trial_df.columns else psub
        sex_str, n_f, n_m, n_sex_known = _format_sex_ratio(psub["sex"])
        n_age_known = int(psub["age"].notna().sum())

        rows.append({
            "cohort": cohort,
            "n_participants": int(len(psub)),
            "n_trials": int(len(tsub)),
            "age_mean": float(psub["age"].mean()) if n_age_known else np.nan,
            "age_sd": float(psub["age"].std(ddof=1)) if n_age_known > 1 else np.nan,
            "age_mean_sd": _format_age_mean_sd(psub["age"]),
            "n_age_reported": n_age_known,
            "n_female": n_f,
            "n_male": n_m,
            "n_sex_reported": n_sex_known,
            "sex_ratio": sex_str,
            "laterality": _format_laterality(psub["laterality"]),
        })

    if not rows:
        return pd.DataFrame()

    table = pd.DataFrame(rows)

    total_p = participants
    total_t = trial_df
    sex_str, n_f, n_m, n_sex_known = _format_sex_ratio(total_p["sex"])
    total_row = {
        "cohort": "Total",
        "n_participants": int(len(total_p)),
        "n_trials": int(len(total_t)),
        "age_mean": float(total_p["age"].mean()) if total_p["age"].notna().any() else np.nan,
        "age_sd": float(total_p["age"].std(ddof=1)) if total_p["age"].notna().sum() > 1 else np.nan,
        "age_mean_sd": _format_age_mean_sd(total_p["age"]),
        "n_age_reported": int(total_p["age"].notna().sum()),
        "n_female": n_f,
        "n_male": n_m,
        "n_sex_reported": n_sex_known,
        "sex_ratio": sex_str,
        "laterality": _format_laterality(total_p["laterality"]),
    }
    return pd.concat([table, pd.DataFrame([total_row])], ignore_index=True)


def demographics_to_markdown(table: pd.DataFrame) -> str:
    lines = [
        "# Table 1 — Demographics by cohort",
        "",
        "Participant-level summary from `trial_metadata.csv` "
        "(one row per `participant_id`; trials counted separately).",
        "",
        "| Cohort | *n* (participants) | Trials | Age (years) | Sex (F/M) | Laterality |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.cohort} | {int(row.n_participants)} | {int(row.n_trials)} | "
            f"{row.age_mean_sd} | {row.sex_ratio} | {row.laterality} |"
        )
    lines.extend([
        "",
        "Age: mean ± SD among participants with recorded age. "
        "Sex: counts and female proportion among participants with recorded sex. "
        "Laterality: affected/dominant side when present in metadata.",
        "",
    ])
    return "\n".join(lines)


def demographics_to_latex(table: pd.DataFrame) -> str:
    rows = []
    for row in table.itertuples(index=False):
        cohort = str(row.cohort).replace("_", r"\_")
        if row.cohort == "Total":
            cohort = r"\textbf{Total}"
        age = str(row.age_mean_sd).replace("±", r"$\pm$")
        sex = str(row.sex_ratio).replace("%", r"\%")
        lat = str(row.laterality).replace("_", r"\_")
        rows.append(
            f"{cohort} & {int(row.n_participants)} & {int(row.n_trials)} & "
            f"{age} & {sex} & {lat} \\\\"
        )
    return TABLE1_LATEX_TEMPLATE.replace("ROWS", "\n".join(rows))


def save_demographics_table(
    table: pd.DataFrame,
    metrics_dir: Path,
    *,
    prefix: str = "table1",
) -> dict[str, Path]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": metrics_dir / f"{prefix}_demographics.csv",
        "md": metrics_dir / f"{prefix}_demographics.md",
        "tex": metrics_dir / f"{prefix}_demographics.tex",
    }
    table.to_csv(paths["csv"], index=False)
    paths["md"].write_text(demographics_to_markdown(table), encoding="utf-8")
    paths["tex"].write_text(demographics_to_latex(table), encoding="utf-8")
    return paths


def generate_demographics_table(config: dict) -> pd.DataFrame | None:
    """Load trial metadata, build Table 1, write to results/metrics."""
    try:
        trial_df = load_trial_metadata(config)
    except FileNotFoundError as exc:
        logger.warning(str(exc))
        return None

    if trial_df.empty:
        logger.warning("trial_metadata.csv is empty — skipping demographics table.")
        return None

    table = build_demographics_by_cohort(trial_df)
    metrics_dir = Path(config.get("paths", {}).get("metrics", "results/metrics"))
    paths = save_demographics_table(table, metrics_dir)
    total = table[table["cohort"] == "Total"].iloc[0]
    logger.info(
        f"Table 1 demographics: {int(total['n_participants'])} participants, "
        f"{int(total['n_trials'])} trials → {paths['csv']}"
    )
    return table
