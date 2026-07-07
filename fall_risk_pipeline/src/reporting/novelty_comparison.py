"""
Section 2 novelty comparison — methodological firsts vs competitor literature.

GaitGuard's three unambiguous firsts (manuscript bullets):
  1. First strict subject-level LOSO on the full 8-cohort Voisard dataset.
  2. First 3-method one-class ensemble (BiLSTM-AE + Isolation Forest + OC-SVM) under LOSO.
  3. First zero-shot cross-dataset generalization to DAPHNET FOG (4-sensor train → 1-sensor LB eval).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

GAITGUARD_STUDY = "GaitGuard (this work)"


@dataclass(frozen=True)
class NoveltyRow:
    study: str
    year: int
    dataset: str
    strict_loso: bool
    one_class_ensemble: bool
    cross_dataset_eval: bool
    cohort_count: int | str
    notes: str = ""

    @property
    def all_firsts(self) -> bool:
        return self.strict_loso and self.one_class_ensemble and self.cross_dataset_eval


def _mark(value: bool) -> str:
    return "✓" if value else "—"


# Comparator papers aligned with pipeline baselines (classical + DL + severity).
# Boolean flags reflect published methodology vs GaitGuard claims — not re-implemented here.
NOVELTY_COMPARATORS: tuple[NoveltyRow, ...] = (
    NoveltyRow(
        study="Moon et al.",
        year=2020,
        dataset="Single-site IMU gait (PD vs healthy)",
        strict_loso=False,
        one_class_ensemble=False,
        cross_dataset_eval=False,
        cohort_count=2,
        notes="Handcrafted IMU features + supervised classifiers; trial/random splits reported.",
    ),
    NoveltyRow(
        study="Trabassi et al.",
        year=2022,
        dataset="PD gait cohort",
        strict_loso=False,
        one_class_ensemble=False,
        cross_dataset_eval=False,
        cohort_count=1,
        notes="Kinematic IMU features; disease-specific endpoint, not 8-cohort screening.",
    ),
    NoveltyRow(
        study="Dempster et al. (ROCKET)",
        year=2019,
        dataset="UCR/UEA time-series archive",
        strict_loso=False,
        one_class_ensemble=False,
        cross_dataset_eval=False,
        cohort_count="117 datasets",
        notes="Classifier benchmark method paper; per-dataset train/test splits, not clinical LOSO.",
    ),
    NoveltyRow(
        study="Dempster et al. (MINIROCKET)",
        year=2021,
        dataset="UCR/UEA time-series archive",
        strict_loso=False,
        one_class_ensemble=False,
        cross_dataset_eval=False,
        cohort_count="117 datasets",
        notes="Fast ROCKET variant; archive classification, no wearable pathology tiers.",
    ),
    NoveltyRow(
        study="Ismail Fawaz et al. (InceptionTime)",
        year=2020,
        dataset="UCR/UEA time-series archive",
        strict_loso=False,
        one_class_ensemble=False,
        cross_dataset_eval=False,
        cohort_count="128 datasets",
        notes="Deep CNN benchmark leader; fixed archive splits, not multi-cohort IMU screening.",
    ),
    NoveltyRow(
        study="Ordóñez & Roggen (DeepConvLSTM)",
        year=2016,
        dataset="OPPORTUNITY / PAMAP2 HAR",
        strict_loso=False,
        one_class_ensemble=False,
        cross_dataset_eval=False,
        cohort_count="activity classes",
        notes="Supervised HAR on wrist/body sensors; different task and label granularity.",
    ),
    NoveltyRow(
        study="Navita et al.",
        year=2025,
        dataset="Gait clinic (UPDRS regression)",
        strict_loso=False,
        one_class_ensemble=False,
        cross_dataset_eval=False,
        cohort_count="≤3",
        notes="AdaBoost/GB ensemble on gait features → UPDRS; supervised regression, not one-class LOSO.",
    ),
    NoveltyRow(
        study="Sadeghsalehi et al.",
        year=2025,
        dataset="Clinical gait (imbalanced screening)",
        strict_loso=False,
        one_class_ensemble=False,
        cross_dataset_eval=False,
        cohort_count="≤4",
        notes="MCC-focused screening; no 8-cohort Voisard LOSO or cross-dataset FOG transfer reported.",
    ),
    NoveltyRow(
        study=GAITGUARD_STUDY,
        year=2026,
        dataset="Voisard 8-cohort + DAPHNET FOG (zero-shot)",
        strict_loso=True,
        one_class_ensemble=True,
        cross_dataset_eval=True,
        cohort_count=8,
        notes=(
            "Healthy-only one-class training; 3-method latent ensemble; sealed DAPHNET eval with "
            "4-sensor train → lower-back-only zero-padded transfer (asymmetric sensor layout)."
        ),
    ),
)


def novelty_dataframe(rows: tuple[NoveltyRow, ...] | None = None) -> pd.DataFrame:
    data = rows or NOVELTY_COMPARATORS
    records = []
    for row in data:
        records.append(
            {
                "study": row.study,
                "year": row.year,
                "dataset": row.dataset,
                "strict_loso": row.strict_loso,
                "one_class_ensemble": row.one_class_ensemble,
                "cross_dataset_eval": row.cross_dataset_eval,
                "cohort_count": row.cohort_count,
                "all_methodological_firsts": row.all_firsts,
                "notes": row.notes,
            }
        )
    return pd.DataFrame(records)


def gaitguard_unique_full_firsts(df: pd.DataFrame | None = None) -> bool:
    frame = df if df is not None else novelty_dataframe()
    full = frame[frame["all_methodological_firsts"]]
    return len(full) == 1 and str(full.iloc[0]["study"]) == GAITGUARD_STUDY


def three_firsts_bullets() -> list[str]:
    return [
        (
            "**First strict LOSO on full 8-cohort Voisard.** No prior wearable gait paper evaluates "
            "all eight Voisard pathology cohorts (Healthy, HipOA, KneeOA, ACL, PD, CVA, CIPN, RIL) "
            "under leave-one-subject-out holdout."
        ),
        (
            "**First 3-method one-class ensemble under LOSO.** BiLSTM-AE reconstruction + Isolation Forest "
            "on latent activations + one-class SVM boundary distance, trained on healthy gait only per fold."
        ),
        (
            "**First zero-shot cross-dataset FOG transfer in this comparator set.** Sealed DAPHNET "
            "freezing-of-gait evaluation with asymmetric sensing: four-sensor Voisard training → "
            "single lower-back sensor at test time (zero-padded layout), which is strictly harder than "
            "matched-sensor transfer."
        ),
    ]


def render_novelty_markdown(df: pd.DataFrame | None = None) -> str:
    frame = df if df is not None else novelty_dataframe()
    lines = [
        "# Table 1 — Methodological novelty vs competitor literature (Section 2)",
        "",
        "Comparison of **evaluation rigor** features across wearable gait competitors benchmarked "
        "in GaitGuard. Numeric performance lives in Table 2 (`docs/paper/table2_prior_work.md`).",
        "",
        "| Study | Year | Dataset | Strict LOSO | 3-method one-class ensemble | Cross-dataset eval | Cohorts |",
        "|---|---:|---|:---:|:---:|:---:|---|",
    ]
    for row in frame.itertuples(index=False):
        highlight = "**" if str(row.study) == GAITGUARD_STUDY else ""
        end = "**" if highlight else ""
        lines.append(
            f"| {highlight}{row.study}{end} | {int(row.year)} | {row.dataset} | "
            f"{_mark(bool(row.strict_loso))} | {_mark(bool(row.one_class_ensemble))} | "
            f"{_mark(bool(row.cross_dataset_eval))} | {row.cohort_count} |"
        )
    lines.extend(["", "## Three unambiguous firsts (GaitGuard only)", ""])
    lines.extend(f"- {b}" for b in three_firsts_bullets())
    lines.extend(
        [
            "",
            "## Footnotes",
            "",
            "- **Strict LOSO:** leave-one-participant-out; no trial from the held-out subject appears in training.",
            "- **3-method one-class ensemble:** BiLSTM-AE + Isolation Forest (latent) + one-class SVM (latent); "
            "pathological gait never used for manifold fitting.",
            "- **Cross-dataset eval:** train on Voisard, evaluate on an external dataset without target-domain "
            "retraining (DAPHNET FOG).",
            "- Competitor flags reflect **published protocols** for the cited benchmark papers, not re-runs on Voisard.",
            "",
        ]
    )
    if not gaitguard_unique_full_firsts(frame):
        lines.append(
            "_Warning: GaitGuard is not the sole row with all three checkmarks — review `NOVELTY_COMPARATORS`._\n"
        )
    return "\n".join(lines)


def write_novelty_artifacts(
    metrics_dir: Path,
    *,
    paper_docs_dir: Path | None = None,
) -> dict[str, Any]:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    df = novelty_dataframe()

    csv_path = metrics_dir / "novelty_comparison_table.csv"
    md_metrics_path = metrics_dir / "novelty_comparison_table.md"
    json_path = metrics_dir / "novelty_comparison_summary.json"

    df.to_csv(csv_path, index=False)
    md_body = render_novelty_markdown(df)
    md_metrics_path.write_text(md_body, encoding="utf-8")

    summary = {
        "gaitguard_unique_full_firsts": gaitguard_unique_full_firsts(df),
        "three_firsts": three_firsts_bullets(),
        "n_comparators": int(len(df)),
        "artifacts": {
            "csv": str(csv_path),
            "markdown": str(md_metrics_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    paper_path = None
    if paper_docs_dir is not None:
        paper_docs_dir.mkdir(parents=True, exist_ok=True)
        paper_path = paper_docs_dir / "table1_novelty_comparison.md"
        paper_path.write_text(md_body, encoding="utf-8")
        summary["artifacts"]["paper_markdown"] = str(paper_path)

    return summary
