"""
Generate publication-facing summary artifacts from evaluation outputs.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from loguru import logger


LATEX_TEMPLATE = r"""
\begin{table}[!t]
\renewcommand{\arraystretch}{1.2}
\caption{Fall risk model performance (Table~2; subject-grouped validation). $p_{\mathrm{DeLong}}$: DeLong test vs REFERENCE_MODEL (Sun \& Xu 2014). $p_{\mathrm{McNemar}}$: McNemar test on paired LOSO classifications (statsmodels). Demographics: Table~\ref{tab:demographics}.}
\label{tab:model_performance}
\centering
\begin{tabular}{lcccccc}
\hline\hline
\textbf{Model} & \textbf{AUC} & \textbf{Acc.} & \textbf{F1} & \textbf{Sens.} & \textbf{$p_{\mathrm{DeLong}}$} & \textbf{$p_{\mathrm{McNemar}}$} \\
\hline
ROWS
\hline\hline
\end{tabular}
\end{table}
"""


class ReportGenerator:

    def __init__(self, config: dict):
        self.config = config
        self.metrics_dir = Path(config.get("paths", {}).get("metrics", "results/metrics"))
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        self._regenerate_demographics()
        self._ensure_significance_pvalues()

        metrics_path = self.metrics_dir / "metrics.csv"
        if not metrics_path.exists():
            logger.warning(f"{metrics_path} not found. Run evaluation first.")
            return

        df = pd.read_csv(metrics_path)
        if df.empty:
            logger.warning("metrics.csv is empty.")
            return

        required_cols = [
            "model",
            "auc",
            "accuracy",
            "f1",
            "sensitivity",
            "validation_strategy",
            "participants",
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0 if col not in ("model", "validation_strategy") else "unknown"

        for col in ["auc", "accuracy", "f1", "sensitivity", "participants", "p_delong_vs_best"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "p_delong_fmt" not in df.columns:
            df["p_delong_fmt"] = df.get("p_delong_vs_best", pd.Series(dtype=float)).map(
                self._format_pvalue_cell
            )
        if "p_mcnemar_fmt" not in df.columns:
            df["p_mcnemar_fmt"] = df.get("p_mcnemar_vs_best", pd.Series(dtype=float)).map(
                self._format_pvalue_cell
            )

        df = df.sort_values("auc", ascending=False)
        self._generate_latex_table(df)
        self._generate_markdown_report(df)

    @staticmethod
    def _format_pvalue_cell(p: float) -> str:
        if not pd.notna(p):
            return "—"
        if p >= 0.999:
            return "ref"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    def _generate_latex_table(self, df: pd.DataFrame):
        rows = []
        best_model = df.iloc[0]["model"]
        ref_label = str(df.iloc[0].get("auc_reference_model", best_model)).replace("_", " ")

        for row in df.itertuples(index=False):
            name = str(row.model).replace("_", " ").title()
            if row.model == best_model:
                name = r"\textbf{" + name + "}"

            p_delong = getattr(row, "p_delong_fmt", None)
            if p_delong is None or (isinstance(p_delong, float) and not pd.notna(p_delong)):
                p_delong = self._format_pvalue_cell(getattr(row, "p_delong_vs_best", float("nan")))
            else:
                p_delong = str(p_delong)

            p_mcnemar = getattr(row, "p_mcnemar_fmt", None)
            if p_mcnemar is None or (isinstance(p_mcnemar, float) and not pd.notna(p_mcnemar)):
                p_mcnemar = self._format_pvalue_cell(getattr(row, "p_mcnemar_vs_best", float("nan")))
            else:
                p_mcnemar = str(p_mcnemar)

            rows.append(
                f"{name} & "
                f"{float(row.auc):.3f} & "
                f"{float(row.accuracy):.3f} & "
                f"{float(row.f1):.3f} & "
                f"{float(row.sensitivity):.3f} & "
                f"{p_delong} & "
                f"{p_mcnemar} \\\\"
            )

        table = (
            LATEX_TEMPLATE.replace("ROWS", "\n".join(rows))
            .replace("REFERENCE_MODEL", ref_label)
        )
        path = self.metrics_dir / "ieee_table.tex"
        path.write_text(table, encoding="utf-8")
        logger.info(f"LaTeX table saved -> {path}")

    def _generate_markdown_report(self, df: pd.DataFrame):
        best = df.iloc[0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        from src.dataset.label_policy import label_mode_description

        label_mode = self.config.get("dataset", {}).get("label_mode", "unknown")
        label_note = label_mode_description(self.config)
        validation_strategy = str(best.get("validation_strategy", "unknown"))
        participants = int(float(best.get("participants", 0)))

        model = str(best.get("model", "unknown"))
        auc = float(best.get("auc", 0))
        acc = float(best.get("accuracy", 0))
        f1 = float(best.get("f1", 0))
        sens = float(best.get("sensitivity", 0))

        feature_section = self._feature_selection_section()
        ablation_section = self._ablation_section()
        clinical_threshold_section = self._clinical_threshold_section()
        threshold_comparison_section = self._threshold_comparison_section()
        ethics_section = self._ethics_section()
        limitations_section = self._limitations_section()
        ensemble_section = self._ensemble_comparison_section()
        leakage_section = self._leakage_comparison_section()
        sensor_ablation_section = self._sensor_ablation_section()
        cross_cohort_section = self._cross_cohort_section()
        dl_section = self._deep_learning_comparison_section()
        class_section = self._append_class_distribution_section()
        inference_section = self._deployment_inference_section()
        demographics_section = self._demographics_section()

        report = f"""# Pathology-Tier Gait Screening Pipeline - Results Report
Generated: {timestamp}

## Dataset
- Participants: {participants}
- Sensors: 4 IMUs
- Label mode: {label_mode} — {label_note}

{demographics_section}
{class_section}
{feature_section}
{ablation_section}
{clinical_threshold_section}
{threshold_comparison_section}
{ethics_section}
{limitations_section}
{ensemble_section}
{dl_section}
{sensor_ablation_section}
{cross_cohort_section}
{inference_section}
## Validation
- Strategy: {validation_strategy}
- Classification threshold: argmax (multiclass) or Youden J per LOSO train fold (binary; `threshold_source=unbiased_train_fold`).
- **Primary Table 2 metrics use unbiased train-fold Youden only.** Do **not** cite `optimistic_eval_set` rows or `accuracy_eval_youden` as generalization performance (eval-set Youden is optimistic threshold tuning on pooled OOF labels).
- Supplemental baselines: `metrics_threshold_comparison.csv` (`fixed_0.5`, `optimistic_eval_set`).
- Note: Reported metrics are intended for subject-grouped evaluation output, not in-sample prediction export.

## Model Performance (Table 2)

| Model | AUC | Accuracy | F1 | Sensitivity | p (DeLong) | p (McNemar) |
|---|---|---|---|---:|---:|---:|
"""

        for row in df.itertuples(index=False):
            name = str(row.model)
            mark = " *" if name == model else ""
            p_delong = getattr(row, "p_delong_fmt", None)
            if p_delong is None or (isinstance(p_delong, float) and not pd.notna(p_delong)):
                p_delong = self._format_pvalue_cell(getattr(row, "p_delong_vs_best", float("nan")))
            else:
                p_delong = str(p_delong)
            p_mcnemar = getattr(row, "p_mcnemar_fmt", None)
            if p_mcnemar is None or (isinstance(p_mcnemar, float) and not pd.notna(p_mcnemar)):
                p_mcnemar = self._format_pvalue_cell(getattr(row, "p_mcnemar_vs_best", float("nan")))
            else:
                p_mcnemar = str(p_mcnemar)
            report += (
                f"| {name}{mark} | "
                f"{float(row.auc):.3f} | "
                f"{float(row.accuracy):.3f} | "
                f"{float(row.f1):.3f} | "
                f"{float(row.sensitivity):.3f} | "
                f"{p_delong} | "
                f"{p_mcnemar} |\n"
            )

        report += f"""

## Best Model
**{model}**

- AUC: **{auc:.4f}**
- Accuracy: **{acc:.4f}**
- F1 Score: **{f1:.4f}**
- Sensitivity: **{sens:.4f}**

## Outputs
- table1_demographics.csv / .md / .tex
- metrics.csv
- predictions.csv
- SHAP plots
- ROC / PR curves

## Subject-Leakage Comparison (Grouped vs Ungrouped CV)

{leakage_section}

## Reproducibility

python main.py --config configs/pipeline_config.yaml
"""

        path = self.metrics_dir / "pipeline_report.md"
        path.write_text(report, encoding="utf-8")
        logger.info(f"Markdown report saved -> {path}")

    def _regenerate_demographics(self) -> None:
        """(Re)generate Table 1 demographics from trial_metadata.csv."""
        try:
            from src.reporting.demographics_table import generate_demographics_table
            table = generate_demographics_table(self.config)
            if table is not None:
                logger.info("Table 1 demographics (re)generated in report stage.")
            else:
                logger.warning(
                    "Demographics table could not be generated — "
                    "trial_metadata.csv missing or empty. Run ingest first."
                )
        except Exception as exc:
            logger.warning(f"Demographics table generation failed: {exc}")

    def _ensure_significance_pvalues(self) -> None:
        """Recompute DeLong / McNemar p-values from saved OOF predictions if missing."""
        label_mode = self.config.get("dataset", {}).get("label_mode", "multiclass")
        if label_mode != "binary":
            logger.info(
                "Skipping DeLong/McNemar recomputation in report (binary-only tests; "
                f"label_mode={label_mode})."
            )
            return

        metrics_path = self.metrics_dir / "metrics.csv"
        oof_path = self.metrics_dir / "oof_predictions.parquet"
        if not metrics_path.exists() or not oof_path.exists():
            return

        df = pd.read_csv(metrics_path)
        need_delong = (
            "p_delong_vs_best" not in df.columns or not df["p_delong_vs_best"].notna().any()
        )
        need_mcnemar = (
            "p_mcnemar_vs_best" not in df.columns or not df["p_mcnemar_vs_best"].notna().any()
        )
        if not need_delong and not need_mcnemar:
            return

        oof = pd.read_parquet(oof_path)
        results: dict = {}
        for model in oof["model"].unique():
            sub = oof[oof["model"] == model]
            y_true = sub["y_true"].values
            y_prob = sub["y_prob"].values
            from sklearn.metrics import roc_auc_score, accuracy_score

            entry: dict = {
                "y_true": y_true,
                "y_prob": y_prob,
                "auc": float(roc_auc_score(y_true, y_prob)),
                "accuracy": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
            }
            if "y_pred" in sub.columns:
                entry["y_pred"] = sub["y_pred"].values.astype(int)
            if "participant_id" in sub.columns:
                entry["participant_ids"] = sub["participant_id"].values
            results[model] = entry

        eval_cfg = self.config.get("models", {}).get("evaluation", {})

        if need_delong:
            from src.evaluation.auc_significance import pairwise_auc_significance

            ref = eval_cfg.get("auc_reference_model") or max(
                results, key=lambda n: results[n]["auc"]
            )
            n_boot = int(eval_cfg.get("delong_bootstrap_n", 1000))
            seed = int(eval_cfg.get("random_state", 42))
            pairwise_df, vs_ref_df = pairwise_auc_significance(
                results, reference=ref, n_bootstrap=n_boot, random_state=seed
            )
            if not pairwise_df.empty:
                pairwise_df.to_csv(self.metrics_dir / "auc_pairwise_pvalues.csv", index=False)
            if not vs_ref_df.empty:
                vs_ref_df.to_csv(self.metrics_dir / "auc_vs_best_pvalues.csv", index=False)
                p_map = vs_ref_df.set_index("model").to_dict(orient="index")
                for i, row in df.iterrows():
                    pvals = p_map.get(row["model"], {})
                    df.at[i, "p_delong_vs_best"] = pvals.get("p_delong_vs_reference", float("nan"))
                    df.at[i, "p_bootstrap_mwu_vs_best"] = pvals.get(
                        "p_bootstrap_mwu_vs_reference", float("nan")
                    )
                    df.at[i, "auc_reference_model"] = pvals.get("reference_model", ref)
                    df.at[i, "p_delong_fmt"] = pvals.get("p_delong_fmt", "")

        if need_mcnemar:
            from src.evaluation.classification_significance import (
                pairwise_classification_significance,
            )

            ref = eval_cfg.get("mcnemar_reference_model") or eval_cfg.get("auc_reference_model")
            if not ref:
                ref = max(results, key=lambda n: results[n]["accuracy"])
            exact = bool(eval_cfg.get("mcnemar_exact", False))
            m_pairwise, m_vs_ref, m_folds = pairwise_classification_significance(
                results, reference=ref, exact_mcnemar=exact
            )
            if not m_pairwise.empty:
                m_pairwise.to_csv(self.metrics_dir / "mcnemar_pairwise_pvalues.csv", index=False)
            if not m_folds.empty:
                m_folds.to_csv(self.metrics_dir / "mcnemar_fold_discordant.csv", index=False)
            if not m_vs_ref.empty:
                m_vs_ref.to_csv(self.metrics_dir / "mcnemar_vs_best_pvalues.csv", index=False)
                m_map = m_vs_ref.set_index("model").to_dict(orient="index")
                for i, row in df.iterrows():
                    mvals = m_map.get(row["model"], {})
                    df.at[i, "p_mcnemar_vs_best"] = mvals.get("p_mcnemar_vs_reference", float("nan"))
                    df.at[i, "p_mcnemar_fmt"] = mvals.get("p_mcnemar_fmt", "")

        df.to_csv(metrics_path, index=False)
        logger.info("Merged significance p-values into metrics.csv from OOF predictions")

    def _demographics_section(self) -> str:
        md_path = self.metrics_dir / "table1_demographics.md"
        if not md_path.exists():
            return (
                "## Table 1 — Demographics\n\n"
                "_Not generated — run `ingest` to create `trial_metadata.csv`, "
                "then `report` (or ingest alone writes Table 1 to metrics)._\n\n"
            )
        return md_path.read_text(encoding="utf-8") + "\n"

    def _deployment_inference_section(self) -> str:
        """Paper-ready note: API single-trial inference vs patient-level training."""
        return (
            "## Deployment inference (API vs training)\n\n"
            "Training and LOSO evaluation use **patient-level** feature rows "
            "(N participants; trials aggregated with mean, std, range, trend per "
            "trial feature). The public `POST /predict` API accepts **one trial** per "
            "request and maps it into the same column schema (`_mean` = trial value; "
            "`_std` = 0; `_range` = 0; `_trend` = NaN). That projection is **not** "
            "equivalent to multi-trial patient aggregation.\n\n"
            "**Suggested paper wording (Limitations):**\n\n"
            "> Deployment inference accepted one uploaded trial per API request. To match "
            "the trained feature schema, trial values populated patient-level mean columns "
            "while standard deviation and range were set to zero and trend was undefined; "
            "scores therefore do not replicate full multi-trial patient aggregation used in "
            "training (multi-trial patient aggregation). Reported confidence reflects the "
            "model maximum class probability, not external clinical calibration.\n\n"
            "See `docs/inference_single_trial_limitation.md`. API responses include "
            "`inference_scope` and `limitations` fields.\n\n"
        )

    def _append_class_distribution_section(self) -> str:
        md_path = self.metrics_dir / "class_distribution_report.md"
        cohort_path = self.metrics_dir / "class_distribution_by_cohort.csv"
        if not md_path.exists():
            return ""

        text = md_path.read_text(encoding="utf-8")
        lines = [
            "## Class distribution (training labels)",
            "",
            f"Label policy: {self.config.get('dataset', {}).get('label_mode', 'unknown')} "
            "(see `docs/label_binning.md`). Multiclass tiers: 0=Healthy, "
            "1=orthopedic (HipOA/KneeOA/ACL), 2=neurological (PD/CVA/CIPN/RIL). "
            "Legacy binary at threshold≥1 conflates orthopedic and neurological risk.",
            "",
        ]
        if cohort_path.exists():
            dfc = pd.read_csv(cohort_path)
            lines.append("| Cohort | N | Multiclass | Train label | Binary≥1 | Binary≥2 | Fall % |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for row in dfc.itertuples(index=False):
                mc = getattr(row, "multiclass_label", getattr(row, "raw_multiclass_label", 0))
                train = getattr(row, "training_label", getattr(row, "binary_risk_label", 0))
                b1 = getattr(row, "binary_at_threshold_1", int(mc >= 1))
                b2 = getattr(row, "binary_at_threshold_2", int(mc >= 2))
                lines.append(
                    f"| {row.cohort} | {int(row.n_participants)} | "
                    f"{int(mc)} | {int(train)} | {int(b1)} | {int(b2)} | "
                    f"{float(row.reference_fall_probability_pct):.1f} |"
                )
            lines.append("")
        lines.append("See `class_distribution_report.md` for full counts.")
        lines.append("")
        return "\n".join(lines)

    def _append_feature_selection_section(self) -> None:
        fs_report = self.metrics_dir / "feature_selection_report.md"
        if fs_report.exists():
            logger.info(f"Feature selection report available -> {fs_report}")

    def _feature_selection_section(self) -> str:
        comp_path = self.metrics_dir / "feature_selection_comparison.csv"
        if not comp_path.exists():
            return ""

        dfc = pd.read_csv(comp_path)
        lines = [
            "## Feature selection (dimensionality control)",
            "",
            "With patient-level features (mean, std, range, trend per trial feature) "
            "the dimensionality may be high relative to sample size. RFECV (Guyon & Elisseeff, 2002) "
            "and SHAP pruning reduce p to ≤20 before final training; see Tibshirani (1996) for Lasso-style sparsity.",
            "",
            "| Stage | Features (p) | Grouped CV AUC |",
            "|---|---:|---:|",
        ]
        for row in dfc.itertuples(index=False):
            lines.append(
                f"| {row.stage} | {int(row.n_features)} | "
                f"{float(row.cv_auc_mean):.4f} ± {float(row.cv_auc_std):.4f} |"
            )
        lines.append("")
        lines.append("Full report: `feature_selection_report.md`")
        lines.append("")
        return "\n".join(lines)

    def _ablation_section(self) -> str:
        path = self.metrics_dir / "feature_ablation.md"
        if not path.exists():
            return ""
        body = path.read_text(encoding="utf-8").strip()
        return body + "\n"

    def _clinical_threshold_section(self) -> str:
        path = self.metrics_dir / "clinical_threshold.json"
        if not path.exists():
            return ""
        try:
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        pc = data.get("primary_cutoff", {})
        morse = data.get("clinical_screening_tools", {}).get("morse_fall_scale", {})
        strat = data.get("clinical_screening_tools", {}).get("stratify", {})
        lines = [
            "## Clinical threshold (Youden J)",
            "",
            "Risk classification uses a **data-driven Youden J cutoff** on LOSO "
            "out-of-fold elevated-risk probability — not fixed API score bands (70/40).",
            "",
            f"- **Primary cutoff (deployment):** P(elevated) ≥ **{pc.get('probability', float('nan')):.3f}** "
            f"(risk score ≥ {pc.get('risk_score_0_100', '—')})",
            f"- **Sensitivity** at primary cutoff: **{pc.get('sensitivity', float('nan')):.3f}**",
            f"- **Specificity** at primary cutoff: **{pc.get('specificity', float('nan')):.3f}**",
            f"- **PPV / NPV:** {pc.get('ppv', float('nan')):.3f} / {pc.get('npv', float('nan')):.3f}",
            f"- Source: `{pc.get('source', 'unknown')}` on model `{data.get('reference_model', '—')}`",
            "",
            "### Clinical screening context (not used as IMU cutoffs)",
            "",
            f"- **Morse Fall Scale:** {morse.get('typical_cutoff', '')} ({morse.get('citation', '')})",
            f"- **STRATIFY:** {strat.get('typical_cutoff', '')} ({strat.get('citation', '')})",
            "",
            "Full artifact: `clinical_threshold.json`. See `docs/clinical_thresholds.md`.",
            "",
        ]
        return "\n".join(lines)

    def _threshold_comparison_section(self) -> str:
        path = self.metrics_dir / "metrics_threshold_comparison.csv"
        if not path.exists():
            return ""
        return (
            "## Binary threshold comparison (supplemental)\n"
            "\n"
            "**Warning:** `metrics.csv` and Table 2 report accuracy, F1, sensitivity, and "
            "specificity at **`threshold_source=unbiased_train_fold`** (Youden J fit on each "
            "LOSO train fold, applied to held-out subjects). Rows labeled "
            "**`optimistic_eval_set`** in `metrics_threshold_comparison.csv` re-tune Youden J on "
            "the **full pooled OOF set** — metrics such as `accuracy_eval_youden` are "
            "**optimistic** and must **not** be reported as unbiased generalization. "
            "`delta_accuracy_vs_train_fold` quantifies that optimism gap only.\n"
            "\n"
            "Artifact: `metrics_threshold_comparison.csv` (`unbiased_train_fold`, "
            "`fixed_0.5`, `optimistic_eval_set`).\n"
            "\n"
        )

    def _ethics_section(self) -> str:
        ethics_path = Path(__file__).resolve().parents[3] / "docs" / "ethics.md"
        paper_path = Path(__file__).resolve().parents[3] / "docs" / "paper" / "ethics_statement.md"
        if not ethics_path.exists():
            return ""
        return (
            "## Ethics\n\n"
            "This study used a publicly available, de-identified dataset "
            "(DOI: [10.6084/m9.figshare.28806086](https://doi.org/10.6084/m9.figshare.28806086)). "
            "The original data collection was approved by the **Comité de Protection des "
            "Personnes Île-de-France II** (CPP 2014-10-04 RNI), with written informed consent "
            "obtained by the original investigators (Voisard et al., *Scientific Data* 2025, "
            "[10.1038/s41597-025-05959-w](https://doi.org/10.1038/s41597-025-05959-w)). "
            "**No new human data were collected.**\n\n"
            f"Manuscript text: `{paper_path.name}` (repo `docs/paper/`).\n\n"
        )

    def _limitations_section(self) -> str:
        repo_path = Path(__file__).resolve().parents[3] / "docs" / "limitations.md"
        metrics_path = self.metrics_dir / "limitations.md"
        if metrics_path.exists():
            return metrics_path.read_text(encoding="utf-8").strip() + "\n\n"
        if repo_path.exists():
            return repo_path.read_text(encoding="utf-8").strip() + "\n\n"
        from src.evaluation.research_disclaimers import (
            RESEARCH_PROTOTYPE_DISCLAIMER,
            LIMITATIONS_BULLETS,
        )

        lines = ["## Limitations", "", f"**{RESEARCH_PROTOTYPE_DISCLAIMER}**", ""]
        for bullet in LIMITATIONS_BULLETS:
            lines.append(f"- {bullet}")
        lines.append("")
        return "\n".join(lines)

    def _sensor_ablation_section(self) -> str:
        sa_path = self.metrics_dir / "sensor_ablation.csv"
        if not sa_path.exists():
            return ""
        df = pd.read_csv(sa_path)
        lines = [
            "## Sensor Position Ablation",
            "",
            "Leave-one-subject-out macro-OVR AUC for each subset of the four IMU "
            "positions (head, lower back, left foot, right foot). Same validation "
            "protocol as the primary evaluator and feature ablation. Identifies the "
            "minimum sensor configuration for acceptable screening performance.",
            "",
            "| Sensor Subset | # Sensors | # Features | AUC (mean) | AUC (std) |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in df.itertuples(index=False):
            lines.append(
                f"| {row.sensor_subset} | {row.n_sensors} | {row.n_features} | "
                f"{row.auc_mean:.4f} | {row.auc_std:.4f} |"
            )
        best_overall = df.iloc[0]
        single_sensor = df[df["n_sensors"] == 1]
        best_single = (
            single_sensor.sort_values("auc_mean", ascending=False).iloc[0]
            if not single_sensor.empty
            else None
        )
        lines.append("")
        lines.append(
            f"**Best overall subset:** {best_overall.sensor_subset} "
            f"(AUC {best_overall.auc_mean:.4f})."
        )
        if best_single is not None:
            lines.append(
                f"**Best single-sensor:** {best_single.sensor_subset} "
                f"(AUC {best_single.auc_mean:.4f})."
            )
        return "\n".join(lines)

    def _cross_cohort_section(self) -> str:
        cc_path = self.metrics_dir / "cross_cohort_transfer.csv"
        if not cc_path.exists():
            return ""
        df = pd.read_csv(cc_path)
        lines = [
            "## Cross-Cohort Transfer (Leave-One-Cohort-Out)",
            "",
            "Train on all subjects from N-1 cohorts, test on the held-out cohort. "
            "Answers: 'Can a model trained without any PD patients still detect PD?'",
            "",
            "| Held-Out Cohort | N (test) | AUC | Mean True-Class Prob. | Accuracy | F1 (macro) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in df.itertuples(index=False):
            auc_str = f"{row.auc:.4f}" if pd.notna(row.auc) else "N/A"
            truep_str = (
                f"{row.mean_true_class_proba:.4f}"
                if hasattr(row, "mean_true_class_proba") and pd.notna(row.mean_true_class_proba)
                else "N/A"
            )
            lines.append(
                f"| {row.test_cohort} | {row.n_test} | "
                f"{auc_str} | {truep_str} | {row.accuracy:.4f} | {row.f1_macro:.4f} |"
            )
        if "auc_status" in df.columns and (df["auc_status"] == "undefined_single_class_test").any():
            lines.extend([
                "",
                "AUC is **undefined** for single-class held-out cohorts (all rows in this dataset),",
                "so `N/A` is expected. Mean true-class probability is reported as the transfer-confidence fallback.",
            ])
        lines.append("")
        pair_path = self.metrics_dir / "cross_cohort_pairwise.csv"
        if pair_path.exists():
            lines.append(
                "See `cross_cohort_pairwise.csv` for the full 8x8 train-on-A / test-on-B "
                "matrix (macro-F1, macro OvR AUC, and accuracy). "
                "The primary heatmap `cross_cohort_pairwise.{pdf,png}` uses **macro-F1** "
                "(preferred under class imbalance); "
                "`cross_cohort_pairwise_auc.{pdf,png}` is supplemental."
            )
        return "\n".join(lines)

    def _leakage_comparison_section(self) -> str:
        lc_path = self.metrics_dir / "leakage_comparison.csv"
        if not lc_path.exists():
            return "_Not available — run the evaluate stage to generate._"

        df = pd.read_csv(lc_path)
        lines = [
            "Compares LOSO (grouped, no subject leakage) against standard "
            "StratifiedKFold (ungrouped, permits subject leakage) to quantify "
            "the optimistic bias introduced when the same participant appears "
            "in both train and test sets.",
            "",
            "| Model | AUC (Grouped LOSO) | AUC (Ungrouped KFold) | Inflation | Inflation % |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in df.itertuples(index=False):
            lines.append(
                f"| {row.model} | {row.auc_grouped_loso:.4f} | "
                f"{row.auc_ungrouped_kfold:.4f} | "
                f"{row.auc_inflation:+.4f} | "
                f"{row.inflation_pct:+.1f}% |"
            )
        mean_infl = df["inflation_pct"].mean()
        lines.append("")
        lines.append(
            f"**Mean AUC inflation from subject leakage: {mean_infl:+.1f}%**"
        )
        return "\n".join(lines)

    def _ensemble_comparison_section(self) -> str:
        comp_path = self.metrics_dir / "ensemble_comparison.csv"
        if not comp_path.exists():
            return ""

        dfc = pd.read_csv(comp_path)
        lines = [
            "## Ensemble method comparison (nested LOSO)",
            "",
            "Base learners: top-K models by grouped CV AUC. **Soft voting** averages "
            "positive-class probabilities; **stacking** fits a logistic regression "
            "meta-learner on out-of-fold base probabilities (inner StratifiedGroupKFold).",
            "",
            "| Method | AUC | AUC 95% CI | F1 |",
            "|---|---:|---|---:|",
        ]
        for row in dfc.itertuples(index=False):
            ci = (
                f"[{float(row.auc_ci_low):.3f}, {float(row.auc_ci_high):.3f}]"
                if pd.notna(getattr(row, "auc_ci_low", float("nan")))
                else "—"
            )
            lines.append(
                f"| {row.ensemble_method} | {float(row.auc):.4f} | {ci} | {float(row.f1):.4f} |"
            )

        pair_path = self.metrics_dir / "ensemble_pairwise_pvalues.csv"
        if pair_path.exists():
            pair = pd.read_csv(pair_path)
            if not pair.empty:
                r = pair.iloc[0]
                lines.extend([
                    "",
                    f"Paired DeLong (soft voting vs stacking): p = {r.get('p_delong_fmt', r.get('p_delong', '—'))}.",
                    "See `ensemble_pairwise_pvalues.csv`.",
                ])
        lines.append("")
        return "\n".join(lines)

    def _deep_learning_comparison_section(self) -> str:
        dl_path = self.metrics_dir / "deep_learning_metrics.csv"
        if not dl_path.exists():
            return ""

        dl = pd.read_csv(dl_path)
        if dl.empty:
            return ""
        dl["auc"] = pd.to_numeric(dl.get("auc"), errors="coerce")
        dl["macro_f1"] = pd.to_numeric(dl.get("macro_f1"), errors="coerce")
        dl["accuracy"] = pd.to_numeric(dl.get("accuracy"), errors="coerce")
        has_auc = dl["auc"].notna().any()
        if has_auc:
            dl = dl.sort_values(["auc", "macro_f1", "accuracy"], ascending=False)
        else:
            dl = dl.sort_values(["macro_f1", "accuracy"], ascending=False)
        lines = [
            "## Deep learning comparison (LOSO, raw IMU windows)",
            "",
            "Architectures trained end-to-end on windowed raw sensor signals "
            "(4 IMUs × 13 channels, 256-sample windows at 100 Hz). "
            "Each model evaluated under the same Leave-One-Subject-Out protocol "
            "as the classical ML models (N participants).",
            "",
            "| Architecture | AUC | 95% CI | Macro F1 | Accuracy |",
            "|---|---:|---|---:|---:|",
        ]
        for row in dl.itertuples(index=False):
            name = str(row.model).replace("dl_", "").replace("_", " ").title()
            auc_cell = f"{float(row.auc):.4f}" if pd.notna(row.auc) else "—"
            ci = (
                f"[{float(row.auc_ci_low):.3f}, {float(row.auc_ci_high):.3f}]"
                if pd.notna(getattr(row, "auc_ci_low", float("nan")))
                else "—"
            )
            lines.append(
                f"| {name} | {auc_cell} | {ci} | "
                f"{float(row.macro_f1):.4f} | {float(row.accuracy):.4f} |"
            )

        metrics_path = self.metrics_dir / "metrics.csv"
        if metrics_path.exists():
            mdf = pd.read_csv(metrics_path)
            classical = mdf[~mdf["model"].str.startswith("dl_")]
            if not classical.empty:
                best_cl = classical.loc[classical["auc"].idxmax()]
                lines.extend([
                    "",
                    f"Best classical ML: **{best_cl['model']}** (AUC {float(best_cl['auc']):.4f})",
                ])
                best_dl = dl.iloc[0]
                if has_auc:
                    lines.append(
                        f"Best deep learning: **{best_dl['model']}** "
                        f"(AUC {float(best_dl['auc']):.4f})"
                    )
                else:
                    lines.append(
                        f"Best deep learning by macro-F1: **{best_dl['model']}** "
                        f"(macro-F1 {float(best_dl['macro_f1']):.4f})."
                    )
                    lines.append(
                        "Deep-model AUC is unavailable in this export because per-participant "
                        "OOF probability vectors were not saved."
                    )

        lines.append("")
        return "\n".join(lines)
