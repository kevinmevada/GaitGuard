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
\caption{Fall Risk Prediction Performance (Subject-Grouped Validation)}
\label{tab:results}
\centering
\begin{tabular}{lcccc}
\hline\hline
\textbf{Model} & \textbf{AUC} & \textbf{Acc.} & \textbf{F1} & \textbf{Sens.} \\
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

        for col in ["auc", "accuracy", "f1", "sensitivity", "participants"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df = df.sort_values("auc", ascending=False)
        self._generate_latex_table(df)
        self._generate_markdown_report(df)

    def _generate_latex_table(self, df: pd.DataFrame):
        rows = []
        best_model = df.iloc[0]["model"]

        for row in df.itertuples(index=False):
            name = str(row.model).replace("_", " ").title()
            if row.model == best_model:
                name = r"\textbf{" + name + "}"

            rows.append(
                f"{name} & "
                f"{float(row.auc):.3f} & "
                f"{float(row.accuracy):.3f} & "
                f"{float(row.f1):.3f} & "
                f"{float(row.sensitivity):.3f} \\\\"
            )

        table = LATEX_TEMPLATE.replace("ROWS", "\n".join(rows))
        path = self.metrics_dir / "ieee_table.tex"
        path.write_text(table, encoding="utf-8")
        logger.info(f"LaTeX table saved -> {path}")

    def _generate_markdown_report(self, df: pd.DataFrame):
        best = df.iloc[0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        label_mode = self.config.get("dataset", {}).get("label_mode", "unknown")
        validation_strategy = str(best.get("validation_strategy", "unknown"))
        participants = int(float(best.get("participants", 0)))

        model = str(best.get("model", "unknown"))
        auc = float(best.get("auc", 0))
        acc = float(best.get("accuracy", 0))
        f1 = float(best.get("f1", 0))
        sens = float(best.get("sensitivity", 0))

        report = f"""# Fall Risk Prediction Pipeline - Results Report
Generated: {timestamp}

## Dataset
- Participants: {participants}
- Sensors: 4 IMUs
- Label mode: {label_mode}

## Validation
- Strategy: {validation_strategy}
- Note: Reported metrics are intended for subject-grouped evaluation output, not in-sample prediction export.

## Model Performance

| Model | AUC | Accuracy | F1 | Sensitivity |
|---|---|---|---|---|
"""

        for row in df.itertuples(index=False):
            name = str(row.model)
            mark = " *" if name == model else ""
            report += (
                f"| {name}{mark} | "
                f"{float(row.auc):.3f} | "
                f"{float(row.accuracy):.3f} | "
                f"{float(row.f1):.3f} | "
                f"{float(row.sensitivity):.3f} |\n"
            )

        report += f"""

## Best Model
**{model}**

- AUC: **{auc:.4f}**
- Accuracy: **{acc:.4f}**
- F1 Score: **{f1:.4f}**
- Sensitivity: **{sens:.4f}**

## Outputs
- metrics.csv
- predictions.csv
- SHAP plots
- ROC / PR curves

## Reproducibility

python main.py --config configs/pipeline_config.yaml
"""

        path = self.metrics_dir / "pipeline_report.md"
        path.write_text(report, encoding="utf-8")
        logger.info(f"Markdown report saved -> {path}")
