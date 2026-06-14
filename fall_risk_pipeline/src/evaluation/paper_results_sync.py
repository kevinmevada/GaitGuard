"""
Sync docs/paper/results.md and abstract metrics from live pipeline artifacts (PUB-001).

Run automatically via the ``report`` stage or manually:

    python scripts/regenerate_paper_results.py
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from src.evaluation.primary_endpoint import (
    ENDPOINT_ANOMALY_ENSEMBLE,
    PROTOCOL_ANOMALY_LOSO,
    PROTOCOL_DEPLOY_GLOBAL,
    PROTOCOL_NESTED_RFECV,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_RESULTS_PATH = REPO_ROOT / "docs" / "paper" / "results.md"
PAPER_ABSTRACT_PATH = REPO_ROOT / "docs" / "paper" / "abstract.md"

MISSING_ARTIFACTS_STUB = """# Results

> **Artifact status:** `fall_risk_pipeline/results/metrics/metrics.csv` was not found at generation time.
>
> **Action required:** run the full post-fix pipeline, then regenerate this file:
>
> ```bash
> cd fall_risk_pipeline
> python main.py
> python ../scripts/regenerate_paper_results.py
> ```
>
> Until regeneration completes, **do not cite numerical results** from older manuscript drafts.
> Prior tables mixed pre-fix code with grouped CV ablation protocols that no longer match the codebase.

## Protocol (current code — post ML-014 fixes)

| Analysis | Validation protocol |
|----------|---------------------|
| Primary tabular LOSO | Leave-one-subject-out + nested RFECV per train fold (`feature_selection_protocol: nested_rfecv_per_loso_fold`) |
| Primary deep LOSO | Leave-one-subject-out; per-fold Optuna LR search when `loso_hyperparameter_tuning.enabled` (ML-042) |
| Train `model_comparison_cv.csv` | Nested StratifiedGroupKFold + per-outer-fold RFECV |
| Feature ablation | LOSO (`feature_ablation.py`) |
| Sensor ablation | LOSO (`sensor_ablation.py`) |
| Cross-cohort transfer | LOCO + nested RFECV per train fold |

## Cohort composition (dataset constants)

N = 260 participants, 1,356 trials, eight cohorts, four IMUs — see `docs/paper/methods.md`.
"""


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _fmt_ci(row) -> str:
    low = getattr(row, "auc_ci_low", float("nan"))
    high = getattr(row, "auc_ci_high", float("nan"))
    if pd.notna(low) and pd.notna(high):
        return f"[{float(low):.4f}, {float(high):.4f}]"
    return "N/A"


def _is_dl_model(model_name: str) -> bool:
    return str(model_name).lower().startswith("dl_")


def filter_tabular_nested_loso(df: pd.DataFrame) -> pd.DataFrame:
    """
    ML-047: tabular primary table — nested RFECV LOSO rows only.

    Excludes deep-learning rows and mixed-protocol rows when columns are present.
    """
    if df.empty:
        return df
    out = df[~df["model"].astype(str).map(_is_dl_model)].copy()
    if "evaluation_mode" in out.columns:
        out = out[out["evaluation_mode"].astype(str) != "loso_dl"]
    if "feature_selection_protocol" in out.columns:
        out = out[
            out["feature_selection_protocol"].astype(str) == PROTOCOL_NESTED_RFECV
        ]
    return out


def filter_deploy_schema_loso(df: pd.DataFrame) -> pd.DataFrame:
    """Deploy/API-schema LOSO rows (global selected_features.json)."""
    if df.empty:
        return df
    out = df[~df["model"].astype(str).map(_is_dl_model)].copy()
    if "feature_selection_protocol" in out.columns:
        masked = out[
            out["feature_selection_protocol"].astype(str) == PROTOCOL_DEPLOY_GLOBAL
        ]
        if not masked.empty:
            return masked
    ensemble = out[out["model"].astype(str).str.contains("ensemble", case=False, na=False)]
    return ensemble if not ensemble.empty else out.iloc[0:0]


def _anomaly_primary_section(metrics_dir: Path) -> str:
    path = metrics_dir / "anomaly_metrics.csv"
    if not path.is_file():
        return (
            "## 2. Primary anomaly screening performance (LOSO OOF)\n\n"
            "_`anomaly_metrics.csv` not found — run `python main.py --stage anomaly`._\n"
        )
    df = pd.read_csv(path)
    lines = [
        "## 2. Primary anomaly screening performance (LOSO OOF)",
        "",
        "Healthy-reference one-class ensemble evaluated with leave-one-subject-out "
        f"(`feature_selection_protocol: {PROTOCOL_ANOMALY_LOSO}`). "
        "Pseudo ground truth: non-Healthy trial = positive (screening target).",
        "",
        "| Method | ROC-AUC | PR-AUC | Sensitivity | Specificity |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in df.sort_values("auc", ascending=False, na_position="last").itertuples(index=False):
        auc_pr = getattr(row, "auc_pr", float("nan"))
        auc_pr_str = f"{float(auc_pr):.4f}" if pd.notna(auc_pr) else "N/A"
        lines.append(
            f"| {row.method} | {float(row.auc):.4f} | {auc_pr_str} | "
            f"{float(row.sensitivity):.4f} | {float(row.specificity):.4f} |"
        )
    ens = df[df["method"] == "ensemble"]
    if not ens.empty:
        r = ens.iloc[0]
        lines.extend([
            "",
            f"**Primary endpoint (`{ENDPOINT_ANOMALY_ENSEMBLE}`):** ensemble ROC-AUC "
            f"{float(r['auc']):.4f}.",
            "",
        ])
    return "\n".join(lines)


def _supervised_secondary_section(df: pd.DataFrame) -> str:
    lines = [
        "## 2b. Secondary supervised pathology-tier performance (tabular models)",
        "",
        "Supplementary to primary anomaly screening — nested RFECV LOSO from `metrics.csv`.",
        "",
        "| Model | AUC | 95% CI | Macro-F1 | Accuracy |",
        "|---|---:|---|---:|---:|",
    ]
    tabular = filter_tabular_nested_loso(df)
    if tabular.empty:
        return "\n".join(lines + [
            "",
            "_No nested-RFECV tabular LOSO rows in `metrics.csv` "
            f"(expected `{PROTOCOL_NESTED_RFECV}`)._",
            "",
        ])
    sort_df = tabular.sort_values("auc", ascending=False)
    for row in sort_df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {float(row.auc):.4f} | {_fmt_ci(row)} | "
            f"{float(row.f1):.4f} | {float(row.accuracy):.4f} |"
        )
    best = sort_df.iloc[0]
    lines.extend([
        "",
        f"**Best supervised LOSO macro-OVR AUC:** {best['model']} ({float(best['auc']):.4f}).",
        "",
    ])
    return "\n".join(lines)


def _deep_learning_section(metrics_dir: Path) -> str:
    path = metrics_dir / "deep_learning_metrics.csv"
    if not path.exists():
        return (
            "## 4. Deep learning LOSO benchmark\n\n"
            "_`deep_learning_metrics.csv` not found — run `python main.py --stage train_deep`._\n"
        )
    df = pd.read_csv(path)
    lines = [
        "## 4. Deep learning LOSO benchmark",
        "",
        "Participant-level LOSO; early-stopping val AUC aggregated per participant (ML-016).",
        "",
        "| Deep model | Macro-OVR AUC | Macro-F1 | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in df.itertuples(index=False):
        auc = getattr(row, "auc", float("nan"))
        auc_str = f"{float(auc):.4f}" if pd.notna(auc) else "N/A"
        lines.append(
            f"| {row.model} | {auc_str} | {float(row.f1):.4f} | {float(row.accuracy):.4f} |"
        )
    return "\n".join(lines) + "\n"


def _deploy_loso_gap_section(metrics_dir: Path) -> str:
    gap_path = metrics_dir / "deploy_loso_gap.csv"
    if not gap_path.is_file():
        return (
            "## 3. Deploy-schema vs nested-RFECV LOSO gap\n\n"
            "_(deploy_loso_gap.csv not found — run evaluate stage)_\n"
        )

    gap_df = pd.read_csv(gap_path)
    lines = [
        "## 3. Deploy-schema vs nested-RFECV LOSO gap (ML-032)",
        "",
        "Section 2b reports nested per-fold RFECV LOSO (`metrics.csv`). "
        "API/deploy checkpoints use `selected_features.json`. Deploy-schema LOSO AUCs:",
        "",
        "| Model | Nested RFECV AUC | Deploy schema AUC | Δ (deploy − nested) |",
        "|---|---:|---:|---:|",
    ]
    for row in gap_df.itertuples(index=False):
        lines.append(
            f"| {row.model} | {float(row.loso_auc_nested_rfecv):.4f} | "
            f"{float(row.loso_auc_deploy_schema):.4f} | "
            f"{float(row.delta_auc_deploy_minus_nested):+.4f} |"
        )

    ep_path = metrics_dir / "primary_endpoint.json"
    if ep_path.is_file():
        import json

        reg = json.loads(ep_path.read_text(encoding="utf-8"))
        pe = reg.get("primary_endpoint", "deploy_ensemble")
        lines.extend(
            ["", f"**Pre-registered primary endpoint:** `{pe}` — see `primary_endpoint.json`."]
        )
    return "\n".join(lines) + "\n"


def build_paper_results_md(config: dict, reporter) -> str:
    """Compose manuscript results from metrics artifacts + reporter sections."""
    metrics_dir = Path(config["paths"]["metrics"])
    metrics_path = metrics_dir / "metrics.csv"
    if not metrics_path.exists():
        return MISSING_ARTIFACTS_STUB

    df = pd.read_csv(metrics_path)
    if df.empty:
        return MISSING_ARTIFACTS_STUB

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sha = _git_sha()

    parts = [
        "# Results",
        "",
        f"_Auto-generated {ts} from pipeline artifacts (git `{sha}`). "
        "Do not edit by hand — run `scripts/regenerate_paper_results.py` after each pipeline run._",
        "",
        "## 1. Cohort composition",
        "",
        "The analysis set contains 260 participants and 1,356 walking trials across eight "
        "cohorts with four synchronized IMUs. Demographics: "
        "`fall_risk_pipeline/results/metrics/table1_demographics.md`.",
        "",
    ]

    ep_path = metrics_dir / "primary_endpoint.json"
    primary_is_anomaly = False
    if ep_path.is_file():
        import json

        reg = json.loads(ep_path.read_text(encoding="utf-8"))
        pe = reg.get("primary_endpoint", ENDPOINT_ANOMALY_ENSEMBLE)
        primary_is_anomaly = pe == ENDPOINT_ANOMALY_ENSEMBLE

    if primary_is_anomaly:
        parts.extend([
            _anomaly_primary_section(metrics_dir),
            _supervised_secondary_section(df),
        ])
    else:
        parts.extend([
            _supervised_secondary_section(df).replace(
                "## 2b. Secondary supervised pathology-tier performance (tabular models)",
                "## 2. Primary multiclass screening performance (tabular models)",
                1,
            ).replace(
                "Supplementary to primary anomaly screening — nested RFECV LOSO from `metrics.csv`.",
                "Values regenerated from `fall_risk_pipeline/results/metrics/metrics.csv` "
                "(LOSO + nested per-fold RFECV; filtered by `feature_selection_protocol` — ML-047). "
                "Deploy/API parity AUCs: section 3 / `deploy_loso_gap.csv` (ML-032).",
                1,
            ),
        ])

    parts.append(_deploy_loso_gap_section(metrics_dir))
    parts.extend([
        "## 4. Class-wise behavior",
        "",
        "See per-class columns in `metrics.csv` and `pipeline_report.md`.",
        "",
        _deep_learning_section(metrics_dir),
    ])

    ablation = reporter._ablation_section()
    parts.append(
        ablation.replace("## Feature", "## 7. Feature", 1)
        if ablation
        else "## 7. Feature ablation\n\n_(feature_ablation.md not found)_\n"
    )

    sensor = reporter._sensor_ablation_section()
    parts.append(sensor if sensor else "## 8. Sensor ablation\n\n_(sensor_ablation.csv not found)_\n")

    cross = reporter._cross_cohort_section()
    parts.append(cross if cross else "## 9. Cross-cohort transfer\n\n_(cross_cohort_transfer.csv not found)_\n")

    leakage = reporter._split_protocol_comparison_section()
    if leakage:
        parts.append(leakage.replace("## Split-Protocol Comparison", "## 10. Split-protocol sensitivity", 1))

    parts.extend([
        "",
        "## 11. Prior-work comparison",
        "",
        "See `docs/paper/table2_prior_work.md` (update headline AUC from section 2 after each rerun).",
        "",
    ])
    return "\n".join(p for p in parts if p)


def sync_abstract_metrics(df: pd.DataFrame, metrics_dir: Path | None = None) -> None:
    """Refresh abstract metrics fill-in block from pipeline artifacts (PUB-001)."""
    if not PAPER_ABSTRACT_PATH.exists():
        return

    primary_ep = ENDPOINT_ANOMALY_ENSEMBLE
    anomaly_auc = float("nan")
    anomaly_pr = float("nan")
    if metrics_dir is not None:
        ep_path = metrics_dir / "primary_endpoint.json"
        if ep_path.is_file():
            import json

            reg = json.loads(ep_path.read_text(encoding="utf-8"))
            primary_ep = reg.get("primary_endpoint", ENDPOINT_ANOMALY_ENSEMBLE)
        anomaly_path = metrics_dir / "anomaly_metrics.csv"
        if anomaly_path.is_file():
            adf = pd.read_csv(anomaly_path)
            ens = adf[adf["method"] == "ensemble"]
            if not ens.empty:
                anomaly_auc = float(ens.iloc[0]["auc"])
                anomaly_pr = float(ens.iloc[0].get("auc_pr", float("nan")))

    tabular = filter_tabular_nested_loso(df)
    if tabular.empty and pd.isna(anomaly_auc):
        logger.warning("No tabular or anomaly metrics for abstract sync")
        return

    sort_df = tabular.sort_values("auc", ascending=False) if not tabular.empty else tabular
    best = sort_df.iloc[0] if not sort_df.empty else None
    deploy_rows = filter_deploy_schema_loso(df)
    ens = deploy_rows.iloc[0] if not deploy_rows.empty else best
    deploy_auc = float(ens["auc"]) if ens is not None else float("nan")
    deploy_model = str(ens["model"]) if ens is not None else "N/A"

    if metrics_dir is not None:
        ep_path = metrics_dir / "primary_endpoint.json"
        if ep_path.is_file():
            import json

            reg = json.loads(ep_path.read_text(encoding="utf-8"))
            deploy_ep = reg.get("registered_endpoints", {}).get("deploy_ensemble", {})
            if deploy_ep.get("auc") is not None:
                deploy_auc = float(deploy_ep["auc"])
                deploy_model = str(deploy_ep.get("model", deploy_model))

    best_auc_str = f"{float(best['auc']):.4f} ({best['model']})" if best is not None else "N/A"

    if primary_ep == ENDPOINT_ANOMALY_ENSEMBLE:
        block = f"""## Metrics fill-in

_Auto-updated from pipeline artifacts — do not edit manually._

| Metric | Value |
|--------|-------|
| Primary anomaly ensemble ROC-AUC (LOSO OOF) | {anomaly_auc:.4f} |
| Primary anomaly ensemble PR-AUC | {anomaly_pr:.4f} |
| Secondary deployable ensemble macro OvR AUC | {deploy_auc:.4f} ({deploy_model}) |
| Best supervised single-model LOSO macro OvR AUC | {best_auc_str} |

_Headline: report **primary anomaly ensemble** in abstract; cite supervised metrics as supplementary._

Regenerate after each pipeline run:

```bash
cd fall_risk_pipeline && python main.py
python ../scripts/regenerate_paper_results.py
```
"""
    else:
        block = f"""## Metrics fill-in

_Auto-updated from `metrics.csv` — do not edit manually._

| Metric | Value |
|--------|-------|
| Primary deployable ensemble macro OvR AUC | {deploy_auc:.4f} ({deploy_model}, deploy-schema LOSO) |
| Best single-model LOSO macro OvR AUC | {float(best['auc']):.4f} ({best['model']}, nested RFECV) |
| Ensemble 95% bootstrap CI | [{float(ens.get('auc_ci_low', float('nan'))):.2f}, {float(ens.get('auc_ci_high', float('nan'))):.2f}] |

_Headline model: report **primary deployable ensemble** in abstract; cite **best single-model** separately when comparing to prior RF-centric work (RES-003)._

Regenerate after each pipeline run:

```bash
cd fall_risk_pipeline && python main.py
python ../scripts/regenerate_paper_results.py
```
"""

    text = PAPER_ABSTRACT_PATH.read_text(encoding="utf-8")
    marker = "## Metrics fill-in"
    if marker in text:
        head = text.split(marker, 1)[0].rstrip()
        PAPER_ABSTRACT_PATH.write_text(head + "\n\n" + block + "\n", encoding="utf-8")
    else:
        PAPER_ABSTRACT_PATH.write_text(text.rstrip() + "\n\n" + block + "\n", encoding="utf-8")


def sync_paper_results(config: dict) -> Path:
    """Write docs/paper/results.md (and refresh abstract metrics when possible)."""
    from src.evaluation.reporter import ReportGenerator

    reporter = ReportGenerator(config)
    metrics_path = Path(config["paths"]["metrics"]) / "metrics.csv"

    if not metrics_path.exists():
        PAPER_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PAPER_RESULTS_PATH.write_text(MISSING_ARTIFACTS_STUB, encoding="utf-8")
        logger.warning(
            "PUB-001: metrics.csv missing — wrote stub to {}", PAPER_RESULTS_PATH
        )
        return PAPER_RESULTS_PATH

    df = pd.read_csv(metrics_path)
    metrics_dir = Path(config["paths"]["metrics"])
    content = build_paper_results_md(config, reporter)
    PAPER_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAPER_RESULTS_PATH.write_text(content, encoding="utf-8")
    sync_abstract_metrics(df, metrics_dir=metrics_dir)
    logger.info("PUB-001: synced paper results -> {}", PAPER_RESULTS_PATH)
    return PAPER_RESULTS_PATH
