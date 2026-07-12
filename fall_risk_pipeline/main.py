"""
Pipeline entrypoint.

All stages run linearly in sequence.  A Rich progress bar shows overall
completion percentage, current stage number, and elapsed time.  A summary
table is printed at the end with per-stage timings.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.utils.pipeline_stages import PIPELINE_STAGES, resolve_stages, validate_stage_order
from src.utils.progress import stage_spinner
from src.utils.reproducibility import get_pipeline_seed, set_global_seed
from src.utils.torch_device import resolve_torch_device
from src.utils.config_schema import validate_config_or_raise

console = Console()
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

STAGES = list(PIPELINE_STAGES)


def load_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Invalid config format")

        validate_config_or_raise(config)

        config["_pipeline_meta"] = {
            "config_path": str(Path(path).resolve()),
        }
        return config
    except Exception as exc:
        console.print(f"[red]Config load failed: {exc}[/red]")
        sys.exit(1)


def setup_logging(config: dict):
    log_dir = Path(config.get("paths", {}).get("logs", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(log_dir / "pipeline.log", level="DEBUG", rotation="5 MB")


def run_stage(stage: str, config: dict) -> float:
    t0 = time.time()

    if stage == "discover":
        from src.ingestion.dataset_discovery import DatasetDiscovery
        DatasetDiscovery(config).run()
    elif stage == "validate_raw":
        from src.ingestion.raw_data_validator import RawDataValidator
        RawDataValidator(config).run()
    elif stage == "ingest":
        from src.ingestion.data_loader import DataLoader
        DataLoader(config).run()
    elif stage == "validate_gait_events":
        from src.preprocessing.gait_event_validation import GaitEventValidator
        GaitEventValidator(config).run()
    elif stage == "preprocess":
        from src.preprocessing.signal_processor import SignalProcessor
        SignalProcessor(config).run()
    elif stage == "eda":
        from src.visualization.eda import EDAAnalyzer
        EDAAnalyzer(config).run()
    elif stage == "features":
        from src.features.feature_extractor import FeatureExtractor
        FeatureExtractor(config).run()
    elif stage == "phase3_features":
        from src.features.phase3_deep import run_phase3_feature_extraction
        run_phase3_feature_extraction(config)
    elif stage == "select_features":
        from src.features.feature_selector import FeatureSelector
        FeatureSelector(config).run()
    elif stage == "train":
        from src.models.trainer import ModelTrainer
        ModelTrainer(config).run()
    elif stage == "train_deep":
        dl_cfg = config.get("deep_learning", {})
        if dl_cfg.get("enabled", False):
            from src.models.deep_trainer import DeepLearningPipeline
            DeepLearningPipeline(config).run()
        else:
            with stage_spinner("train_deep (skipped)"):
                logger.info("Deep learning disabled in config — skipping train_deep.")
    elif stage == "evaluate":
        from src.evaluation.evaluator import Evaluator
        Evaluator(config).run()
    elif stage == "ablation":
        from src.evaluation.feature_ablation import run_feature_ablation
        run_feature_ablation(config)
    elif stage == "sensor_ablation":
        from src.evaluation.sensor_ablation import run_sensor_ablation
        from src.evaluation.primary_endpoint import ENDPOINT_BILSTM_AE_ENSEMBLE, resolve_primary_endpoint

        run_sensor_ablation(config)
        sa_cfg = (config.get("sensor_ablation") or {}).get("bilstm_ae") or {}
        if sa_cfg.get("enabled", True) and resolve_primary_endpoint(config) == ENDPOINT_BILSTM_AE_ENSEMBLE:
            from src.evaluation.bilstm_ae_sensor_ablation import run_bilstm_ae_sensor_ablation

            run_bilstm_ae_sensor_ablation(config)
    elif stage == "classical_baselines":
        from src.evaluation.classical_baseline_evaluator import run_classical_baselines

        run_classical_baselines(config)
    elif stage == "dl_baselines":
        from src.evaluation.dl_baseline_evaluator import run_dl_baselines

        run_dl_baselines(config)
    elif stage == "competitor_metrics":
        from src.evaluation.competitor_matrix_aggregator import run_competitor_discriminative_matrix

        run_competitor_discriminative_matrix(config)
    elif stage == "severity_regression":
        from src.evaluation.severity_regression_evaluator import run_severity_regression_evaluation

        run_severity_regression_evaluation(config)
    elif stage == "statistical_benchmark":
        from src.evaluation.statistical_benchmark_evaluator import run_statistical_benchmark

        run_statistical_benchmark(config)
    elif stage == "compute_overhead":
        from src.evaluation.compute_overhead_evaluator import run_compute_overhead_benchmark

        run_compute_overhead_benchmark(config)
    elif stage == "novelty_table":
        from src.evaluation.novelty_table_evaluator import run_novelty_comparison_table

        run_novelty_comparison_table(config)
    elif stage == "per_cohort_loso":
        from src.evaluation.per_cohort_loso_evaluator import run_per_cohort_loso_results

        run_per_cohort_loso_results(config)
    elif stage == "fall_risk_spearman":
        from src.evaluation.fall_risk_spearman_evaluator import run_fall_risk_spearman_correlation

        run_fall_risk_spearman_correlation(config)
    elif stage == "cross_cohort":
        from src.evaluation.cross_cohort_transfer import run_cross_cohort_transfer
        run_cross_cohort_transfer(config)
    elif stage == "predict":
        from src.evaluation.predictions import PredictionGenerator
        PredictionGenerator(config).run()
    elif stage == "fit_uncertainty":
        from src.evaluation.fit_uncertainty import run_fit_uncertainty
        run_fit_uncertainty(config)
    elif stage == "anomaly":
        from src.models.anomaly_detector import detect_anomalies
        detect_anomalies(config)
    elif stage == "report":
        from src.evaluation.reporter import ReportGenerator
        ReportGenerator(config).run()
    else:
        raise ValueError(f"Unknown stage: {stage}")

    return time.time() - t0


def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s" if h else f"{m}m {s}s"


def main():
    parser = argparse.ArgumentParser(
        description="Gait screening pipeline — run all stages with `python main.py` or `--stage all`.",
    )
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument(
        "--stage",
        default="all",
        help="Stage name, comma-separated list, or 'all' (default: full pipeline)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    try:
        stages = resolve_stages(args.stage)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)

    for warning in validate_stage_order(stages):
        logger.warning(warning)
        console.print(f"  [yellow][WARN][/yellow] {warning}")

    rep_cfg = config.get("reproducibility") or {}
    seed = get_pipeline_seed(config)
    hash_seed_before_start = os.environ.get("PYTHONHASHSEED")
    set_global_seed(seed, deterministic_torch=bool(rep_cfg.get("deterministic_torch", True)))
    resolve_torch_device(config)
    logger.info(
        "Global RNG seed={} (reproducibility.seed / models.evaluation.random_state); "
        "see docs/reproducibility.md",
        seed,
    )
    eval_rs = (config.get("models") or {}).get("evaluation", {}).get("random_state")
    if eval_rs is not None and int(eval_rs) != seed:
        logger.warning(
            "reproducibility.seed ({}) differs from models.evaluation.random_state ({}); "
            "stages using evaluation.random_state will not match global seed.",
            seed,
            eval_rs,
        )
    if hash_seed_before_start in (None, "random"):
        logger.warning(
            "PYTHONHASHSEED was {!r} at process start; hash iteration order may differ from "
            "a shell launched with PYTHONHASHSEED={}. See docs/reproducibility.md.",
            hash_seed_before_start,
            seed,
        )

    console.print(Panel.fit("[bold]Gait Screening Pipeline[/bold]", border_style="cyan"))
    if args.stage == "all":
        console.print(
            f"[dim]Running all {len(stages)} stages end-to-end — no manual stage chaining required.[/dim]"
        )

    n_stages = len(stages)
    stage_timings: list[tuple[str, float, str]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Pipeline", total=n_stages)

        for i, stage in enumerate(stages, 1):
            pct = int((i - 1) / n_stages * 100)
            progress.update(
                task_id,
                description=f"[{pct}%] Stage {i}/{n_stages}: [bold]{stage}[/bold]",
            )

            progress.console.print(
                f"\n[bold cyan]▶ Stage {i}/{n_stages}: {stage}[/bold cyan]"
            )
            try:
                elapsed = run_stage(stage, config)
                stage_timings.append((stage, elapsed, "[green]OK[/green]"))
                progress.console.print(
                    f"  [green][OK][/green] {stage} completed in {_fmt_time(elapsed)}"
                )
            except Exception as exc:
                stage_timings.append((stage, 0.0, "[red]FAILED[/red]"))
                logger.exception(f"{stage} failed: {exc}")
                progress.console.print(f"  [red][FAIL] {stage}: {exc}[/red]")
                sys.exit(1)

            progress.advance(task_id)

        progress.update(task_id, description="[100%] Pipeline complete")

    total_time = sum(t for _, t, _ in stage_timings)

    table = Table(title="Pipeline Summary", border_style="cyan")
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Stage", style="bold")
    table.add_column("Time", justify="right")
    table.add_column("% of Total", justify="right")
    table.add_column("Status", justify="center")

    for idx, (name, elapsed, status) in enumerate(stage_timings, 1):
        pct_of_total = f"{elapsed / total_time * 100:.1f}%" if total_time > 0 else "-"
        table.add_row(str(idx), name, _fmt_time(elapsed), pct_of_total, status)

    table.add_section()
    table.add_row("", "[bold]Total[/bold]", f"[bold]{_fmt_time(total_time)}[/bold]", "100%", "")

    console.print()
    console.print(table)
    console.print(Panel.fit("[bold green]Pipeline Complete[/bold green]"))


if __name__ == "__main__":
    main()
