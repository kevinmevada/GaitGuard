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

from src.utils.reproducibility import get_pipeline_seed, set_global_seed
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

console = Console()
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

STAGES = [
    "ingest",
    "validate_gait_events",
    "preprocess",
    "eda",
    "features",
    "select_features",
    "train",
    "evaluate",
    "train_deep",
    "ablation",
    "sensor_ablation",
    "cross_cohort",
    "predict",
    "anomaly",
    "report",
]


def load_config(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("Invalid config format")

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


def run_stage(stage: str, config: dict, *, fast_eval: bool = False) -> float:
    t0 = time.time()

    if stage == "ingest":
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
            logger.info("Deep learning disabled in config — skipping train_deep.")
    elif stage == "evaluate":
        from src.evaluation.evaluator import Evaluator
        Evaluator(config, fast=fast_eval).run()
    elif stage == "ablation":
        from src.evaluation.feature_ablation import run_feature_ablation
        run_feature_ablation(config)
    elif stage == "sensor_ablation":
        from src.evaluation.sensor_ablation import run_sensor_ablation
        run_sensor_ablation(config)
    elif stage == "cross_cohort":
        from src.evaluation.cross_cohort_transfer import run_cross_cohort_transfer
        run_cross_cohort_transfer(config)
    elif stage == "predict":
        from src.evaluation.predictions import PredictionGenerator
        PredictionGenerator(config).run()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--stage", default="all")
    parser.add_argument(
        "--fast-eval", "--fast",
        action="store_true",
        dest="fast_eval",
        help="Use per-fold checkpoint refit for evaluate/ablation (faster). "
        "Omit for full nested Optuna LOSO (paper-grade, much slower).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    rep_cfg = config.get("reproducibility") or {}
    seed = get_pipeline_seed(config)
    set_global_seed(seed, deterministic_torch=bool(rep_cfg.get("deterministic_torch", True)))
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
    if os.environ.get("PYTHONHASHSEED") is None:
        logger.warning(
            "PYTHONHASHSEED was not set before process start; export PYTHONHASHSEED={} "
            "for fully stable hash-based ordering across runs.",
            seed,
        )

    console.print(Panel.fit("[bold]Gait Screening Pipeline[/bold]", border_style="cyan"))

    stages = STAGES if args.stage == "all" else [args.stage]
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

            try:
                elapsed = run_stage(stage, config, fast_eval=args.fast_eval)
                stage_timings.append((stage, elapsed, "[green]OK[/green]"))
                console.print(
                    f"  [green][OK][/green] {stage} completed in {_fmt_time(elapsed)}"
                )
            except Exception as exc:
                stage_timings.append((stage, 0.0, "[red]FAILED[/red]"))
                logger.exception(f"{stage} failed: {exc}")
                console.print(f"  [red][FAIL] {stage}: {exc}[/red]")
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
