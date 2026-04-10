"""
Pipeline entrypoint.
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
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

STAGES = ["ingest", "preprocess", "eda", "features", "train", "evaluate", "predict", "anomaly", "report"]


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


def run_stage(stage: str, config: dict):
    t0 = time.time()

    try:
        if stage == "ingest":
            from src.ingestion.data_loader import DataLoader
            DataLoader(config).run()
        elif stage == "preprocess":
            from src.preprocessing.signal_processor import SignalProcessor
            SignalProcessor(config).run()
        elif stage == "eda":
            from src.visualization.eda import EDAAnalyzer
            EDAAnalyzer(config).run()
        elif stage == "features":
            from src.features.feature_extractor import FeatureExtractor
            FeatureExtractor(config).run()
        elif stage == "train":
            from src.models.trainer import ModelTrainer
            ModelTrainer(config).run()
        elif stage == "evaluate":
            from src.evaluation.evaluator import Evaluator
            Evaluator(config).run()
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
    except Exception as exc:
        logger.exception(f"{stage} failed: {exc}")
        console.print(f"[red]Stage {stage} failed[/red]")
        sys.exit(1)

    elapsed = time.time() - t0
    console.print(f"[green][OK] {stage} completed in {elapsed:.1f}s[/green]")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--stage", default="all")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config)

    console.print(Panel.fit("[bold]Fall Risk Prediction Pipeline[/bold]", border_style="cyan"))

    stages = STAGES if args.stage == "all" else [args.stage]
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} stages"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Starting pipeline...", total=len(stages))
        for stage in stages:
            progress.update(task_id, description=f"Running stage: {stage}")
            run_stage(stage, config)
            progress.advance(task_id)
            next_stage = "Pipeline complete" if progress.tasks[0].completed >= len(stages) else "Waiting for next stage..."
            progress.update(task_id, description=next_stage)

    console.print(Panel.fit("[bold green]Pipeline Complete[/bold green]"))


if __name__ == "__main__":
    main()
