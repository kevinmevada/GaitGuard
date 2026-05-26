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

from src.utils.reproducibility import get_pipeline_seed, set_global_seed
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

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


def run_stage(stage: str, config: dict, *, fast_eval: bool = False):
    t0 = time.time()

    try:
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
                logger.info("Deep learning disabled in config; skipping train_deep.")
        elif stage == "evaluate":
            from src.evaluation.evaluator import Evaluator

            Evaluator(config, fast=fast_eval).run()
        elif stage == "ablation":
            from src.evaluation.feature_ablation import run_feature_ablation
            run_feature_ablation(config)
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
    parser.add_argument(
        "--fast-eval",
        action="store_true",
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
            run_stage(stage, config, fast_eval=args.fast_eval)
            progress.advance(task_id)
            next_stage = "Pipeline complete" if progress.tasks[0].completed >= len(stages) else "Waiting for next stage..."
            progress.update(task_id, description=next_stage)

    console.print(Panel.fit("[bold green]Pipeline Complete[/bold green]"))


if __name__ == "__main__":
    main()
