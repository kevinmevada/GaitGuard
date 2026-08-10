"""Preflight: Nested FS cache reuse + ResourceScheduler worker count."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml

from src.core.resources import parallel_n_jobs, reset_resource_scheduler
from src.features.feature_matrix import get_numeric_feature_columns
from src.features.nested_fs_cache import count_cached_folds, nested_fs_cache_dir


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/pipeline_config.yaml").read_text(encoding="utf-8"))
    for key, value in list(cfg.get("paths", {}).items()):
        path = Path(value)
        if not path.is_absolute():
            cfg["paths"][key] = str((ROOT / path).resolve())

    df = pd.read_parquet(Path(cfg["paths"]["features"]) / "patient_features.parquet")
    feat_cols = get_numeric_feature_columns(df)
    n_groups = int(df["participant_id"].astype(str).nunique())
    cache = nested_fs_cache_dir(
        cfg, feat_cols, n_samples=len(df), n_groups=n_groups
    )
    reset_resource_scheduler()
    workers = parallel_n_jobs()
    print(f"cache_dir={cache}")
    print(f"cached_folds={count_cached_folds(cache)}")
    print(f"nested_fs_n_jobs_config={cfg.get('feature_selection', {}).get('nested_fs_n_jobs')}")
    print(f"scheduler_workers={workers}")
    if count_cached_folds(cache) < 1:
        raise SystemExit("ERROR: expected cached folds to resume")
    if workers < 10 or workers > 18:
        raise SystemExit(f"ERROR: unexpected worker count {workers}")
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
