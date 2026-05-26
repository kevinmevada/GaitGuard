"""
Global RNG seeding for the fall-risk pipeline.

Call ``set_global_seed`` once at pipeline entry (``main.py``) before any stage runs.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np

_DEFAULT_SEED = 42


def get_pipeline_seed(config: dict[str, Any]) -> int:
    """
    Master pipeline seed.

    Priority: ``reproducibility.seed`` → ``models.evaluation.random_state`` → 42.
    """
    rep = config.get("reproducibility") or {}
    if rep.get("seed") is not None:
        return int(rep["seed"])
    try:
        return int(config["models"]["evaluation"]["random_state"])
    except (KeyError, TypeError, ValueError):
        return _DEFAULT_SEED


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> int:
    """
    Seed Python, NumPy, and PyTorch RNGs used across pipeline stages.

    Returns the seed applied (for logging).
    """
    seed = int(seed)

    # Best-effort for hash-based ordering in this process; for full effect set
    # PYTHONHASHSEED before launching Python (see docs/reproducibility.md).
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
