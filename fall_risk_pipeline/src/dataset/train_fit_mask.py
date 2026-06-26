"""Train-fold masks for scaler / threshold fitting (v13 leakage fix)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.dataset.subject_split import HEALTHY_COHORT, build_holdout_from_participants


def healthy_train_fit_mask(metadata: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    """
    Boolean mask: Healthy trials belonging to subject-split **train** fold only.

    Use for fitting StandardScaler, one-class models, AE, ROCKET kernels, and
    anomaly / reconstruction thresholds. Never fit on val, test, or all Healthy.
    """
    participants = metadata[["participant_id", "cohort"]].drop_duplicates("participant_id")
    split = build_holdout_from_participants(participants, config)
    train_subjects = set(split.train_ids)
    cohort = metadata["cohort"].astype(str)
    pid = metadata["participant_id"].astype(str)
    return ((cohort == HEALTHY_COHORT) & pid.isin(train_subjects)).to_numpy()
