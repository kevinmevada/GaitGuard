"""
DAPHNET → Voisard sensor mapping (design decision, not preprocessing).

Only defensible pairing:
  DAPHNET trunk accelerometer → Voisard lower_back (lumbar; same gait phenomenon)

Explicitly **not** mapped:
  - ankle → left_foot / right_foot (lateral malleolus ≠ dorsal foot; corrupts zero-shot eval)
  - thigh → (no Voisard equivalent; dropped)

Cross-dataset evaluation uses LB-only input against a model trained on four Voisard sites.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Voisard pipeline sensor key
VOISARD_LOWER_BACK = "lower_back"

# Raw DAPHNET sensor keys (present in flat files; not all are mapped forward)
DAPHNET_TRUNK = "trunk"
DAPHNET_ANKLE = "ankle"
DAPHNET_THIGH = "thigh"

DROPPED_DAPHNET_SENSORS = (DAPHNET_ANKLE, DAPHNET_THIGH)

SENSOR_MAPPING_MANIFEST: dict[str, Any] = {
    "design_decision": (
        "Sensor mapping is a design decision, not a preprocessing step. "
        "DAPHNET trunk → Voisard lower_back is the only anatomically defensible pairing. "
        "This yields a single-sensor zero-shot evaluation — a harder claim than "
        "same-configuration cross-dataset transfer."
    ),
    "mapped": {DAPHNET_TRUNK: VOISARD_LOWER_BACK},
    "dropped": {
        DAPHNET_ANKLE: "lateral malleolus ≠ dorsal foot; different heel-strike dynamics",
        DAPHNET_THIGH: "no Voisard anatomical equivalent",
    },
    "eval_protocol": {
        "train_sensors": ["head", "lower_back", "left_foot", "right_foot"],
        "zero_shot_eval_sensors": [VOISARD_LOWER_BACK],
        "claim": (
            "Multi-sensor Voisard training with LB-only DAPHNET zero-shot eval "
            "(AUC > 0.77 threshold for cross-dataset transfer claim)."
        ),
    },
}


def map_daphnet_signals_to_voisard(
    signals: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Map resampled DAPHNET trunk acc → Voisard ``lower_back``; drop ankle and thigh.
    """
    trunk = signals.get(DAPHNET_TRUNK)
    if trunk is None or trunk.empty:
        return {}
    return {VOISARD_LOWER_BACK: trunk.reset_index(drop=True).copy()}


def daphnet_trial_metadata_extras() -> dict[str, Any]:
    """Metadata columns for DAPHNET trials in ``trial_metadata.csv``."""
    return {
        "source_dataset": "daphnet",
        "sensor_mapping": f"{DAPHNET_TRUNK}_to_{VOISARD_LOWER_BACK}",
        "eval_sensors": VOISARD_LOWER_BACK,
        "dropped_sensors": ",".join(DROPPED_DAPHNET_SENSORS),
    }
