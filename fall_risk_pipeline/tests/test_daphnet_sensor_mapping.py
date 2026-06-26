"""DAPHNET → Voisard sensor mapping (trunk → lower_back only)."""

from __future__ import annotations

import pandas as pd

from src.ingestion.daphnet_sensor_mapping import (
    DROPPED_DAPHNET_SENSORS,
    SENSOR_MAPPING_MANIFEST,
    daphnet_trial_metadata_extras,
    map_daphnet_signals_to_voisard,
)


def test_maps_trunk_to_lower_back_only():
    trunk = pd.DataFrame(
        {"time": [0.0, 0.01], "acc_x": [1.0, 2.0], "acc_y": [3.0, 4.0], "acc_z": [5.0, 6.0]}
    )
    ankle = trunk.copy()
    out = map_daphnet_signals_to_voisard({"trunk": trunk, "ankle": ankle, "thigh": trunk})
    assert set(out) == {"lower_back"}
    assert out["lower_back"]["acc_z"].iloc[1] == 6.0


def test_dropped_sensors_documented():
    assert "ankle" in DROPPED_DAPHNET_SENSORS
    assert "thigh" in DROPPED_DAPHNET_SENSORS
    assert "design decision" in SENSOR_MAPPING_MANIFEST["design_decision"].lower()


def test_metadata_extras():
    meta = daphnet_trial_metadata_extras()
    assert meta["sensor_mapping"] == "trunk_to_lower_back"
    assert meta["eval_sensors"] == "lower_back"
    assert meta["dropped_sensors"] == "ankle,thigh"
