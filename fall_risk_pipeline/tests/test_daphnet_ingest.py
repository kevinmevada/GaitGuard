"""DataLoader integration for DAPHNET per-subject ingest."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ingestion.data_loader import DataLoader


def _daphnet_row(timestamp: float, annotation: int, trunk_z: float = 9.0) -> str:
    return f"{timestamp} 1 2 3 4 5 6 7 8 {trunk_z} {annotation}"


def _write_long_daphnet_file(path: Path, *, n_walk_rows: int = 1200) -> None:
    lines = [_daphnet_row(0, 0)]
    for i in range(n_walk_rows):
        # alternate normal / FOG annotations for label coverage
        ann = 1 if i % 50 != 0 else 2
        lines.append(_daphnet_row(16 * (i + 1), ann))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gait_like_daphnet(path: Path, *, n_walk_rows: int = 2000, freq_hz: float = 5.0) -> None:
    """Synthetic trunk-Z sinusoid at ~64 Hz for PSD verification."""
    lines = [_daphnet_row(0, 0)]
    for i in range(n_walk_rows):
        t_ms = (i + 1) * (1000.0 / 64.0)
        t_s = t_ms / 1000.0
        trunk_z = 500.0 * np.sin(2.0 * np.pi * freq_hz * t_s)
        lines.append(_daphnet_row(t_ms, 1, trunk_z=trunk_z))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_dataloader_ingests_concatenated_subject(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    daphnet_dir = raw / "daphnet"
    daphnet_dir.mkdir(parents=True)

    _write_long_daphnet_file(daphnet_dir / "S01R01.txt")
    _write_long_daphnet_file(daphnet_dir / "S01R02.txt")

    config = {
        "paths": {"raw_data": str(raw), "processed_data": str(processed), "metrics": str(tmp_path / "metrics")},
        "preprocessing": {"min_trial_length_s": 10.0},
        "dataset": {"sampling_rate": 100, "label_mode": "multiclass", "high_risk_threshold": 2},
        "ingestion": {
            "daphnet": {
                "enabled": True,
                "drop_annotation_zero": True,
                "psd_verification": {"enabled": False},
            }
        },
    }

    records = DataLoader(config).run()
    daphnet_recs = [r for r in records if r.trial_id.startswith("daphnet_")]
    assert len(daphnet_recs) == 1
    rec = daphnet_recs[0]
    assert rec.participant_id == "S01"
    assert rec.session == "S01R01+S01R02"
    assert set(rec.signals) == {"lower_back"}
    assert rec.sensor_mapping == "trunk_to_lower_back"
    assert rec.eval_sensors == "lower_back"
    assert rec.source_dataset == "daphnet"
    lb = rec.signals["lower_back"]
    assert rec.duration_s >= 10.0
    assert abs(lb["time"].iloc[1] - lb["time"].iloc[0] - 0.01) < 1e-6

    report = tmp_path / "metrics" / "daphnet_ingest_report.csv"
    assert report.is_file()
    fog_labels = processed / "daphnet" / "fog_labels.npz"
    assert fog_labels.is_file()
    data = np.load(fog_labels)
    assert "S01" in data.files
    assert set(data["S01"]) <= {0, 1}


def test_dataloader_resample_psd_artifacts(tmp_path: Path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    figs = tmp_path / "figs"
    daphnet_dir = raw / "daphnet"
    daphnet_dir.mkdir(parents=True)

    _write_gait_like_daphnet(daphnet_dir / "S01R01.txt")
    _write_gait_like_daphnet(daphnet_dir / "S02R01.txt")

    config = {
        "paths": {"raw_data": str(raw), "processed_data": str(processed), "metrics": str(tmp_path / "metrics")},
        "preprocessing": {"min_trial_length_s": 10.0},
        "dataset": {"sampling_rate": 100, "label_mode": "multiclass", "high_risk_threshold": 2},
        "ingestion": {
            "daphnet": {
                "enabled": True,
                "drop_annotation_zero": True,
                "resample_up": 25,
                "resample_down": 16,
                "psd_verification": {
                    "enabled": True,
                    "min_subjects": 2,
                    "figure_dir": str(figs),
                },
            }
        },
    }

    records = DataLoader(config).run()
    assert len([r for r in records if r.trial_id.startswith("daphnet_")]) == 2
    assert (figs / "psd_check_S01.png").is_file()
    assert (figs / "psd_check_S02.png").is_file()
    assert (tmp_path / "metrics" / "daphnet_psd_verification.csv").is_file()
    assert (tmp_path / "metrics" / "daphnet_sensor_mapping.json").is_file()
