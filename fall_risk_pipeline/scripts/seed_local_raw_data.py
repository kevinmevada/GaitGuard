#!/usr/bin/env python3
"""Generate minimal Voisard raw trials for local HPC worker simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


def _write_voisard_trial(
    trial_dir: Path,
    *,
    trial_id: str,
    subject: str,
    pathology_key: str = "HS",
    cohort: str = "healthy",
    rows_per_sensor: int = 150,
) -> dict:
    """~1.5 s @ 100 Hz (packet counter step 1); enough for local min_trial_length_s."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    meta = {"subject": subject, "pathologyKey": pathology_key}
    (trial_dir / f"{trial_id}_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    for code in ["HE", "LB", "LF", "RF"]:
        lines = ["PacketCounter\tAcc_X\tAcc_Y\tAcc_Z\tGyr_X\tGyr_Y\tGyr_Z\tMag_X\tMag_Y\tMag_Z"]
        for i in range(rows_per_sensor):
            pkt = 100 + i
            lines.append(f"{pkt}\t1.0\t2.0\t3.0\t4.0\t5.0\t6.0\t7.0\t8.0\t9.0")
        (trial_dir / f"{trial_id}_raw_data_{code}.txt").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    rel = f"voisard/{cohort}/{pathology_key}/{subject}/{trial_id}"
    return {
        "dataset": "voisard",
        "subject": subject,
        "cohort": cohort.capitalize() if cohort == "healthy" else pathology_key,
        "trial": trial_id,
        "source_path": rel,
        "sensors": "HE LB LF RF",
        "complete": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5, help="Number of synthetic trials")
    parser.add_argument("--rows", type=int, default=150, help="Rows per sensor file")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PIPELINE_ROOT / "data" / "raw_local",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PIPELINE_ROOT / "data" / "processed",
    )
    args = parser.parse_args()

    raw = args.raw_root
    processed = args.processed_root
    processed.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i in range(1, args.trials + 1):
        subject = f"HS_{i}"
        trial_id = f"HS_{i}_{i}"
        rel = f"voisard/healthy/HS/{subject}/{trial_id}"
        trial_dir = raw / rel
        rows.append(
            _write_voisard_trial(
                trial_dir,
                trial_id=trial_id,
                subject=subject,
                rows_per_sensor=args.rows,
            )
        )

    inv_path = processed / "dataset_inventory.csv"
    pd.DataFrame(rows).to_csv(inv_path, index=False)
    print(f"Wrote {len(rows)} trials under {raw}")
    print(f"Inventory: {inv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
