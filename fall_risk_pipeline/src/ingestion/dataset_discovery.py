"""
Stage 0 — Dataset discovery for multi-source raw layouts (Voisard, DAPHNET).

Scans configured source folders under ``paths.raw_data``, inventories subjects /
trials / cohorts, verifies required sensor files exist, and writes
``dataset_inventory.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from src.ingestion.data_loader import PATHOLOGY_KEY_MAP, SENSOR_FILE_MAPPING
from src.ingestion.daphnet_parser import DAPHNET_FILENAME_RE, DAPHNET_SENSOR_CODES
from src.utils.progress import progress_bar

VOISARD_SOURCE = "voisard"
DAPHNET_SOURCE = "daphnet"

VOISARD_SENSOR_CODES = list(SENSOR_FILE_MAPPING.values())  # HE, LB, LF, RF

INVENTORY_COLUMNS = [
    "dataset",
    "subject",
    "cohort",
    "trial",
    "sensors",
    "required_sensors",
    "missing_sensors",
    "complete",
    "source_path",
]


class DatasetDiscovery:
    def __init__(self, config: dict):
        self.config = config
        self.raw_dir = Path(config["paths"]["raw_data"])
        discovery_cfg = config.get("discovery", {})
        default_out = Path(config["paths"]["processed_data"]) / "dataset_inventory.csv"
        self.out_path = Path(discovery_cfg.get("inventory_path", default_out))
        self.sources = [
            s.lower()
            for s in discovery_cfg.get("sources", [VOISARD_SOURCE, DAPHNET_SOURCE])
        ]

    def run(self) -> pd.DataFrame:
        logger.info(f"Dataset discovery scanning {self.raw_dir}")

        rows: list[dict] = []
        if VOISARD_SOURCE in self.sources:
            voisard_root = self._source_root(VOISARD_SOURCE)
            if voisard_root is None:
                logger.warning(f"No {VOISARD_SOURCE}/ folder under {self.raw_dir}")
            else:
                rows.extend(self._scan_voisard(voisard_root))

        if DAPHNET_SOURCE in self.sources:
            daphnet_root = self._source_root(DAPHNET_SOURCE)
            if daphnet_root is None:
                logger.warning(f"No {DAPHNET_SOURCE}/ folder under {self.raw_dir}")
            else:
                rows.extend(self._scan_daphnet(daphnet_root))

        df = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.out_path, index=False)

        n_complete = int(df["complete"].sum()) if not df.empty else 0
        logger.info(
            f"Dataset inventory → {self.out_path} "
            f"({len(df)} trials, {n_complete} complete)"
        )
        return df

    def _source_root(self, name: str) -> Path | None:
        direct = self.raw_dir / name
        if direct.is_dir():
            return direct
        for child in self.raw_dir.iterdir():
            if child.is_dir() and child.name.lower() == name:
                return child
        return None

    def _scan_voisard(self, root: Path) -> list[dict]:
        rows: list[dict] = []
        meta_paths = sorted(root.rglob("*_meta.json"))
        for meta_path in progress_bar(
            meta_paths, desc="discover voisard", unit="trial"
        ):
            trial_dir = meta_path.parent
            trial_id = trial_dir.name
            meta = self._load_json(meta_path)

            subject = str(meta.get("subject") or meta.get("participant_id") or trial_dir.parent.name)
            cohort = self._voisard_cohort(meta, trial_dir)
            present = self._voisard_present_sensors(trial_dir, trial_id)
            missing = [s for s in VOISARD_SENSOR_CODES if s not in present]

            rows.append(
                {
                    "dataset": VOISARD_SOURCE,
                    "subject": subject,
                    "cohort": cohort,
                    "trial": trial_id,
                    "sensors": " ".join(present),
                    "required_sensors": " ".join(VOISARD_SENSOR_CODES),
                    "missing_sensors": " ".join(missing),
                    "complete": not missing and meta_path.exists(),
                    "source_path": str(trial_dir.relative_to(self.raw_dir)).replace("\\", "/"),
                }
            )
        return rows

    def _scan_daphnet(self, root: Path) -> list[dict]:
        rows: list[dict] = []
        txt_paths = [
            p
            for p in sorted(root.rglob("*.txt"))
            if DAPHNET_FILENAME_RE.match(p.name)
        ]
        for txt_path in progress_bar(txt_paths, desc="discover daphnet", unit="file"):
            match = DAPHNET_FILENAME_RE.match(txt_path.name)
            assert match is not None
            subject = f"S{match.group(1)}"
            trial = txt_path.stem.upper()
            complete = txt_path.is_file() and txt_path.stat().st_size > 0

            rows.append(
                {
                    "dataset": DAPHNET_SOURCE,
                    "subject": subject,
                    "cohort": "PD",
                    "trial": trial,
                    "sensors": " ".join(DAPHNET_SENSOR_CODES),
                    "required_sensors": " ".join(DAPHNET_SENSOR_CODES),
                    "missing_sensors": "" if complete else " ".join(DAPHNET_SENSOR_CODES),
                    "complete": complete,
                    "source_path": str(txt_path.relative_to(self.raw_dir)).replace("\\", "/"),
                }
            )
        return rows

    def _voisard_cohort(self, meta: dict, trial_dir: Path) -> str:
        pkey = meta.get("pathologyKey", "")
        if pkey in PATHOLOGY_KEY_MAP:
            return PATHOLOGY_KEY_MAP[pkey]

        cohort = meta.get("cohort")
        if cohort:
            return str(cohort)

        tokens = {p.lower() for p in trial_dir.parts}
        for key, name in PATHOLOGY_KEY_MAP.items():
            if key.lower() in tokens:
                return name
        if "healthy" in tokens:
            return "Healthy"

        return "Unknown"

    def _voisard_present_sensors(self, trial_dir: Path, trial_id: str) -> list[str]:
        present: list[str] = []
        for code in VOISARD_SENSOR_CODES:
            txt_path = trial_dir / f"{trial_id}_raw_data_{code}.txt"
            csv_key = next(
                (pos for pos, suffix in SENSOR_FILE_MAPPING.items() if suffix == code),
                None,
            )
            csv_path = trial_dir / f"{csv_key}_raw.csv" if csv_key else None
            if txt_path.exists() or (csv_path is not None and csv_path.exists()):
                present.append(code)
        return present

    @staticmethod
    def _load_json(path: Path) -> dict:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(f"Failed to read metadata {path}: {exc}")
            return {}
