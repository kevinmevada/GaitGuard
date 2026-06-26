"""Laterality / affected-side audit (HOA, ACL, CVA)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_meta(trial_dir: Path, trial_id: str, meta: dict) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / f"{trial_id}_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_laterality_audit_matches_reference_counts(tmp_path: Path):
    raw = tmp_path / "voisard"
    # HOA: 15 right
    for i in range(1, 16):
        _write_meta(
            raw / "ortho" / "HOA" / f"HOA_{i}" / f"HOA_{i}_1",
            f"HOA_{i}_1",
            {"subject": f"HOA_{i}", "pathologyKey": "HOA", "laterality": "right"},
        )
    # CVA: 47 right, 2 left
    for i in range(1, 48):
        _write_meta(
            raw / "neuro" / "CVA" / f"CVA_{i}" / f"CVA_{i}_1",
            f"CVA_{i}_1",
            {"subject": f"CVA_{i}", "pathologyKey": "CVA", "laterality": "right"},
        )
    for i in range(48, 50):
        _write_meta(
            raw / "neuro" / "CVA" / f"CVA_{i}" / f"CVA_{i}_1",
            f"CVA_{i}_1",
            {"subject": f"CVA_{i}", "pathologyKey": "CVA", "laterality": "left"},
        )

    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import audit_laterality as al  # noqa: WPS433

    out = tmp_path / "laterality_audit.csv"
    assert al.collect_subjects(raw).shape[0] == 64
    al.main = lambda: 0  # type: ignore
    # run collect + write manually
    df = al.collect_subjects(raw)
    df.to_csv(out, index=False)
    al.write_log(df, tmp_path / "laterality_audit.log")

    hoa = df[df["pathologyKey"] == "HOA"]
    cva = df[df["pathologyKey"] == "CVA"]
    assert (hoa["affected_side"] == "right").sum() == 15
    assert (hoa["affected_side"] == "left").sum() == 0
    assert (cva["affected_side"] == "right").sum() == 47
    assert (cva["affected_side"] == "left").sum() == 2
