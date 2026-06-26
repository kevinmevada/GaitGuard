#!/usr/bin/env python3
"""
Audit affected-limb laterality from Voisard ``meta.json`` (HOA, ACL, CVA).

Primary field: ``laterality`` (matches published cohort bookkeeping:
HOA 15 right / 0 left; CVA 47 right / 2 left / 0 bilateral).

``clinicalDeficitSide`` is reported separately when it disagrees.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "fall_risk_pipeline" / "data" / "raw" / "voisard"
DEFAULT_OUT = REPO_ROOT / "fall_risk_pipeline" / "results" / "metrics" / "laterality_audit.csv"
DEFAULT_LOG = REPO_ROOT / "fall_risk_pipeline" / "results" / "metrics" / "laterality_audit.log"

TARGET_COHORTS = ("HOA", "ACL", "CVA")

# Participant-level reference counts (laterality field, right/left/bilateral)
REFERENCE_LATERALITY = {
    "HOA": {"right": 15, "left": 0, "bilateral": 0},
    "CVA": {"right": 47, "left": 2, "bilateral": 0},
}


def normalize_side(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "missing"
    text = str(value).strip().lower()
    if text in {"left", "l"}:
        return "left"
    if text in {"right", "r"}:
        return "right"
    if text in {"bilateral", "both", "bil"}:
        return "bilateral"
    return text or "missing"


def collect_subjects(root: Path) -> pd.DataFrame:
    records: dict[tuple[str, str], dict] = {}
    for meta_path in sorted(root.rglob("*_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pkey = str(meta.get("pathologyKey", "")).upper()
        if pkey not in TARGET_COHORTS:
            continue
        subject = str(meta.get("subject") or meta.get("participant_id") or meta_path.parent.parent.name)
        key = (pkey, subject)
        if key not in records:
            records[key] = {
                "pathologyKey": pkey,
                "cohort": {"HOA": "HipOA", "ACL": "ACL", "CVA": "CVA"}[pkey],
                "subject": subject,
                "affected_side": normalize_side(meta.get("laterality")),
                "clinical_deficit_side": normalize_side(meta.get("clinicalDeficitSide")),
                "fields_agree": normalize_side(meta.get("laterality"))
                == normalize_side(meta.get("clinicalDeficitSide")),
                "n_trials": 0,
            }
        records[key]["n_trials"] += 1
    return pd.DataFrame(records.values())


def count_sides(df: pd.DataFrame, column: str = "affected_side") -> dict[str, int]:
    c = Counter(df[column])
    return {
        "left": int(c.get("left", 0)),
        "right": int(c.get("right", 0)),
        "bilateral": int(c.get("bilateral", 0)),
        "missing": int(c.get("missing", 0)),
    }


def write_log(df: pd.DataFrame, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# laterality_audit.log — affected-side distribution (Section 4 / limitations)",
        "# primary_field: meta.json → laterality (affected / protocol side)",
        f"# subjects_scanned: {len(df)}",
        "",
    ]
    for pkey in TARGET_COHORTS:
        sub = df[df["pathologyKey"] == pkey]
        lat = count_sides(sub, "affected_side")
        clin = count_sides(sub, "clinical_deficit_side")
        lines.append(f"## {pkey} (n_participants={len(sub)})")
        lines.append(
            f"  laterality (affected_side): right={lat['right']} left={lat['left']} "
            f"bilateral={lat['bilateral']} missing={lat['missing']}"
        )
        lines.append(
            f"  clinicalDeficitSide: right={clin['right']} left={clin['left']} "
            f"bilateral={clin['bilateral']} missing={clin['missing']}"
        )
        n_mismatch = int((~sub["fields_agree"]).sum())
        lines.append(f"  laterality_vs_clinicalDeficitSide_mismatch: {n_mismatch}")
        ref = REFERENCE_LATERALITY.get(pkey)
        if ref:
            ok = all(lat.get(k, 0) == ref.get(k, 0) for k in ("left", "right", "bilateral"))
            lines.append(
                f"  reference_match (right/left/bilateral): "
                f"{'OK' if ok else 'MISMATCH'} expected {ref}"
            )
        lines.append("")

    lines.extend([
        "sidestep_note: HOA and CVA cohorts are laterality-skewed by design (walking path",
        "  may confound limb-specific deficits with turn direction). Document explicitly;",
        "  Moon (2020) and Trabassi (2022) do not report this confound.",
    ])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else RAW_ROOT
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    log_path = out_csv.parent / "laterality_audit.log"

    if not root.is_dir():
        print(f"Missing dataset root: {root}", file=sys.stderr)
        return 2

    df = collect_subjects(root)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    write_log(df, log_path)

    print(f"Wrote {out_csv} ({len(df)} subjects)")
    print(f"Wrote {log_path}")
    for pkey in TARGET_COHORTS:
        sub = df[df["pathologyKey"] == pkey]
        c = count_sides(sub)
        print(f"{pkey} laterality: right={c['right']} left={c['left']} bilateral={c['bilateral']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
