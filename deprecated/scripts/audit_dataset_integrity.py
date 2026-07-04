#!/usr/bin/env python3
"""Audit Voisard Figshare raw dataset integrity (trial counts, files, subjects)."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "fall_risk_pipeline" / "data" / "raw" / "voisard"
DEFAULT_LOG = REPO_ROOT / "fall_risk_pipeline" / "results" / "metrics" / "audit_dataset.log"

PATHOLOGY_KEY_MAP = {
    "HS": "Healthy",
    "HOA": "HipOA",
    "KOA": "KneeOA",
    "ACL": "ACL",
    "PD": "PD",
    "CVA": "CVA",
    "CIPN": "CIPN",
    "RIL": "RIL",
}

TIER_TO_PATHOLOGY = {
    "healthy": {"HS"},
    "neuro": {"PD", "CVA", "CIPN", "RIL"},
    "ortho": {"HOA", "KOA", "ACL"},
}

SENSOR_SUFFIXES = ("HE", "LB", "LF", "RF")
MIN_ROWS = 50
EXPECTED_TRIALS = {"HS": 360, "neuro": 784, "ortho": 212, "total": 1356}


def find_trial_dirs(root: Path) -> list[Path]:
    return sorted(
        p.parent
        for p in root.rglob("*_meta.json")
        if p.is_file()
    )


def load_meta(trial_dir: Path) -> tuple[dict | None, str | None]:
    meta_path = trial_dir / f"{trial_dir.name}_meta.json"
    if not meta_path.exists():
        return None, "missing_meta_json"
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        return None, f"corrupt_meta_json: {exc}"
    except OSError as exc:
        return None, f"corrupt_meta_json: {exc}"
    return meta, None


def audit_txt(path: Path) -> str | None:
    if not path.exists():
        return "missing_txt"
    try:
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    except Exception as exc:
        return f"corrupt_txt: {exc}"
    if df.empty:
        return "empty_txt"
    numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all(axis=None):
        return "non_numeric_txt"
    if len(df) < MIN_ROWS:
        return f"short_txt_rows={len(df)}"
    return None


def participant_key(trial_dir: Path, meta: dict) -> str:
    pkey = str(meta.get("pathologyKey", "")).upper()
    subject = meta.get("subject") or meta.get("participant_id") or trial_dir.parent.name
    return f"{pkey}:{subject}"


def run_audit(root: Path) -> dict:
    trial_dirs = find_trial_dirs(root)
    issues: list[dict] = []
    trials_by_tier: Counter[str] = Counter()
    trials_by_pkey: Counter[str] = Counter()
    trials_by_cohort: Counter[str] = Counter()
    participant_trials: defaultdict[str, list[str]] = defaultdict(list)

    for trial_dir in trial_dirs:
        rel = trial_dir.relative_to(root)
        tier = rel.parts[0] if rel.parts else "unknown"
        trial_name = trial_dir.name

        meta, meta_issue = load_meta(trial_dir)
        if meta_issue:
            issues.append({"trial": str(rel), "issue": meta_issue})
            trials_by_tier[tier] += 1
            continue

        pkey = str(meta.get("pathologyKey", "")).upper()
        trials_by_tier[tier] += 1
        trials_by_pkey[pkey] += 1
        trials_by_cohort[PATHOLOGY_KEY_MAP.get(pkey, pkey or "UNKNOWN")] += 1

        # Tier vs pathologyKey consistency
        allowed = TIER_TO_PATHOLOGY.get(tier, set())
        if pkey and allowed and pkey not in allowed:
            issues.append({
                "trial": str(rel),
                "issue": f"pathologyKey_mismatch: tier={tier} key={pkey}",
            })

        pid = participant_key(trial_dir, meta)
        participant_trials[pid].append(trial_name)

        for suffix in SENSOR_SUFFIXES:
            txt_path = trial_dir / f"{trial_name}_raw_data_{suffix}.txt"
            txt_issue = audit_txt(txt_path)
            if txt_issue:
                issues.append({
                    "trial": str(rel),
                    "issue": f"{suffix}: {txt_issue}",
                })

    sparse_subjects = {
        pid: trials
        for pid, trials in participant_trials.items()
        if len(trials) < 2
    }

    hs_count = trials_by_pkey.get("HS", 0)
    neuro_count = sum(trials_by_pkey[k] for k in TIER_TO_PATHOLOGY["neuro"])
    ortho_count = sum(trials_by_pkey[k] for k in TIER_TO_PATHOLOGY["ortho"])

    return {
        "root": str(root),
        "n_trial_dirs": len(trial_dirs),
        "trials_by_tier": dict(trials_by_tier),
        "trials_by_pathology_key": dict(sorted(trials_by_pkey.items())),
        "trials_by_cohort": dict(sorted(trials_by_cohort.items())),
        "hs_neuro_ortho": {"HS": hs_count, "neuro": neuro_count, "ortho": ortho_count},
        "expected_match": {
            "HS": hs_count == EXPECTED_TRIALS["HS"],
            "neuro": neuro_count == EXPECTED_TRIALS["neuro"],
            "ortho": ortho_count == EXPECTED_TRIALS["ortho"],
            "total": len(trial_dirs) == EXPECTED_TRIALS["total"],
        },
        "n_participants": len(participant_trials),
        "sparse_subjects_lt2": sparse_subjects,
        "n_issues": len(issues),
        "issues": issues,
    }


def print_report(report: dict) -> int:
    print("=" * 72)
    print("DATASET INTEGRITY AUDIT")
    print("=" * 72)
    print(f"Root: {report['root']}")
    print(f"Trial directories: {report['n_trial_dirs']}")
    print(f"Unique participants: {report['n_participants']}")
    print()

    print("--- Trial counts by tier (directory) ---")
    for tier, n in sorted(report["trials_by_tier"].items()):
        print(f"  {tier:8s}  {n:4d}")
    print()

    hno = report["hs_neuro_ortho"]
    exp = report["expected_match"]
    print("--- Expected pathology groups ---")
    print(f"  HS (Healthy):     {hno['HS']:4d}  expected 360  {'OK' if exp['HS'] else 'MISMATCH'}")
    print(f"  Neuro (PD/CVA/   {hno['neuro']:4d}  expected 784  {'OK' if exp['neuro'] else 'MISMATCH'}")
    print(f"       CIPN/RIL):")
    print(f"  Ortho (HOA/KOA/  {hno['ortho']:4d}  expected 212  {'OK' if exp['ortho'] else 'MISMATCH'}")
    print(f"       ACL):")
    print(f"  Total:            {report['n_trial_dirs']:4d}  expected 1356 {'OK' if exp['total'] else 'MISMATCH'}")
    print()

    print("--- Trial counts by pathologyKey ---")
    for key, n in report["trials_by_pathology_key"].items():
        cohort = PATHOLOGY_KEY_MAP.get(key, key)
        print(f"  {key:6s} ({cohort:8s})  {n:4d}")
    print()

    sparse = report["sparse_subjects_lt2"]
    print(f"--- Subjects with < 2 trials: {len(sparse)} ---")
    if sparse:
        for pid, trials in sorted(sparse.items()):
            print(f"  {pid}: {len(trials)} trial(s) -> {trials}")
    else:
        print("  (none)")
    print()

    print(f"--- File / integrity issues: {report['n_issues']} ---")
    if report["issues"]:
        by_type: Counter[str] = Counter()
        for item in report["issues"]:
            issue = item["issue"].split(":")[0]
            by_type[issue] += 1
        print("  By issue type:")
        for kind, n in by_type.most_common():
            print(f"    {kind}: {n}")
        print()
        print("  First 30 issues:")
        for item in report["issues"][:30]:
            print(f"    {item['trial']}: {item['issue']}")
        if report["n_issues"] > 30:
            print(f"    ... and {report['n_issues'] - 30} more")
    else:
        print("  (none)")
    print()

    all_ok = (
        all(report["expected_match"].values())
        and report["n_issues"] == 0
        and len(sparse) == 0
    )
    print("=" * 72)
    print("VERDICT:", "PASS" if all_ok else "ISSUES FOUND")
    print("=" * 72)
    return 0 if all_ok else 1


def write_audit_log(report: dict, log_path: Path) -> None:
    """Paper-ready log: cohort_counts, missing_files[], short_subjects[]."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    hno = report["hs_neuro_ortho"]
    exp = report["expected_match"]
    missing = [
        f"{item['trial']}: {item['issue']}"
        for item in report["issues"]
        if any(
            token in item["issue"]
            for token in ("missing_meta_json", "missing_txt", "corrupt_meta", "corrupt_txt", "empty_txt")
        )
    ]
    short_subjects = [
        f"{pid}: {trials}"
        for pid, trials in sorted(report["sparse_subjects_lt2"].items())
    ]

    lines = [
        "# audit_dataset.log — Voisard raw integrity (Section 3 citation)",
        f"# generated_utc: {datetime.now(timezone.utc).isoformat()}",
        f"# root: {report['root']}",
        f"# participants: {report['n_participants']}",
        "",
        "cohort_counts:",
        f"  HS:    {hno['HS']:4d}  (expected 360)  {'OK' if exp['HS'] else 'MISMATCH'}",
        f"  Neuro: {hno['neuro']:4d}  (expected 784)  {'OK' if exp['neuro'] else 'MISMATCH'}",
        f"  Ortho: {hno['ortho']:4d}  (expected 212)  {'OK' if exp['ortho'] else 'MISMATCH'}",
        f"  Total: {report['n_trial_dirs']:4d}  (expected 1356) {'OK' if exp['total'] else 'MISMATCH'}",
        "",
        "cohort_counts_by_pathologyKey:",
    ]
    for key, n in report["trials_by_pathology_key"].items():
        lines.append(f"  {key}: {n}")

    lines.extend(["", f"missing_files[]: ({len(missing)})"])
    if missing:
        lines.extend(f"  - {m}" for m in missing)
    else:
        lines.append("  (none)")

    lines.extend(["", f"short_subjects[]: ({len(short_subjects)})  # <2 trials; weak LOSO fold"])
    if short_subjects:
        lines.extend(f"  - {s}" for s in short_subjects)
    else:
        lines.append("  (none)")

    other_issues = [
        item for item in report["issues"]
        if f"{item['trial']}: {item['issue']}" not in missing
    ]
    lines.extend(["", f"other_issues[]: ({len(other_issues)})"])
    if other_issues:
        lines.extend(f"  - {item['trial']}: {item['issue']}" for item in other_issues)
    else:
        lines.append("  (none)")

    all_ok = (
        all(exp.values())
        and not missing
    )
    lines.extend([
        "",
        f"verdict: {'PASS' if all_ok else 'ISSUES_FOUND'}",
        f"note: short_subjects do not fail cohort counts; flag for LOSO sensitivity only.",
    ])
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    root = RAW_ROOT
    log_path = DEFAULT_LOG
    args = sys.argv[1:]
    if args:
        root = Path(args[0])
    if len(args) > 1:
        log_path = Path(args[1])
    if not root.is_dir():
        print(f"Dataset root not found: {root}", file=sys.stderr)
        return 2
    report = run_audit(root)
    write_audit_log(report, log_path)
    print(f"Wrote {log_path}")
    return print_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
