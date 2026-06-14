"""Stratified participant sampling for LOSO SHAP aggregation (MED-008)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def stratified_shap_subject_order(
    groups: np.ndarray,
    max_oof: int,
    *,
    cohorts: np.ndarray | None = None,
    seed: int = 42,
) -> list:
    """
    Select up to ``max_oof`` participant IDs with proportional cohort representation.

    When ``cohorts`` is None, subjects are chosen uniformly at random.
    """
    subjects = np.unique(groups)
    if len(subjects) <= max_oof:
        return list(subjects)

    rng = np.random.default_rng(seed)

    if cohorts is None:
        chosen = rng.choice(subjects, size=max_oof, replace=False)
        return list(chosen)

    by_cohort: dict[str, list] = defaultdict(list)
    for subj in subjects:
        row_idx = int(np.where(groups == subj)[0][0])
        cohort = str(cohorts[row_idx])
        by_cohort[cohort].append(subj)

    for cohort_subs in by_cohort.values():
        rng.shuffle(cohort_subs)

    n_total = len(subjects)
    targets: dict[str, int] = {}
    for cohort, cohort_subs in by_cohort.items():
        raw = len(cohort_subs) / n_total * max_oof
        targets[cohort] = max(1, int(np.floor(raw)))

    # Largest-remainder adjustment to hit exactly max_oof (or len(subjects) cap).
    while sum(targets.values()) > max_oof:
        cohort = max(targets, key=targets.get)
        if targets[cohort] > 1:
            targets[cohort] -= 1
        else:
            break

    remainders = sorted(
        (
            (len(by_cohort[c]) / n_total * max_oof - np.floor(len(by_cohort[c]) / n_total * max_oof), c)
            for c in by_cohort
        ),
        reverse=True,
    )
    idx = 0
    while sum(targets.values()) < max_oof and idx < len(remainders):
        _, cohort = remainders[idx]
        if targets[cohort] < len(by_cohort[cohort]):
            targets[cohort] += 1
        idx += 1

    selected: list = []
    for cohort, cohort_subs in by_cohort.items():
        take = min(targets.get(cohort, 0), len(cohort_subs))
        selected.extend(cohort_subs[:take])

    if len(selected) > max_oof:
        selected = selected[:max_oof]
    elif len(selected) < max_oof:
        remaining = [s for s in subjects if s not in set(selected)]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_oof - len(selected)])

    return selected
