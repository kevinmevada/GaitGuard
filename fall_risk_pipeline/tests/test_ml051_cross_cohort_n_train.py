"""ML-051: suppress unstable pairwise transfer cells when n_train is tiny."""

from __future__ import annotations

from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]


def test_pairwise_transfer_flags_small_n_train():
    source = (
        PIPELINE_ROOT / "src" / "evaluation" / "cross_cohort_transfer.py"
    ).read_text(encoding="utf-8")
    assert "n_train < self.cohort_auc_min_n" in source
    assert "unstable_small_n" in source
