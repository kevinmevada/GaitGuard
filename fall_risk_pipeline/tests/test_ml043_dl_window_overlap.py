"""ML-043: deduplicate correlated overlapping windows for DL train/val."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from src.models.deep_models import (
    create_windows,
    independent_stride_window_indices,
)
from src.models.deep_trainer import DeepLearningPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
DEEP_TRAINER_PATH = REPO_ROOT / "fall_risk_pipeline" / "src" / "models" / "deep_trainer.py"
METHODS_PATH = REPO_ROOT / "docs" / "paper" / "methods.md"


def _loso_evaluate_source() -> str:
    tree = ast.parse(DEEP_TRAINER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_loso_evaluate":
            return ast.get_source_segment(
                DEEP_TRAINER_PATH.read_text(encoding="utf-8"), node
            ) or ""
    raise AssertionError("_loso_evaluate not found")


def test_create_windows_with_starts():
    signal = np.arange(4 * 20, dtype=np.float32).reshape(4, 20)
    windows, starts = create_windows(signal, window_len=8, overlap=0.5, return_starts=True)
    assert windows.shape[0] == len(starts)
    assert starts[0] == 0
    assert starts[1] == 4


def test_independent_stride_indices_one_per_block():
    window_len = 8
    starts = np.array([0, 4, 8, 12, 0, 4], dtype=int)
    trial_ids = np.array(["t1", "t1", "t1", "t1", "t2", "t2"], dtype=object)
    keep = independent_stride_window_indices(starts, trial_ids, window_len, seed=42)
    # t1 blocks 0 and 1; t2 block 0 only (starts 0 and 4 map to block 0)
    assert len(keep) == 3
    kept_trials = trial_ids[keep]
    assert set(kept_trials) == {"t1", "t2"}


def _synthetic_participants(n_pids: int = 4, windows_per_pid: int = 6):
    participants = {}
    for i in range(n_pids):
        pid = f"p{i}"
        starts = np.array([0, 128, 256, 384, 512, 640][:windows_per_pid], dtype=int)
        participants[pid] = {
            "windows": np.random.randn(windows_per_pid, 4, 256).astype(np.float32),
            "window_starts": starts,
            "window_trial_ids": np.array([f"{pid}_t0"] * windows_per_pid, dtype=object),
            "label": i % 3,
            "trial_ids": [f"{pid}_t0"],
        }
    return participants


def test_concat_deduplication_reduces_window_count():
    participants = _synthetic_participants()
    pids = list(participants.keys())
    all_x, _ = DeepLearningPipeline.concat_participant_windows(pids, participants)
    dedup_x, _ = DeepLearningPipeline.concat_participant_windows(
        pids,
        participants,
        independent_stride_only=True,
        window_len=256,
        seed=7,
    )
    assert len(dedup_x) < len(all_x)
    assert len(dedup_x) == 4 * 3  # 3 stride blocks per trial at len 256


def test_window_participant_ids_align_with_dedup():
    participants = _synthetic_participants()
    pids = list(participants.keys())[:2]
    _, y = DeepLearningPipeline.concat_participant_windows(
        pids,
        participants,
        independent_stride_only=True,
        window_len=256,
        seed=7,
    )
    row_pids = DeepLearningPipeline.window_participant_ids(
        pids,
        participants,
        independent_stride_only=True,
        window_len=256,
        seed=7,
    )
    assert len(row_pids) == len(y)


def test_loso_evaluate_exports_window_protocol():
    source = _loso_evaluate_source()
    assert "training_window_protocol" in source
    assert "independent_stride_only" in source


def test_methods_document_window_overlap_protocol():
    text = METHODS_PATH.read_text(encoding="utf-8")
    assert "ML-043" in text or "training_window_deduplication" in text
    assert "inference_window_protocol" in text or "soft-vote" in text.lower()
