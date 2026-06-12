"""Tests for per-fold DataLoader generator seeding in deep learning."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset


def _first_batch_indices(seed: int, n: int = 32, batch_size: int = 8) -> list[int]:
    ds = TensorDataset(torch.arange(n), torch.zeros(n))
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    batch = next(iter(loader))
    return batch[0].tolist()


def test_fresh_generator_per_fold_is_reproducible_in_isolation():
    seed_a = _first_batch_indices(42 + 3)
    seed_b = _first_batch_indices(42 + 3)
    assert seed_a == seed_b

    seed_other = _first_batch_indices(42 + 4)
    assert seed_a != seed_other
