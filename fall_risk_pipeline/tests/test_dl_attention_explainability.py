"""Tests for GaitTransformer attention weight extraction (ML-010)."""

from __future__ import annotations

import numpy as np
import torch

from src.models.deep_models import GaitTransformer, build_deep_model
from src.models.deep_trainer import extract_attention_weights


def test_extract_attention_weights_shape_and_finite():
    n_channels, seq_len, n_classes = 8, 64, 3
    model = build_deep_model("gait_transformer", n_channels, seq_len, n_classes)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, n_channels, seq_len)).astype(np.float32)

    weights = extract_attention_weights(model, X, torch.device("cpu"))

    assert weights.shape == (seq_len,)
    assert np.all(np.isfinite(weights))
    assert weights.min() >= 0.0
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)


def test_gait_transformer_forward_still_runs_after_extraction():
    model = GaitTransformer(n_channels=4, seq_len=32, n_classes=3, n_layers=2, n_heads=2)
    x = torch.randn(2, 4, 32)
    logits = model(x)
    assert logits.shape == (2, 3)
