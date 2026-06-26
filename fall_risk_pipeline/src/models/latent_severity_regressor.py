"""
Linear regression head on pooled BiLSTM-AE latent activations (severity / UPDRS proxy).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.reproducibility import get_pipeline_seed


class LatentSeverityRegressor(nn.Module):
    """MLP regression head: pooled h_t → scalar severity."""

    def __init__(self, latent_dim: int, *, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


def train_latent_regression_head(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
    *,
    Z_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
) -> LatentSeverityRegressor:
    cfg = (config.get("severity_regression") or {}).get("regression_head") or {}
    latent_dim = int(Z_train.shape[1])
    hidden = int(cfg.get("hidden_dim", 32))
    dropout = float(cfg.get("dropout", 0.1))
    lr = float(cfg.get("learning_rate", 1e-3))
    max_epochs = int(cfg.get("max_epochs", 80))
    patience = int(cfg.get("early_stopping_patience", 10))
    batch_size = int(cfg.get("batch_size", 64))
    seed = get_pipeline_seed(config)

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LatentSeverityRegressor(latent_dim, hidden_dim=hidden, dropout=dropout).to(device)
    Z_t = torch.tensor(Z_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Z_t, y_t), batch_size=batch_size, shuffle=True)

    val_x = val_y = None
    if Z_val is not None and y_val is not None and len(Z_val) > 0:
        val_x = torch.tensor(Z_val, dtype=torch.float32, device=device)
        val_y = torch.tensor(y_val, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    stale = 0

    for _ in range(max_epochs):
        model.train()
        for z_b, y_b in loader:
            z_b = z_b.to(device)
            y_b = y_b.to(device)
            opt.zero_grad()
            pred = model(z_b)
            loss = torch.mean((pred - y_b) ** 2)
            loss.backward()
            opt.step()

        if val_x is not None and val_y is not None:
            model.eval()
            with torch.no_grad():
                val_loss = float(torch.mean((model(val_x) - val_y) ** 2).item())
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_latent_regression(
    model: LatentSeverityRegressor,
    Z: np.ndarray,
    *,
    device: torch.device | None = None,
) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        z = torch.tensor(Z, dtype=torch.float32, device=device)
        return model(z).cpu().numpy()
