"""
BiLSTM autoencoder for healthy-gait manifold learning (Phase 3 primary model).

Encoder produces bottleneck activations h_t; decoder reconstructs per-sensor channel blocks.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.checkpoint_io import register_checkpoint_file, verify_checkpoint_bytes
from src.utils.reproducibility import get_pipeline_seed
from src.utils.torch_device import resolve_torch_device


@dataclass(frozen=True)
class SensorChannelSlice:
    name: str
    start: int
    end: int


class BiLSTMAutoencoder(nn.Module):
    """
    Bidirectional LSTM encoder → latent h_t; linear decoder → channel reconstruction.

    Input / output shape: (batch, n_channels, seq_len).
    """

    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        *,
        hidden_size: int = 128,
        latent_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.2,
        sensor_slices: list[SensorChannelSlice] | None = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.sensor_slices = sensor_slices or []

        self.encoder = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.to_latent = nn.Linear(hidden_size * 2, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_size * 2)
        self.decoder = nn.Linear(hidden_size * 2, n_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent sequence h_t of shape (B, T, latent_dim)."""
        x = x.transpose(1, 2)
        lstm_out, _ = self.encoder(x)
        return self.to_latent(lstm_out)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Reconstruct (B, n_channels, T) from latent sequence."""
        hidden = torch.relu(self.from_latent(h))
        recon = self.decoder(hidden)
        return recon.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.decode(h), h

    def per_sensor_mse(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Per-sensor block MSE (batch mean)."""
        out: dict[str, torch.Tensor] = {}
        for sl in self.sensor_slices:
            block = x[:, sl.start : sl.end, :]
            block_hat = x_hat[:, sl.start : sl.end, :]
            out[sl.name] = torch.mean((block - block_hat) ** 2, dim=(1, 2))
        out["total"] = torch.mean((x - x_hat) ** 2, dim=(1, 2))
        return out


def train_bilstm_autoencoder(
    X_train: np.ndarray,
    *,
    sensor_slices: list[SensorChannelSlice],
    config: dict[str, Any],
    checkpoint_path: Path,
) -> BiLSTMAutoencoder:
    """Train on Healthy-only windows; save state_dict + normalization stats."""
    ae_cfg = (
        (config.get("primary_model") or {}).get("bilstm_ae_ensemble", {}).get("bilstm_autoencoder")
        or (config.get("features") or {}).get("phase3_deep", {}).get("bilstm_autoencoder")
        or {}
    )
    hidden = int(ae_cfg.get("hidden_size", 128))
    latent_dim = int(ae_cfg.get("latent_dim", 64))
    max_epochs = int(ae_cfg.get("max_epochs", 40))
    batch_size = int(ae_cfg.get("batch_size", 64))
    lr = float(ae_cfg.get("learning_rate", 1e-3))
    patience = int(ae_cfg.get("early_stopping_patience", 8))
    seed = get_pipeline_seed(config)

    torch.manual_seed(seed)
    n_channels = int(X_train.shape[1])
    seq_len = int(X_train.shape[2])
    model = BiLSTMAutoencoder(
        n_channels,
        seq_len,
        hidden_size=hidden,
        latent_dim=latent_dim,
        sensor_slices=sensor_slices,
    )
    device = resolve_torch_device(config)
    model.to(device)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    n_val = max(1, int(0.1 * len(X_t)))
    perm = torch.randperm(len(X_t))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train_loader = DataLoader(
        TensorDataset(X_t[train_idx]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_x = X_t[val_idx].to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
    best_state = None
    stale = 0

    for _epoch in range(max_epochs):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, _ = model(batch)
            loss = torch.mean((recon - batch) ** 2)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            recon, _ = model(val_x)
            val_loss = float(torch.mean((recon - val_x) ** 2).item())
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

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "n_channels": n_channels,
            "seq_len": seq_len,
            "hidden_size": hidden,
            "latent_dim": latent_dim,
            "sensor_slices": [{"name": s.name, "start": s.start, "end": s.end} for s in sensor_slices],
            "val_mse": best_val,
        },
        checkpoint_path,
    )
    register_checkpoint_file(checkpoint_path, manifest_dir=checkpoint_path.parent)
    return model


def load_bilstm_autoencoder(
    checkpoint_path: Path,
    device: torch.device | None = None,
    *,
    require_manifest: bool = False,
) -> BiLSTMAutoencoder:
    device = device or resolve_torch_device()
    checkpoint_path = Path(checkpoint_path)
    raw_bytes = checkpoint_path.read_bytes()
    verify_checkpoint_bytes(
        checkpoint_path.name,
        raw_bytes,
        checkpoint_path.parent,
        require_manifest=require_manifest,
    )
    ckpt = torch.load(io.BytesIO(raw_bytes), map_location=device, weights_only=False)
    slices = [
        SensorChannelSlice(s["name"], s["start"], s["end"])
        for s in ckpt.get("sensor_slices", [])
    ]
    model = BiLSTMAutoencoder(
        int(ckpt["n_channels"]),
        int(ckpt["seq_len"]),
        hidden_size=int(ckpt.get("hidden_size", 128)),
        latent_dim=int(ckpt.get("latent_dim", 64)),
        sensor_slices=slices,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model
