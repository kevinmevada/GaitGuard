"""
Deep learning models for raw IMU gait classification.

Architectures
─────────────
1.  InceptionTime  – multi-scale 1-D Inception residual network
    (Fawaz et al., "InceptionTime: Finding AlexNet for Time Series
    Classification", Data Mining Knowl. Disc., 2020)
2.  Gait Transformer  – positional-encoding + multi-head self-attention
    encoder (Vaswani et al., 2017; adapted for 1-D sensor streams)
3.  TCN  – Temporal Convolutional Network with dilated causal convolutions
    (Bai et al., "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling", 2018)
4.  CNN-1D  – residual 1-D CNN with channel attention (baseline)
5.  BiLSTM-Attention  – bidirectional LSTM with multi-head attention

All models accept (batch, channels, time) and output (batch, n_classes).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from collections.abc import Callable
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.utils.reproducibility import get_pipeline_seed
from src.utils.torch_device import resolve_torch_device


# ═════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════

class GaitSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = groups

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def participant_balanced_ce_class_weights(
    y: np.ndarray,
    participant_ids: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """
    CrossEntropyLoss class weights from **participant** counts (not windows).

    Matches sklearn/XGBoost multiclass balancing: N / (n_classes * count_per_class)
    where N is the number of unique training participants.
    """
    y = np.asarray(y, dtype=int)
    pids = np.asarray(participant_ids)
    if len(y) != len(pids):
        raise ValueError("y and participant_ids must have the same length")

    pid_to_label: dict[str, int] = {}
    for pid, label in zip(pids, y):
        key = str(pid)
        label_i = int(label)
        if key in pid_to_label and pid_to_label[key] != label_i:
            raise ValueError(f"Participant {key} has conflicting window labels")
        pid_to_label[key] = label_i

    participant_labels = np.fromiter(pid_to_label.values(), dtype=int)
    counts = np.bincount(participant_labels, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    return (len(participant_labels) / (n_classes * counts)).astype(np.float32)


def window_level_ce_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Legacy inverse-frequency weights from duplicated per-window labels."""
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    return (weights / weights.sum()).astype(np.float32)


# ═════════════════════════════════════════════════════════════════
# 1.  InceptionTime
# ═════════════════════════════════════════════════════════════════

class _InceptionBlock(nn.Module):
    """Single Inception module with three parallel conv branches + MaxPool."""

    def __init__(self, in_ch: int, out_ch: int = 32, bottleneck: int = 32):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, bottleneck, 1, bias=False) if in_ch > 1 else None
        bn_ch = bottleneck if self.bottleneck else in_ch

        self.conv10 = nn.Conv1d(bn_ch, out_ch, kernel_size=10, padding=4, bias=False)
        self.conv20 = nn.Conv1d(bn_ch, out_ch, kernel_size=20, padding=9, bias=False)
        self.conv40 = nn.Conv1d(bn_ch, out_ch, kernel_size=40, padding=19, bias=False)

        self.mp = nn.MaxPool1d(3, stride=1, padding=1)
        self.conv_mp = nn.Conv1d(in_ch, out_ch, 1, bias=False)

        self.bn = nn.BatchNorm1d(out_ch * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bn = self.bottleneck(x) if self.bottleneck else x
        # Even kernel sizes (10, 20, 40) yield length L-1 vs MaxPool L for even seq_len;
        # align branch lengths before concat (e.g. window_len=256).
        branches = [
            self.conv10(x_bn),
            self.conv20(x_bn),
            self.conv40(x_bn),
            self.conv_mp(self.mp(x)),
        ]
        t = min(b.size(-1) for b in branches)
        branches = [b[..., :t] for b in branches]
        y = torch.cat(branches, dim=1)
        return F.relu(self.bn(y))

    def branch_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Parallel branch activations before concat (lengths 10, 20, 40, MP)."""
        x_bn = self.bottleneck(x) if self.bottleneck else x
        branches = {
            "ms10": self.conv10(x_bn),
            "ms20": self.conv20(x_bn),
            "ms40": self.conv40(x_bn),
            "ms_mp": self.conv_mp(self.mp(x)),
        }
        t = min(b.size(-1) for b in branches.values())
        return {k: v[..., :t] for k, v in branches.items()}


class InceptionMultiscaleExtractor(nn.Module):
    """
    Multi-scale context maps from the first Inception block (kernels 10/20/40).

    Fawaz et al. 2020 — micro-tremor vs macro-gait parallel filters.
    """

    def __init__(self, n_channels: int, filters: int = 32, bottleneck: int = 32):
        super().__init__()
        self.block = _InceptionBlock(n_channels, filters, bottleneck)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        maps = self.block.branch_maps(x)
        feats = []
        for branch in ("ms10", "ms20", "ms40", "ms_mp"):
            gap = maps[branch].mean(dim=2)
            feats.append(gap)
        return torch.cat(feats, dim=1)

    @staticmethod
    def feature_names(filters: int = 32) -> list[str]:
        names: list[str] = []
        for branch in ("ms10", "ms20", "ms40", "ms_mp"):
            for i in range(filters):
                names.append(f"it_{branch}_{i:02d}")
        return names


class InceptionTime(nn.Module):
    """
    InceptionTime: stacked Inception blocks with residual shortcuts.
    Fawaz et al. 2020.  Benchmark leader on UCR/UEA archives.
    """

    def __init__(self, n_channels: int, seq_len: int, n_classes: int = 3,
                 n_blocks: int = 6, filters: int = 32, bottleneck: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        layers, shortcuts = [], []
        in_ch = n_channels
        for i in range(n_blocks):
            layers.append(_InceptionBlock(in_ch, filters, bottleneck))
            if i % 3 == 2:
                shortcuts.append(nn.Sequential(
                    nn.Conv1d(n_channels if i == 2 else filters * 4, filters * 4, 1, bias=False),
                    nn.BatchNorm1d(filters * 4),
                ))
            in_ch = filters * 4

        self.blocks = nn.ModuleList(layers)
        self.shortcuts = nn.ModuleList(shortcuts)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i % 3 == 2:
                sc_idx = i // 3
                sc = self.shortcuts[sc_idx](residual)
                t = min(x.size(-1), sc.size(-1))
                x = x[..., :t] + sc[..., :t]
                x = F.relu(x)
                residual = x
        x = self.gap(x).squeeze(-1)
        return self.fc(self.dropout(x))


# ═════════════════════════════════════════════════════════════════
# 2.  Gait Transformer
# ═════════════════════════════════════════════════════════════════

class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class GaitTransformer(nn.Module):
    """
    Transformer encoder for 1-D sensor time series.
    Projects sensor channels → d_model, adds positional encoding, runs
    n_layers of multi-head self-attention, and pools to class logits.
    """

    def __init__(self, n_channels: int, seq_len: int, n_classes: int = 3,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 4,
                 dim_ff: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_enc = _PositionalEncoding(d_model, max_len=seq_len + 64, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, T, C)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.fc(self.dropout(x))


# ═════════════════════════════════════════════════════════════════
# 3.  TCN  —  Temporal Convolutional Network
# ═════════════════════════════════════════════════════════════════

class _TCNBlock(nn.Module):
    """Two dilated causal conv layers with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int,
                 dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(2)
        out = self.dropout(F.relu(self.bn1(self.conv1(x)[:, :, :T])))
        out = self.dropout(F.relu(self.bn2(self.conv2(out)[:, :, :T])))
        return F.relu(out + self.residual(x))


class TCN(nn.Module):
    """
    Temporal Convolutional Network (Bai et al. 2018).
    Stacks dilated causal 1-D convolutions with exponentially growing
    receptive field, followed by global average pooling.
    """

    def __init__(self, n_channels: int, seq_len: int, n_classes: int = 3,
                 n_blocks: int = 6, filters: int = 64, kernel_size: int = 7,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        in_ch = n_channels
        for i in range(n_blocks):
            layers.append(_TCNBlock(in_ch, filters, kernel_size, dilation=2 ** i, dropout=dropout))
            in_ch = filters
        self.network = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(self.dropout(x))


# ═════════════════════════════════════════════════════════════════
# 4.  CNN-1D with residual blocks + channel attention
# ═════════════════════════════════════════════════════════════════

class CNN1D(nn.Module):
    def __init__(self, n_channels: int, seq_len: int, n_classes: int = 3,
                 base_filters: int = 64, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, base_filters, 7, padding=3)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.conv1b = nn.Conv1d(base_filters, base_filters, 5, padding=2)
        self.bn1b = nn.BatchNorm1d(base_filters)
        self.res1 = nn.Conv1d(n_channels, base_filters, 1)

        self.conv2 = nn.Conv1d(base_filters, base_filters * 2, 5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm1d(base_filters * 2)
        self.conv2b = nn.Conv1d(base_filters * 2, base_filters * 2, 3, padding=1)
        self.bn2b = nn.BatchNorm1d(base_filters * 2)
        self.res2 = nn.Conv1d(base_filters, base_filters * 2, 1, stride=2)

        self.conv3 = nn.Conv1d(base_filters * 2, base_filters * 4, 3, padding=1, stride=2)
        self.bn3 = nn.BatchNorm1d(base_filters * 4)
        self.conv3b = nn.Conv1d(base_filters * 4, base_filters * 4, 3, padding=1)
        self.bn3b = nn.BatchNorm1d(base_filters * 4)
        self.res3 = nn.Conv1d(base_filters * 2, base_filters * 4, 1, stride=2)

        self.attention = nn.Linear(base_filters * 4, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(base_filters * 4, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def _res_block(self, x, conv_a, bn_a, conv_b, bn_b, res_conv):
        out = F.relu(bn_a(conv_a(x)))
        out = bn_b(conv_b(out))
        return F.relu(out + res_conv(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._res_block(x, self.conv1, self.bn1, self.conv1b, self.bn1b, self.res1)
        x = self._res_block(x, self.conv2, self.bn2, self.conv2b, self.bn2b, self.res2)
        x = self._res_block(x, self.conv3, self.bn3, self.conv3b, self.bn3b, self.res3)
        attn = torch.softmax(self.attention(x.transpose(1, 2)), dim=1)
        x = (x * attn.transpose(1, 2)).sum(dim=2)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ═════════════════════════════════════════════════════════════════
# 5.  BiLSTM with multi-head attention
# ═════════════════════════════════════════════════════════════════

class LSTMClassifier(nn.Module):
    def __init__(self, n_channels: int, seq_len: int = 0, n_classes: int = 3,
                 hidden_size: int = 128, n_layers: int = 2, dropout: float = 0.3,
                 n_heads: int = 4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels, hidden_size=hidden_size, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out)
        out = out.mean(dim=1)
        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)


# ═════════════════════════════════════════════════════════════════
# 6.  DeepConvLSTM (Ordóñez & Roggen 2016)
# ═════════════════════════════════════════════════════════════════

class DeepConvLSTM(nn.Module):
    """
    Convolutional feature maps + stacked LSTM classifier.

    Ordóñez & Roggen 2016 — deep conv front-end over sensor channels,
    LSTM temporal pooling, linear head. Benchmark for BiLSTM-AE comparison.
    """

    def __init__(
        self,
        n_channels: int,
        seq_len: int,
        n_classes: int = 3,
        *,
        conv_filters: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, conv_filters, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.conv2 = nn.Conv1d(conv_filters, conv_filters, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(conv_filters)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(
            conv_filters,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(self.dropout(out))


# ═════════════════════════════════════════════════════════════════
# Model registry
# ═════════════════════════════════════════════════════════════════

DEEP_MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "inception_time": InceptionTime,
    "deep_conv_lstm": DeepConvLSTM,
    "gait_transformer": GaitTransformer,
    "tcn": TCN,
    "cnn1d": CNN1D,
    "bilstm_attention": LSTMClassifier,
}


def build_deep_model(name: str, n_channels: int, seq_len: int,
                     n_classes: int, **kwargs) -> nn.Module:
    cls = DEEP_MODEL_REGISTRY[name]
    return cls(n_channels=n_channels, seq_len=seq_len, n_classes=n_classes, **kwargs)


# ═════════════════════════════════════════════════════════════════
# Windowing
# ═════════════════════════════════════════════════════════════════

CHANNEL_ORDER = [
    "acc_x", "acc_y", "acc_z",
    "gyr_x", "gyr_y", "gyr_z",
    "acc_x_grav_free", "acc_y_grav_free", "acc_z_grav_free",
    "acc_resultant", "gyr_resultant",
    "roll_rad", "pitch_rad",
]


def trial_to_tensor(
    trial_id: str,
    signals_dir,
    sensor_positions: list[str],
    channels: list[str] = CHANNEL_ORDER,
    *,
    require_all_sensors: bool = True,
) -> np.ndarray | None:
    """
    Load cleaned per-sensor parquets for one trial and stack into a
    (n_channels_total, T_min) array.  Returns None on failure.

    When ``require_all_sensors=False``, stacks every available sensor (e.g. DAPHNET LB-only).
    """
    from pathlib import Path
    signals_dir = Path(signals_dir)
    arrays = []
    min_len = float("inf")
    for pos in sensor_positions:
        path = signals_dir / f"{trial_id}_{pos}.parquet"
        if not path.exists():
            if require_all_sensors:
                return None
            continue
        import pandas as pd
        df = pd.read_parquet(path)
        usable = [c for c in channels if c in df.columns]
        if not usable:
            if require_all_sensors:
                return None
            continue
        arr = df[usable].values.T  # (usable_ch, T)
        arrays.append(arr)
        min_len = min(min_len, arr.shape[1])

    if not arrays or min_len < 64:
        return None
    stacked = np.concatenate([a[:, :min_len] for a in arrays], axis=0)
    return stacked.astype(np.float32)


def create_windows(
    signal: np.ndarray,
    window_len: int,
    overlap: float,
    *,
    return_starts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Slide fixed-length windows over a (C, T) signal array for a **single trial**.

    Windows must not span trial boundaries — call once per trial, never on
    concatenated multi-trial buffers (see ``preprocessing.windowing``).
    Returns (n_windows, C, window_len), optionally with start indices per window.
    """
    if overlap >= 1.0:
        raise ValueError("Overlap must be < 1")
    step = max(1, int(window_len * (1 - overlap)))
    C, T = signal.shape
    windows = []
    starts: list[int] = []
    start = 0
    while start + window_len <= T:
        windows.append(signal[:, start:start + window_len])
        starts.append(start)
        start += step
    if not windows:
        empty = np.empty((0, C, window_len), dtype=np.float32)
        if return_starts:
            return empty, np.array([], dtype=int)
        return empty
    stacked = np.stack(windows, axis=0)
    if return_starts:
        return stacked, np.asarray(starts, dtype=int)
    return stacked


def independent_stride_window_indices(
    starts: np.ndarray,
    trial_ids: np.ndarray,
    window_len: int,
    *,
    seed: int,
) -> np.ndarray:
    """
    Select at most one window per non-overlapping stride block within each trial (ML-043).

    With 50% overlap, consecutive windows share half their samples; this keeps one window
    per ``[k * window_len, (k+1) * window_len)`` block per trial for training/validation.
    """
    from collections import defaultdict

    if len(starts) != len(trial_ids):
        raise ValueError("starts and trial_ids must have the same length")

    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, (tid, start) in enumerate(zip(trial_ids, starts, strict=True)):
        block = int(start) // window_len
        groups[(str(tid), block)].append(i)

    keep = sorted(int(rng.choice(indices)) for indices in groups.values())
    return np.asarray(keep, dtype=int)


# ═════════════════════════════════════════════════════════════════
# Trainer
# ═════════════════════════════════════════════════════════════════

class DeepModelTrainer:

    def __init__(self, config: dict):
        dl_cfg = config["deep_learning"]
        self.batch_size = dl_cfg["batch_size"]
        self.max_epochs = dl_cfg["max_epochs"]
        self.lr = dl_cfg["learning_rate"]
        self.patience = dl_cfg["early_stopping_patience"]
        self.seed = get_pipeline_seed(config)

        self.device = resolve_torch_device(config, log=False)

        # Mixed precision: substantial speedup on RTX GPUs while preserving stability.
        self.mixed_precision = bool(dl_cfg.get("mixed_precision", True)) and self.device.type == "cuda"
        amp_dtype = str(dl_cfg.get("amp_dtype", "float16")).lower()
        self.amp_dtype = torch.bfloat16 if amp_dtype in {"bfloat16", "bf16"} else torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.mixed_precision and self.amp_dtype == torch.float16,
        )

    def _autocast_context(self):
        if self.mixed_precision:
            return torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)
        return nullcontext()

    def train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        *,
        participant_ids_train: np.ndarray | None = None,
        participant_ids_val: np.ndarray | None = None,
        shuffle_seed: int | None = None,
        on_epoch: Callable[[int, float, float, float], None] | None = None,
    ) -> nn.Module:
        if len(X_train) == 0:
            raise ValueError("Empty training data")

        model = model.to(self.device)
        train_ds = GaitSequenceDataset(X_train, y_train)
        val_ds = GaitSequenceDataset(X_val, y_val)

        train_generator = torch.Generator()
        train_generator.manual_seed(self.seed if shuffle_seed is None else int(shuffle_seed))
        train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=train_generator,
            drop_last=len(train_ds) > self.batch_size,
        )
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        n_classes = int(max(y_train.max(), y_val.max())) + 1
        if participant_ids_train is not None:
            weights = participant_balanced_ce_class_weights(
                y_train, participant_ids_train, n_classes
            )
        else:
            logger.warning(
                "DeepModelTrainer.train called without participant_ids_train; "
                "using window-level class weights (deprecated)"
            )
            weights = window_level_ce_class_weights(y_train, n_classes)
        weights_t = torch.tensor(weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weights_t)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        best_auc = 0.0
        patience_ctr = 0
        best_state = None

        for epoch in range(self.max_epochs):
            model.train()
            running_loss = 0.0
            n_batches = 0
            for Xb, yb in train_dl:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                with self._autocast_context():
                    logits = model(Xb)
                    loss = criterion(logits, yb)
                if self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                running_loss += float(loss.item())
                n_batches += 1

            scheduler.step()
            val_auc = self._evaluate_auc(
                model, val_dl, n_classes, participant_ids_val=participant_ids_val
            )
            mean_loss = running_loss / max(n_batches, 1)

            if on_epoch is not None:
                on_epoch(epoch + 1, mean_loss, val_auc, best_auc)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        return model

    def predict_proba(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.array([])
        model = model.to(self.device).eval()
        ds = GaitSequenceDataset(X, np.zeros(len(X), dtype=int))
        dl = DataLoader(ds, batch_size=self.batch_size)
        probs = []
        with torch.no_grad():
            for Xb, _ in dl:
                with self._autocast_context():
                    logits = model(Xb.to(self.device))
                # Force FP32 before softmax export to avoid AMP underflow/rounding
                # that can produce invalid probability rows for sklearn AUC checks.
                prob = torch.softmax(logits.float(), dim=1).cpu().numpy().astype(np.float32)
                probs.append(prob)
        return np.vstack(probs)

    def _evaluate_auc(
        self,
        model: nn.Module,
        dl: DataLoader,
        n_classes: int,
        *,
        participant_ids_val: np.ndarray | None = None,
    ) -> float:
        """Validation AUC; aggregates window probs per participant when IDs given (ML-016)."""
        model.eval()
        all_y: list[int] = []
        all_prob: list[np.ndarray] = []
        with torch.no_grad():
            for Xb, yb in dl:
                with self._autocast_context():
                    logits = model(Xb.to(next(model.parameters()).device))
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                all_prob.append(prob)
                all_y.extend(yb.numpy().tolist())

        if not all_y:
            return 0.5

        y_arr = np.asarray(all_y, dtype=int)
        prob_arr = np.vstack(all_prob)

        if participant_ids_val is not None and len(participant_ids_val) == len(y_arr):
            pids = np.asarray(participant_ids_val)
            unique_pids = np.unique(pids)
            agg_y: list[int] = []
            agg_prob: list[np.ndarray] = []
            for pid in unique_pids:
                mask = pids == pid
                agg_y.append(int(y_arr[mask][0]))
                agg_prob.append(prob_arr[mask].mean(axis=0))
            y_arr = np.asarray(agg_y, dtype=int)
            prob_arr = np.vstack(agg_prob)

        if len(np.unique(y_arr)) < 2:
            return 0.5
        try:
            if n_classes == 2:
                return float(roc_auc_score(y_arr, prob_arr[:, 1]))
            return float(
                roc_auc_score(y_arr, prob_arr, multi_class="ovr", average="macro")
            )
        except ValueError:
            return 0.5
