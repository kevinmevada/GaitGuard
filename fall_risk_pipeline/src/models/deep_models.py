"""
src/models/deep_models.py
FINAL FIXED VERSION (robust + stable)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────

class GaitSequenceDataset(Dataset):

    def __init__(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.groups = groups

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────
# CNN
# ─────────────────────────────────────────

class CNN1D(nn.Module):

    def __init__(self, n_channels, seq_len, n_classes=2, base_filters=64, dropout=0.3):
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
        out = out + res_conv(x)
        return F.relu(out)

    def forward(self, x):
        x = self._res_block(x, self.conv1, self.bn1, self.conv1b, self.bn1b, self.res1)
        x = self._res_block(x, self.conv2, self.bn2, self.conv2b, self.bn2b, self.res2)
        x = self._res_block(x, self.conv3, self.bn3, self.conv3b, self.bn3b, self.res3)

        attn = torch.softmax(self.attention(x.transpose(1, 2)), dim=1)
        x = (x * attn.transpose(1, 2)).sum(dim=2)

        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ─────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────

class LSTMClassifier(nn.Module):

    def __init__(self, n_channels, n_classes=2, hidden_size=128, n_layers=2, dropout=0.3, n_heads=4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.norm(lstm_out + attn_out)

        out = out.mean(dim=1)

        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)


# ─────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────

class DeepModelTrainer:

    def __init__(self, config: dict):
        dl_cfg = config["deep_learning"]

        self.batch_size = dl_cfg["batch_size"]
        self.max_epochs = dl_cfg["max_epochs"]
        self.lr = dl_cfg["learning_rate"]
        self.patience = dl_cfg["early_stopping_patience"]

        dev = dl_cfg["device"]
        if dev == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(dev)

    def train(self, model, X_train, y_train, X_val, y_val):

        if len(X_train) == 0:
            raise ValueError("Empty training data")

        model = model.to(self.device)

        train_ds = GaitSequenceDataset(X_train, y_train)
        val_ds = GaitSequenceDataset(X_val, y_val)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size)

        counts = np.bincount(y_train)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum()
        weights = torch.tensor(weights, dtype=torch.float32).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)

        best_auc = 0
        patience_ctr = 0
        best_state = None

        for _ in range(self.max_epochs):

            model.train()

            for Xb, yb in train_dl:
                Xb, yb = Xb.to(self.device), yb.to(self.device)

                optimizer.zero_grad()
                loss = criterion(model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            val_auc = self._evaluate_auc(model, val_dl)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        return model

    def predict_proba(self, model, X):

        if len(X) == 0:
            return np.array([])

        model = model.to(self.device).eval()

        ds = GaitSequenceDataset(X, np.zeros(len(X), dtype=int))
        dl = DataLoader(ds, batch_size=self.batch_size)

        probs = []

        with torch.no_grad():
            for Xb, _ in dl:
                logits = model(Xb.to(self.device))
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(prob)

        return np.vstack(probs)

    def _evaluate_auc(self, model, dl):
        model.eval()

        all_y = []
        all_prob = []

        with torch.no_grad():
            for Xb, yb in dl:
                logits = model(Xb.to(self.device))
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

                all_prob.extend(prob)
                all_y.extend(yb.numpy())

        if len(set(all_y)) < 2:
            return 0.5

        return roc_auc_score(all_y, all_prob)


# ─────────────────────────────────────────
# Windowing
# ─────────────────────────────────────────

def create_windows(signal_dict, window_len, overlap, fs):

    if overlap >= 1.0:
        raise ValueError("Overlap must be < 1")

    step = max(1, int(window_len * (1 - overlap)))

    channels = []
    chan_names = []

    for pos in ["lower_back", "left_foot", "right_foot", "head"]:
        df = signal_dict.get(pos)

        if df is None or df.empty:
            continue

        for ax in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]:
            if ax in df.columns:
                channels.append(df[ax].values)
                chan_names.append(f"{pos}_{ax}")

    if not channels:
        return np.empty((0, 0, window_len)), []

    min_len = min(len(c) for c in channels)
    channels = [c[:min_len] for c in channels]

    mat = np.stack(channels, axis=0)

    windows = []
    start = 0

    while start + window_len <= mat.shape[1]:
        windows.append(mat[:, start:start + window_len])
        start += step

    if not windows:
        return np.empty((0, len(channels), window_len)), chan_names

    return np.stack(windows, axis=0), chan_names