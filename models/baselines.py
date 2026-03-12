"""
models/baselines.py — Baseline models for comparison.
All take (B, seq_len, n_features) → logits (B, 2).
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
                            bidirectional=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :]))


class GRUBaseline(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.gru  = nn.GRU(n_features, hidden, n_layers,
                           batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(self.drop(out[:, -1, :]))


class TransformerBaseline(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64,
                 nhead: int = 4, n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.head    = nn.Linear(d_model, 2)

    def forward(self, x):
        out = self.encoder(self.proj(x))
        return self.head(out[:, -1, :])
