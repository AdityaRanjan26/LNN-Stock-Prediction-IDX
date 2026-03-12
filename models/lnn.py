"""
models/lnn.py — Liquid Neural Network (LNN) implementation.

Architecture:
  CNN feature extractor
  → Stacked Wired LTC cells (NCP topology)
  → Multi-head attention
  → Direction classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) cell — vectorized over full sequence.
    Processes (B, T, input_size) → (B, T, hidden_size) in one forward call.
    """

    def __init__(self, input_size: int, hidden_size: int, ode_unfolds: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.ode_unfolds = ode_unfolds
        self.eps = 1e-8

        self.W_in  = nn.Linear(input_size, hidden_size, bias=True)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)

        self.tau   = nn.Parameter(torch.ones(hidden_size))
        self.E_rev = nn.Parameter(torch.zeros(hidden_size))
        self.gleak = nn.Parameter(torch.ones(hidden_size))
        self.Eleak = nn.Parameter(torch.zeros(hidden_size))

        nn.init.xavier_uniform_(self.W_in.weight)
        nn.init.orthogonal_(self.W_rec.weight)
        nn.init.uniform_(self.tau,   1.0, 10.0)
        nn.init.uniform_(self.E_rev, -1.0, 1.0)
        nn.init.uniform_(self.gleak, 0.1,  1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_size) → (B, T, hidden_size)"""
        B, T, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device, dtype=x.dtype)

        inp_proj = self.W_in(x)                          # (B, T, hidden)
        tau_c    = self.tau.clamp(min=self.eps)
        gl_c     = self.gleak.clamp(min=self.eps)
        dt       = 1.0 / self.ode_unfolds
        W_rec_T  = self.W_rec.weight.t()

        outputs = []
        for t in range(T):
            ip = inp_proj[:, t, :]
            for _ in range(self.ode_unfolds):
                act   = torch.sigmoid(ip + h @ W_rec_T)
                g_tot = gl_c + act
                E_w   = (gl_c * self.Eleak + act * self.E_rev) / (g_tot + self.eps)
                h     = h + ((-g_tot * h + g_tot * E_w) / tau_c) * dt
            outputs.append(h)

        return torch.stack(outputs, dim=1)               # (B, T, hidden)


class WiredLTC(nn.Module):
    """NCP-inspired wiring: inter → command → motor neurons with skip."""

    def __init__(self, input_size: int, inter: int, command: int,
                 motor: int, ode_unfolds: int = 3):
        super().__init__()
        self.inter_cell   = LTCCell(input_size, inter,   ode_unfolds)
        self.command_cell = LTCCell(inter,       command, ode_unfolds)
        self.motor_cell   = LTCCell(command,     motor,   ode_unfolds)
        self.skip         = nn.Linear(inter, motor, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hi = self.inter_cell(x)
        hc = self.command_cell(hi)
        hm = self.motor_cell(hc) + 0.1 * self.skip(hi)
        return hm


class LNNClassifier(nn.Module):
    """
    Full LNN model for binary direction classification.

    Pipeline:
      (B, seq_len, features)
      → CNN (residual) feature extractor
      → Stacked WiredLTC layers
      → Multi-head attention
      → FC → logit (direction up/down)
    """

    def __init__(self, n_features: int, seq_len: int = 30,
                 inter: int = 64, command: int = 32, motor: int = 16,
                 n_layers: int = 2, dropout: float = 0.3, ode_unfolds: int = 3):
        super().__init__()
        self.n_features = n_features

        # ── CNN feature extractor ──────────────────────────────
        self.cnn_in = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        self.cnn_res = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64)
        )
        self.cnn_drop = nn.Dropout(dropout)

        # ── Stacked WiredLTC ──────────────────────────────────
        self.ltc_layers = nn.ModuleList()
        self.ltc_norms  = nn.ModuleList()
        for i in range(n_layers):
            in_sz = 64 if i == 0 else motor
            self.ltc_layers.append(WiredLTC(in_sz, inter, command, motor, ode_unfolds))
            self.ltc_norms.append(nn.LayerNorm(motor))

        # ── Multi-head attention ──────────────────────────────
        self.attention = nn.MultiheadAttention(
            motor, num_heads=3, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(motor)
        self.attn_drop = nn.Dropout(dropout)

        # ── Classification head ───────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(motor, motor * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(motor * 4, 2)   # 2 classes: down / up
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, n_features) → logits: (B, 2)"""
        # CNN: expects (B, features, seq_len)
        xc = x.permute(0, 2, 1)
        xc = self.cnn_in(xc)
        xc = F.gelu(xc + self.cnn_res(xc))   # residual
        xc = self.cnn_drop(xc).permute(0, 2, 1)  # → (B, seq_len, 64)

        # Stacked LTC
        out = xc
        for ltc, norm in zip(self.ltc_layers, self.ltc_norms):
            out = norm(ltc(out))

        # Attention
        ao, _ = self.attention(out, out, out)
        out   = self.attn_norm(out + self.attn_drop(ao))

        # Take last timestep → classify
        feat = out[:, -1, :]
        return self.classifier(feat)           # (B, 2)
