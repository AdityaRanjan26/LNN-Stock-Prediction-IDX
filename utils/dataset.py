"""
utils/dataset.py — PyTorch Dataset and DataLoader construction.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, ConcatDataset
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple, Optional


class StockDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str],
                 seq_len: int, scaler: RobustScaler, target_col: str = 'Close'):

        self.seq_len = seq_len

        cols = [c for c in feature_cols if c in df.columns]
        feat = df[cols].values.astype(np.float64)
        feat = scaler.transform(feat)
        feat = np.nan_to_num(feat, nan=0.0, posinf=3.0, neginf=-3.0)
        feat = np.clip(feat, -10.0, 10.0).astype(np.float32)

        close  = df[target_col].values
        labels = (close[1:] > close[:-1]).astype(np.int64)

        # Align: we need seq_len rows of X to predict label at position seq_len
        # X[i : i+seq_len] → labels[i+seq_len-1]
        # Valid i: 0 to len(labels)-seq_len  (inclusive)
        # So n_samples = len(labels) - seq_len + 1
        # But labels[i+seq_len-1] max index = len(labels)-1
        # So max i = len(labels) - seq_len
        # n_samples = len(labels) - seq_len

        n = len(labels) - seq_len
        if n <= 0:
            self.X_seqs  = np.zeros((0, seq_len, len(cols)), dtype=np.float32)
            self.y_labels = np.zeros(0, dtype=np.int64)
        else:
            self.X_seqs   = np.stack([feat[i: i + seq_len] for i in range(n)])
            self.y_labels = np.array([labels[i + seq_len - 1] for i in range(n)],
                                      dtype=np.int64)

        # Keep labels for sampler
        self.labels = self.y_labels

    def __len__(self) -> int:
        return len(self.X_seqs)

    def __getitem__(self, idx: int):
        return (torch.from_numpy(self.X_seqs[idx]),
                torch.tensor(self.y_labels[idx], dtype=torch.long))


def make_loaders(
    train_dfs: Dict[str, pd.DataFrame],
    val_dfs:   Dict[str, pd.DataFrame],
    test_dfs:  Dict[str, pd.DataFrame],
    feature_cols: List[str],
    scaler: RobustScaler,
    cfg: dict
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Build train/val DataLoaders and per-ticker test datasets."""

    seq_len = cfg['seq_len']

    train_datasets, val_datasets = [], []
    per_ticker_test = {}

    for t in train_dfs:
        tr = StockDataset(train_dfs[t], feature_cols, seq_len, scaler)
        va = StockDataset(val_dfs[t],   feature_cols, seq_len, scaler)
        te = StockDataset(test_dfs[t],  feature_cols, seq_len, scaler)
        if len(tr) > 0: train_datasets.append(tr)
        if len(va) > 0: val_datasets.append(va)
        if len(te) > 0: per_ticker_test[t] = te

    if not train_datasets:
        raise RuntimeError("No training sequences — all stocks had too few rows.")
    if not val_datasets:
        raise RuntimeError("No validation sequences.")

    # ── Balanced sampler ─────────────────────────────────────
    all_labels = np.concatenate([ds.y_labels for ds in train_datasets])
    n_up    = int(all_labels.sum())
    n_down  = int(len(all_labels) - n_up)
    n_maj   = max(n_up, n_down)
    w_up    = n_maj / max(n_up,   1)
    w_down  = n_maj / max(n_down, 1)
    weights = np.where(all_labels == 1, w_up, w_down).astype(np.float32)
    sampler = WeightedRandomSampler(
        weights.tolist(),
        num_samples=2 * n_maj,
        replacement=True
    )
    print(f"[DATA] Train: {len(all_labels):,} samples | "
          f"UP: {n_up} ({100*n_up/len(all_labels):.1f}%) | "
          f"DOWN: {n_down} ({100*n_down/len(all_labels):.1f}%)")

    # Windows: num_workers > 0 causes deadlock on Windows
    nw = 2 if os.name == 'nt' else 4

    train_loader = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=cfg['batch_size'],
        sampler=sampler,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        ConcatDataset(val_datasets),
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=True
    )

    return train_loader, val_loader, per_ticker_test
