"""
utils/features.py — Feature selection (MI) and scaling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


PROTECTED_COLS = {'Close', 'Log_Return', 'RSI', 'MACD', 'ATR_norm'}


def select_features_mi(
    train_dfs: Dict[str, pd.DataFrame],
    all_cols: List[str],
    n_select: int = 30,
    target_col: str = 'Close'
) -> List[str]:
    """Select top-N features by mutual information. Always keeps protected cols."""
    print(f"[FEAT] Mutual information selection (top {n_select})...")

    feat_cols = [c for c in all_cols if c not in [target_col, 'Open']]
    all_X, all_y = [], []

    for df in list(train_dfs.values())[:10]:
        cols_available = [c for c in feat_cols if c in df.columns]
        if not cols_available:
            continue
        X = df[cols_available].values
        y = (df[target_col].shift(-1) > df[target_col]).astype(int).values
        X, y = X[:-1], y[:-1]
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if mask.sum() > 10:
            all_X.append(X[mask])
            all_y.append(y[mask])

    if not all_X:
        print("[WARN] MI selection failed — returning all columns")
        selected = feat_cols[:n_select]
        for p in PROTECTED_COLS:
            if p in all_cols and p not in selected:
                selected.append(p)
        return selected

    X_all = np.nan_to_num(np.vstack(all_X), nan=0.0, posinf=3.0, neginf=-3.0)
    y_all = np.concatenate(all_y)

    # Use only columns available in first 10 dfs
    cols_available = [c for c in feat_cols if c in list(train_dfs.values())[0].columns]
    mi_scores = mutual_info_classif(X_all, y_all, random_state=42)
    scored    = sorted(zip(cols_available, mi_scores), key=lambda x: x[1], reverse=True)
    selected  = [f for f, _ in scored[:n_select]]

    # Always include protected columns
    for p in PROTECTED_COLS:
        if p in all_cols and p not in selected:
            selected.append(p)

    print(f"[FEAT] Top 5: {[f for f, _ in scored[:5]]}")
    return selected


def remove_correlated(
    selected_cols: List[str],
    train_dfs: Dict[str, pd.DataFrame],
    threshold: float = 0.95
) -> List[str]:
    """Remove highly correlated features, protecting key columns."""
    print(f"[FEAT] Removing correlated features (r>{threshold})...")

    dfs_list = list(train_dfs.values())[:8]
    cols_available = [c for c in selected_cols
                      if all(c in df.columns for df in dfs_list)]
    if len(cols_available) < 2:
        return selected_cols

    try:
        all_vals = np.nan_to_num(
            np.vstack([df[cols_available].values for df in dfs_list]),
            nan=0.0, posinf=3.0, neginf=-3.0
        )
        corr = np.corrcoef(all_vals.T)
        np.fill_diagonal(corr, 0.0)
    except Exception:
        return selected_cols

    drop = set()
    for i in range(len(cols_available)):
        if i in drop or cols_available[i] in PROTECTED_COLS:
            continue
        for j in range(i + 1, len(cols_available)):
            if j in drop or cols_available[j] in PROTECTED_COLS:
                continue
            if abs(corr[i, j]) > threshold:
                drop.add(j)

    kept = [c for i, c in enumerate(cols_available) if i not in drop]
    # Add back any selected_cols not in cols_available
    for c in selected_cols:
        if c not in cols_available and c not in kept:
            kept.append(c)

    print(f"[FEAT] Dropped {len(drop)} correlated → {len(kept)} remaining")
    return kept


def fit_scaler(train_dfs: Dict[str, pd.DataFrame],
               feature_cols: List[str]) -> RobustScaler:
    """Fit RobustScaler on training data only."""
    cols = [c for c in feature_cols if all(c in df.columns for df in train_dfs.values())]
    all_vals = np.vstack([df[cols].values for df in train_dfs.values()])
    all_vals = np.nan_to_num(all_vals, nan=0.0, posinf=3.0, neginf=-3.0)
    scaler = RobustScaler()
    scaler.fit(all_vals)
    return scaler
