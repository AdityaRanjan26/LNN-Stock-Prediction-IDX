"""
utils/data_loader.py — Data loading, cleaning, feature engineering.
All functions are pure (no side effects) and crash-safe.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────
# 1. RAW DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_ticker(ticker: str, period: str = '7y') -> Optional[pd.DataFrame]:
    """Download OHLCV data for one ticker. Returns None if insufficient data."""
    try:
        raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if raw is None or len(raw) < 300:
            print(f"  [SKIP] {ticker}: insufficient data ({len(raw) if raw is not None else 0} rows)")
            return None

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Keep only OHLCV
        needed = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [c for c in needed if c not in raw.columns]
        if missing:
            print(f"  [SKIP] {ticker}: missing columns {missing}")
            return None

        df = raw[needed].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Remove duplicate dates
        df = df[~df.index.duplicated(keep='last')]

        # Remove rows with zero or negative prices
        df = df[(df['Close'] > 0) & (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0)]

        # Remove extreme daily moves (>50% in one day = data error)
        daily_ret = df['Close'].pct_change().abs()
        df = df[daily_ret < 0.50]

        # Forward fill up to 3 days (handles holidays), then drop remaining NaN
        df = df.ffill(limit=3).dropna()

        if len(df) < 200:
            print(f"  [SKIP] {ticker}: too few rows after cleaning ({len(df)})")
            return None

        print(f"  [OK] {ticker}: {len(df)} rows")
        return df

    except Exception as e:
        print(f"  [SKIP] {ticker}: {e}")
        return None


def load_macro(macro_tickers: Dict[str, str], period: str = '7y') -> pd.DataFrame:
    """Download macro indicators. Returns empty DataFrame if all fail."""
    frames = {}
    print("[MACRO] Loading macro data...")
    for name, ticker in macro_tickers.items():
        try:
            raw = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if raw is None or len(raw) < 100:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            close = raw['Close'].sort_index()
            ret   = np.log(close / close.shift(1)).fillna(0.0)
            frames[f'Macro_{name}_ret']   = ret
            frames[f'Macro_{name}_mom5']  = ret.rolling(5).sum().fillna(0.0)
            frames[f'Macro_{name}_mom20'] = ret.rolling(20).sum().fillna(0.0)
            print(f"  [OK] {name}: {len(close)} rows")
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    if not frames:
        print("  [WARN] No macro data — continuing without it")
        return pd.DataFrame()

    macro_df = pd.DataFrame(frames)
    macro_df.index = pd.to_datetime(macro_df.index)
    return macro_df.ffill(limit=3).fillna(0.0)


# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators. Input: clean OHLCV DataFrame."""
    df = df.copy()
    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume'].replace(0, np.nan).ffill()

    # ── Trend ────────────────────────────────────────────────
    for w in [5, 10, 20, 50]:
        df[f'SMA_{w}'] = close.rolling(w).mean()
        df[f'EMA_{w}'] = close.ewm(span=w, adjust=False).mean()

    df['SMA_20_50_cross'] = (df['SMA_20'] > df['SMA_50']).astype(float)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist']   = df['MACD'] - df['MACD_signal']

    # ── Momentum ─────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    for w in [3, 5, 10, 20]:
        df[f'ROC_{w}'] = close.pct_change(w)

    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    df['Stoch_K'] = 100 * (close - low14) / (high14 - low14 + 1e-8)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ── Volatility ───────────────────────────────────────────
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['BB_upper'] = bb_mid + 2 * bb_std
    df['BB_lower'] = bb_mid - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (bb_mid + 1e-8)
    df['BB_pos']   = (close - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-8)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['ATR']      = tr.rolling(14).mean()
    df['ATR_norm'] = df['ATR'] / (close + 1e-8)

    log_ret = np.log(close / close.shift(1))
    df['HV_10']     = log_ret.rolling(10).std() * np.sqrt(252)
    df['HV_20']     = log_ret.rolling(20).std() * np.sqrt(252)
    df['Log_Return'] = log_ret

    # ── Volume ───────────────────────────────────────────────
    vol_sma = volume.rolling(20).mean()
    df['Volume_ratio'] = volume / (vol_sma + 1e-8)
    df['OBV']          = (np.sign(close.diff()) * volume).cumsum()
    df['Log_Volume']   = np.log(volume / (vol_sma + 1e-8) + 1e-8)

    # ── Price-derived ────────────────────────────────────────
    for w in [1, 2, 3, 5]:
        df[f'Ret_{w}d'] = log_ret.rolling(w).sum()

    df['HL_ratio']  = (high - low) / (close + 1e-8)
    df['OC_ratio']  = (close - df['Open']) / (df['Open'] + 1e-8)
    df['Gap']       = (df['Open'] - close.shift(1)) / (close.shift(1) + 1e-8)

    # ── Regime ───────────────────────────────────────────────
    sma200 = close.rolling(200, min_periods=50).mean()
    df['Regime_bull']     = ((close > sma200) & (df['SMA_20'] > df['SMA_50'])).astype(float)
    df['Regime_bear']     = ((close < sma200) & (df['SMA_20'] < df['SMA_50'])).astype(float)
    df['Regime_strength'] = (close - sma200) / (sma200 + 1e-8)

    # ── Calendar ─────────────────────────────────────────────
    df['Day_sin']   = np.sin(2 * np.pi * df.index.dayofweek / 5)
    df['Day_cos']   = np.cos(2 * np.pi * df.index.dayofweek / 5)
    df['Month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    return df


def add_sector_momentum(df: pd.DataFrame, peers: List[pd.DataFrame]) -> pd.DataFrame:
    """Add sector peer average return (lagged 1 day — zero leakage)."""
    if not peers:
        return df
    try:
        peer_rets = pd.concat([
            np.log(p['Close'] / p['Close'].shift(1)).rename(f'peer_{i}')
            for i, p in enumerate(peers)
        ], axis=1)
        sector_ret = peer_rets.mean(axis=1).shift(1)   # shift(1) = lagged
        df = df.copy()
        df['Sector_mom'] = sector_ret.reindex(df.index).fillna(0.0)
    except Exception:
        pass
    return df


def merge_macro(df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join macro features onto stock df. Safe if macro_df is empty."""
    if macro_df.empty:
        return df
    try:
        merged = df.join(macro_df, how='left')
        merged[macro_df.columns] = merged[macro_df.columns].ffill(limit=3).fillna(0.0)
        return merged
    except Exception:
        return df


def clip_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clip known features to valid ranges before scaling."""
    clips = {
        'RSI':          (0, 100),
        'Stoch_K':      (0, 100),
        'Stoch_D':      (0, 100),
        'BB_pos':       (-0.5, 1.5),
        'Volume_ratio': (0, 10),
        'ATR_norm':     (0, 0.2),
        'HV_10':        (0, 3),
        'HV_20':        (0, 3),
    }
    df = df.copy()
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df

def explore_dataset(raw_dfs: dict, plots_dir: str):
    """Print EDA summary and save distribution plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import os

    print(f"\n{'='*60}")
    print(f"  EXPLORATORY DATA ANALYSIS")
    print(f"{'='*60}")

    all_rows, all_nulls, all_returns = [], [], []

    for t, df in raw_dfs.items():
        rows     = len(df)
        nulls    = df.isnull().sum().sum()
        ret      = df['Close'].pct_change().dropna()
        all_rows.append(rows)
        all_nulls.append(nulls)
        all_returns.extend(ret.tolist())
        print(f"  {t:<12} rows={rows:<6} nulls={nulls:<4} "
              f"mean_ret={ret.mean()*100:.3f}%  "
              f"std={ret.std()*100:.2f}%  "
              f"min={ret.min()*100:.1f}%  "
              f"max={ret.max()*100:.1f}%")

    print(f"\n  Total stocks  : {len(raw_dfs)}")
    print(f"  Avg rows/stock: {int(sum(all_rows)/len(all_rows))}")
    print(f"  Total nulls   : {sum(all_nulls)}")
    print(f"  Return mean   : {float(sum(all_returns)/len(all_returns))*100:.4f}%")
    print(f"  Return std    : {float(pd.Series(all_returns).std())*100:.2f}%")
    print(f"  Return skew   : {float(pd.Series(all_returns).skew()):.3f}")
    print(f"  Return kurt   : {float(pd.Series(all_returns).kurtosis()):.3f}")

    up   = sum(1 for r in all_returns if r > 0)
    down = sum(1 for r in all_returns if r < 0)
    print(f"  UP days       : {up} ({100*up/(up+down):.1f}%)")
    print(f"  DOWN days     : {down} ({100*down/(up+down):.1f}%)")
    print(f"{'='*60}\n")

    # Plot return distribution
    try:
        os.makedirs(plots_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('EDA — IDX Stock Universe', fontweight='bold')

        # Return distribution
        axes[0].hist(all_returns, bins=100, color='steelblue', alpha=0.7, edgecolor='none')
        axes[0].axvline(0, color='red', lw=1.5, linestyle='--')
        axes[0].set_title('Daily Return Distribution')
        axes[0].set_xlabel('Return'); axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)

        # Rows per stock
        axes[1].bar(list(raw_dfs.keys()), all_rows, color='steelblue', alpha=0.8)
        axes[1].set_title('Rows per Stock')
        axes[1].set_xlabel('Ticker'); axes[1].set_ylabel('Rows')
        axes[1].tick_params(axis='x', rotation=90)
        axes[1].grid(True, alpha=0.3, axis='y')

        # UP vs DOWN pie
        axes[2].pie([up, down], labels=['UP', 'DOWN'],
                    colors=['#2ca02c', '#d62728'],
                    autopct='%1.1f%%', startangle=90)
        axes[2].set_title('UP vs DOWN Days (all stocks)')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'eda_summary.png'), dpi=150)
        plt.close()
        print(f"[EDA] Plot saved → {plots_dir}/eda_summary.png")
    except Exception as e:
        print(f"[WARN] EDA plot failed: {e}")