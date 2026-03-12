"""
train.py — Main entry point for LNN Stock Prediction (TENCON 2026)

Usage:
  python train.py                        # train all models (default)
  python train.py --mode train           # train LNN ensemble + baselines
  python train.py --mode evaluate        # evaluate saved models only
  python train.py --mode compare         # generate comparison plots
  python train.py --epochs 50            # override epochs
"""

import os, sys, json, pickle, random, argparse, warnings
import numpy as np
import torch
warnings.filterwarnings('ignore')

# ── Project imports ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import CONFIG, IDX_UNIVERSE, MACRO_TICKERS
from utils.data_loader import load_ticker, load_macro, add_technical_features, explore_dataset
from utils.data_loader  import add_sector_momentum, merge_macro, clip_features
from utils.features     import select_features_mi, remove_correlated, fit_scaler
from utils.dataset      import StockDataset, make_loaders
from utils.trainer      import train_model, ensemble_predict
from utils.metrics      import compute_metrics, stat_tests, numpy_safe
from utils.plotter      import (plot_training_curves, plot_model_comparison,
                                plot_sector_breakdown, plot_confidence_analysis)
from models.lnn         import LNNClassifier
from models.baselines   import LSTMBaseline, GRUBaseline, TransformerBaseline


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────
# STEP 1: LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────────

def load_all_data(cfg: dict):
    """Load, clean, feature-engineer all stocks. Returns split dicts."""

    tickers = list(IDX_UNIVERSE.keys())
    period  = cfg['period']

    # ── Load raw stock data ───────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  STEP 1 — Loading {len(tickers)} IDX stocks ({period})")
    print(f"{'='*60}")
    raw_dfs = {}
    for t in tickers:
        df = load_ticker(t, period)
        if df is not None:
            raw_dfs[t] = df
    print(f"[DATA] Loaded {len(raw_dfs)}/{len(tickers)} stocks")

    try:
        explore_dataset(raw_dfs, cfg['plots_dir'])
    except Exception as e:
        print(f"[WARN] EDA failed: {e}")

    if not raw_dfs:
        raise RuntimeError("No stocks loaded — check internet connection.")

    # ── Load macro data ───────────────────────────────────────
    macro_df = load_macro(MACRO_TICKERS, period)

    # ── Add technical features ────────────────────────────────
    print("\n[FEAT] Computing technical indicators...")
    feat_dfs = {}
    for t, df in raw_dfs.items():
        try:
            df_feat = add_technical_features(df)

            # Add sector momentum (peers in same sector)
            sector = IDX_UNIVERSE[t][1]
            peers  = [raw_dfs[p] for p in raw_dfs
                      if IDX_UNIVERSE.get(p, ('', ''))[1] == sector and p != t]
            df_feat = add_sector_momentum(df_feat, peers)

            # Merge macro (left join)
            df_feat = merge_macro(df_feat, macro_df)

            # Clip known features to valid ranges
            df_feat = clip_features(df_feat)

            # Drop NaN rows (from rolling indicators)
            df_feat = df_feat.dropna()

            if len(df_feat) > cfg['seq_len'] + 10:
                feat_dfs[t] = df_feat
        except Exception as e:
            print(f"  [WARN] Feature engineering failed for {t}: {e}")

    print(f"[FEAT] {len(feat_dfs)} stocks with full feature sets")
    if not feat_dfs:
        raise RuntimeError("Feature engineering failed for all stocks.")

    # ── Chronological split BEFORE any feature selection ─────
    print("\n[DATA] Splitting train/val/test (chronological)...")
    train_dfs, val_dfs, test_dfs = {}, {}, {}
    tr_r  = cfg['train_ratio']
    va_r  = cfg['val_ratio']

    for t, df in feat_dfs.items():
        n      = len(df)
        n_tr   = int(n * tr_r)
        n_va   = int(n * va_r)
        train_dfs[t] = df.iloc[:n_tr]
        val_dfs[t]   = df.iloc[n_tr: n_tr + n_va]
        test_dfs[t]  = df.iloc[n_tr + n_va:]

    # ── All columns from training data ────────────────────────
    all_cols = sorted(set.intersection(*[set(df.columns) for df in train_dfs.values()]))
    all_cols = [c for c in all_cols if c != 'Open']

    # ── Feature selection (MI) — on training data only ───────
    print("\n[FEAT] Selecting features...")
    selected = select_features_mi(train_dfs, all_cols,
                                  n_select=cfg['n_features_mi'])
    selected = remove_correlated(selected, train_dfs, cfg['corr_threshold'])
    print(f"[FEAT] Final: {len(selected)} features")

    # ── Fit scaler on training data only ─────────────────────
    print("[DATA] Fitting RobustScaler on training data...")
    scaler = fit_scaler(train_dfs, selected)

    return train_dfs, val_dfs, test_dfs, selected, scaler


# ─────────────────────────────────────────────────────────────
# STEP 2: TRAIN
# ─────────────────────────────────────────────────────────────

def run_training(cfg: dict):
    """Full training pipeline — LNN ensemble + baselines."""

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] Using: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ─────────────────────────────────────────────
    train_dfs, val_dfs, test_dfs, feature_cols, scaler = load_all_data(cfg)
    n_features = len(feature_cols)
    print(f"\n[DATA] n_features={n_features}, seq_len={cfg['seq_len']}")

    # ── Build DataLoaders ─────────────────────────────────────
    print("\n[DATA] Building DataLoaders...")
    train_loader, val_loader, per_ticker_test = make_loaders(
        train_dfs, val_dfs, test_dfs, feature_cols, scaler, cfg
    )

    # ── Save scaler and feature cols (needed for evaluate mode) ──
    os.makedirs(cfg['save_dir'], exist_ok=True)
    with open(os.path.join(cfg['save_dir'], 'shared_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(cfg['save_dir'], 'feature_cols.json'), 'w') as f:
        json.dump(feature_cols, f)
    print(f"[SAVE] Scaler + feature cols → {cfg['save_dir']}/")

    all_results  = {}
    all_histories = {}

    # ════════════════════════════════════════════════════════
    # TRAIN LNN ENSEMBLE
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 2 — LNN Ensemble (seeds: {cfg['ensemble_seeds']})")
    print(f"{'='*60}")

    lnn_models = []
    for seed in cfg['ensemble_seeds']:
        set_seed(seed)
        model = LNNClassifier(
            n_features  = n_features,
            seq_len     = cfg['seq_len'],
            inter       = cfg['inter_neurons'],
            command     = cfg['command_neurons'],
            motor       = cfg['motor_neurons'],
            n_layers    = cfg['num_layers'],
            dropout     = cfg['dropout'],
            ode_unfolds = cfg['ode_unfolds']
        )
        best_state, history = train_model(
            model, train_loader, val_loader, cfg, device, f'lnn_seed{seed}'
        )
        all_histories[f'lnn_seed{seed}'] = history
        try:
            plot_training_curves(history, f'lnn_seed{seed}', cfg['plots_dir'])
        except Exception:
            pass

        if best_state is not None:
            model.load_state_dict(best_state)
            save_path = os.path.join(cfg['save_dir'], f'lnn_seed{seed}_best.pth')
            torch.save({'model_state_dict': best_state,
                        'val_da': max(history['val_da']),
                        'config': cfg}, save_path)
            lnn_models.append(model.to(device))
            print(f"  [SAVE] {save_path}")

    # ════════════════════════════════════════════════════════
    # EVALUATE LNN ENSEMBLE
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 3 — Evaluating LNN Ensemble")
    print(f"{'='*60}")

    lnn_das    = []
    sector_das = {}
    conf_analysis = {thresh: {'DA': [], 'coverage': []} for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]}

    for t, ds in per_ticker_test.items():
        try:
            probs, labels, conf_mask = ensemble_predict(
                lnn_models, ds, device, cfg['confidence_thresh']
            )
            if len(probs) == 0:
                continue
            m = compute_metrics(probs, labels, conf_mask)
            lnn_das.append(m['DA'])
            sector = IDX_UNIVERSE.get(t, ('', 'Unknown'))[1]
            sector_das.setdefault(sector, []).append(m['DA'])
            for thresh in conf_analysis:
                cm = (probs > thresh) | (probs < (1 - thresh))
                if cm.sum() > 5:
                    da_c = float(np.mean((probs[cm] > 0.5).astype(int) == labels[cm]) * 100)
                    conf_analysis[thresh]['DA'].append(da_c)
                    conf_analysis[thresh]['coverage'].append(float(cm.mean() * 100))
            print(f"  {t:<12} DA={m['DA']:.1f}%  conf_DA={m['DA_conf']:.1f}%  "
                  f"cov={m['coverage']:.0f}%")
        except Exception as e:
            print(f"  [WARN] Eval failed for {t}: {e}")

    mean_lnn_da = float(np.mean(lnn_das)) if lnn_das else 0.0
    std_lnn_da  = float(np.std(lnn_das))  if lnn_das else 0.0

    all_results['LNN_Ensemble'] = {
        'DA':     mean_lnn_da,
        'DA_std': std_lnn_da,
        'DA_per_ticker': {t: float(da) for t, da in zip(per_ticker_test.keys(), lnn_das)},
    }
    all_results['_sector_breakdown']   = {s: [float(v) for v in vs] for s, vs in sector_das.items()}
    all_results['_confidence_analysis'] = {str(k): v for k, v in conf_analysis.items()}

    print(f"\n  ✓ LNN Ensemble Test DA: {mean_lnn_da:.2f}% ± {std_lnn_da:.2f}%")

    # ════════════════════════════════════════════════════════
    # TRAIN BASELINES
    # ════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  STEP 4 — Training Baselines")
    print(f"{'='*60}")

    baselines = {
        'LSTM':        LSTMBaseline(n_features),
        'GRU':         GRUBaseline(n_features),
        'Transformer': TransformerBaseline(n_features),
    }

    stat_results = {}

    for bname, bmodel in baselines.items():
        try:
            best_state, history = train_model(
                bmodel, train_loader, val_loader, cfg, device, bname
            )
            all_histories[bname] = history
            try:
                plot_training_curves(history, bname, cfg['plots_dir'])
            except Exception:
                pass

            if best_state is None:
                continue

            bmodel.load_state_dict(best_state)
            bmodel.to(device).eval()

            # Save
            bpath = os.path.join(cfg['save_dir'], f'{bname.lower()}_best.pth')
            torch.save({'model_state_dict': best_state, 'val_da': max(history['val_da'])}, bpath)

            # Evaluate per ticker
            b_das = []
            for t, ds in per_ticker_test.items():
                try:
                    probs, labels, conf_mask = ensemble_predict(
                        [bmodel], ds, device, cfg['confidence_thresh']
                    )
                    if len(probs) > 0:
                        m = compute_metrics(probs, labels, conf_mask)
                        b_das.append(m['DA'])
                except Exception:
                    continue

            mean_da = float(np.mean(b_das)) if b_das else 0.0
            all_results[bname] = {
                'DA': mean_da,
                'DA_std': float(np.std(b_das)) if b_das else 0.0,
            }
            print(f"  ✓ {bname} Test DA: {mean_da:.2f}%")

            # Statistical test vs LNN
            if lnn_das and b_das and len(lnn_das) == len(b_das):
                stat_results[bname] = stat_tests(lnn_das, b_das, bname)

        except Exception as e:
            print(f"  [WARN] Baseline {bname} failed: {e}")

    # ── SAVE RESULTS ──────────────────────────────────────────
    os.makedirs(cfg['results_dir'], exist_ok=True)
    rpath = os.path.join(cfg['results_dir'], 'results_v5.json')
    spath = os.path.join(cfg['results_dir'], 'stat_tests.json')

    with open(rpath, 'w') as f:
        json.dump(numpy_safe(all_results), f, indent=2)
    with open(spath, 'w') as f:
        json.dump(numpy_safe(stat_results), f, indent=2)
    print(f"\n[SAVED] Results → {rpath}")

    # ── PLOTS ─────────────────────────────────────────────────
    try:
        plot_model_comparison(all_results, stat_results, cfg['plots_dir'])
    except Exception as e:
        print(f"  [WARN] Comparison plot failed: {e}")
    try:
        plot_sector_breakdown(sector_das, cfg['plots_dir'])
    except Exception as e:
        print(f"  [WARN] Sector plot failed: {e}")
    try:
        plot_confidence_analysis(conf_analysis, cfg['plots_dir'])
    except Exception as e:
        print(f"  [WARN] Confidence plot failed: {e}")

    # ── FINAL SUMMARY ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL RESULTS — TENCON 2026")
    print(f"{'='*60}")
    print(f"  {'Model':<16} {'Test DA':>8}  {'Std':>6}")
    print(f"  {'─'*35}")
    for m, r in sorted(all_results.items(), key=lambda x: x[1].get('DA', 0)
                        if isinstance(x[1], dict) else 0, reverse=True):
        if m.startswith('_'): continue
        print(f"  {m:<16} {r.get('DA', 0):>7.2f}%  ±{r.get('DA_std', 0):.2f}%")
    print(f"  {'─'*35}")
    print(f"  Random baseline:  50.00%")
    print(f"\n[DONE] Training complete! Results → {cfg['results_dir']}/")


# ─────────────────────────────────────────────────────────────
# EVALUATE MODE — load saved models, skip training
# ─────────────────────────────────────────────────────────────

def run_evaluate(cfg: dict):
    """Load saved models and run evaluation only."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\n  EVALUATE MODE (loading saved models)\n{'='*60}")

    # Check artifacts
    sca_path  = os.path.join(cfg['save_dir'], 'shared_scaler.pkl')
    feat_path = os.path.join(cfg['save_dir'], 'feature_cols.json')
    if not os.path.exists(sca_path) or not os.path.exists(feat_path):
        print("[ERROR] Run --mode train first."); return

    with open(sca_path,  'rb') as f: scaler       = pickle.load(f)
    with open(feat_path, 'r')  as f: feature_cols = json.load(f)
    n_features = len(feature_cols)
    print(f"[OK] Loaded scaler + {n_features} features")

    # Rebuild test data
    train_dfs, val_dfs, test_dfs, _, _ = load_all_data(cfg)
    _, _, per_ticker_test = make_loaders(
        train_dfs, val_dfs, test_dfs, feature_cols, scaler, cfg
    )

    # Load LNN ensemble
    lnn_models = []
    for seed in cfg['ensemble_seeds']:
        path = os.path.join(cfg['save_dir'], f'lnn_seed{seed}_best.pth')
        if not os.path.exists(path):
            print(f"[WARN] Missing {path}"); continue
        m = LNNClassifier(n_features, cfg['seq_len'],
                          cfg['inter_neurons'], cfg['command_neurons'],
                          cfg['motor_neurons'], cfg['num_layers'],
                          cfg['dropout'], cfg['ode_unfolds'])
        ckpt = torch.load(path, map_location=device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.to(device).eval()
        lnn_models.append(m)
        print(f"[OK] Loaded lnn_seed{seed}")

    # Evaluate
    lnn_das = []
    for t, ds in per_ticker_test.items():
        try:
            probs, labels, conf_mask = ensemble_predict(
                lnn_models, ds, device, cfg['confidence_thresh']
            )
            if len(probs) > 0:
                m = compute_metrics(probs, labels, conf_mask)
                lnn_das.append(m['DA'])
                print(f"  {t:<12} DA={m['DA']:.1f}%")
        except Exception as e:
            print(f"  [WARN] {t}: {e}")

    if lnn_das:
        print(f"\n  LNN Ensemble Test DA: {float(np.mean(lnn_das)):.2f}% ± {float(np.std(lnn_das)):.2f}%")
    else:
        print("\n  [WARN] No tickers evaluated successfully.")


# ─────────────────────────────────────────────────────────────
# COMPARE MODE — generate plots from saved results JSON
# ─────────────────────────────────────────────────────────────

def run_compare(cfg: dict):
    rpath = os.path.join(cfg['results_dir'], 'results_v5.json')
    spath = os.path.join(cfg['results_dir'], 'stat_tests.json')
    if not os.path.exists(rpath):
        print(f"[ERROR] {rpath} not found — run --mode train first"); return
    with open(rpath) as f: results   = json.load(f)
    with open(spath) as f: stat_res  = json.load(f) if os.path.exists(spath) else {}
    plot_model_comparison(results, stat_res, cfg['plots_dir'])
    sector_das = results.get('_sector_breakdown', {})
    if sector_das:
        plot_sector_breakdown(sector_das, cfg['plots_dir'])
    print("[DONE] Plots saved →", cfg['plots_dir'])


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LNN Stock v5 — TENCON 2026')
    parser.add_argument('--mode',   type=str, default='train',
                        choices=['train', 'evaluate', 'compare'])
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.epochs is not None:
        cfg['epochs'] = args.epochs

    if   args.mode == 'train':    run_training(cfg)
    elif args.mode == 'evaluate': run_evaluate(cfg)
    elif args.mode == 'compare':  run_compare(cfg)
