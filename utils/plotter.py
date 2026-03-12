"""
utils/plotter.py — All plots for the paper.
Every function is wrapped in try-except so plots never crash training.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from typing import Dict, List


def plot_training_curves(history: Dict, name: str, plots_dir: str):
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'Training Curves — {name}', fontweight='bold')

        axes[0].plot(history['train_loss'], label='Train', color='steelblue')
        axes[0].plot(history['val_loss'],   label='Val',   color='orange')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['val_da'], label='Val DA', color='green')
        axes[1].axhline(50, color='red', linestyle='--', lw=1, label='Random')
        axes[1].set_title('Validation DA (%)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        os.makedirs(plots_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'curves_{name.lower()}.png'), dpi=120)
        plt.close()
    except Exception as e:
        print(f"  [WARN] plot_training_curves failed: {e}")


def plot_model_comparison(results: Dict, stat_tests: Dict, plots_dir: str):
    try:
        model_keys = [k for k in results if not k.startswith('_')
                      and isinstance(results[k], dict) and 'DA' in results[k]]
        if not model_keys:
            return

        models = sorted(model_keys, key=lambda m: results[m]['DA'], reverse=True)
        das    = [results[m]['DA']           for m in models]
        f1s    = [results[m].get('F1', 0.5) * 100 for m in models]
        aucs   = [results[m].get('AUC', 0.5) * 100 for m in models]
        colors = ['#1f77b4' if 'LNN' in m else '#d62728' for m in models]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Model Comparison — IDX Stock Universe\nTENCON 2026, Bali',
                     fontsize=13, fontweight='bold')

        x = np.arange(len(models)); w = 0.25
        axes[0].bar(x - w, das, w, color=colors, alpha=0.9,  label='DA (%)')
        axes[0].bar(x,     f1s, w, color=colors, alpha=0.6,  label='F1×100', hatch='///')
        axes[0].bar(x + w, aucs,w, color=colors, alpha=0.4,  label='AUC×100', hatch='...')
        axes[0].axhline(50, color='black', linestyle='--', lw=1, label='Random')
        axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=15, ha='right')
        axes[0].set_ylim([40, 80]); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].set_title('Performance Metrics')
        for i, (bar_x, da) in enumerate(zip(x - w, das)):
            axes[0].text(bar_x, da + 0.3, f'{da:.1f}', ha='center', fontsize=8, fontweight='bold')

        # p-value bars
        baselines = [m for m in models if 'LNN' not in m]
        pvals = [stat_tests.get(m, {}).get('t_pval', 1.0) for m in baselines]
        colors_p = ['#2ca02c' if p < 0.05 else '#aec7e8' for p in pvals]
        axes[1].barh(baselines, pvals, color=colors_p, alpha=0.8)
        axes[1].axvline(0.05, color='red', linestyle='--', lw=1.5, label='p=0.05')
        axes[1].set_title('LNN vs Baseline (t-test p-value)')
        axes[1].set_xlabel('p-value'); axes[1].legend()
        axes[1].grid(True, alpha=0.3, axis='x')

        os.makedirs(plots_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'model_comparison.png'), dpi=150)
        plt.close()
        print(f"[PLOT] model_comparison.png saved")
    except Exception as e:
        print(f"  [WARN] plot_model_comparison failed: {e}")


def plot_sector_breakdown(sector_das: Dict[str, List[float]], plots_dir: str):
    try:
        if not sector_das:
            return
        sectors = [s for s in sector_das if sector_das[s]]
        means   = [float(np.mean(sector_das[s])) for s in sectors]
        order   = np.argsort(means)[::-1]
        sectors = [sectors[i] for i in order]
        means   = [means[i]   for i in order]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(sectors, means, color='steelblue', alpha=0.85)
        ax.axvline(50, color='red', linestyle='--', lw=1.5, label='Random (50%)')
        ax.set_title('LNN Ensemble DA by Sector — TENCON 2026', fontweight='bold')
        ax.set_xlabel('Directional Accuracy (%)'); ax.set_xlim([40, 75])
        ax.legend(); ax.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, means):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    f'{val:.1f}%', va='center', fontsize=9)
        os.makedirs(plots_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'sector_breakdown.png'), dpi=150)
        plt.close()
        print(f"[PLOT] sector_breakdown.png saved")
    except Exception as e:
        print(f"  [WARN] plot_sector_breakdown failed: {e}")


def plot_confidence_analysis(conf_analysis: Dict, plots_dir: str):
    try:
        thresholds = sorted(conf_analysis.keys())
        das        = [float(np.mean(conf_analysis[t]['DA']))       if conf_analysis[t]['DA'] else 50.0
                      for t in thresholds]
        coverages  = [float(np.mean(conf_analysis[t]['coverage'])) if conf_analysis[t]['coverage'] else 100.0
                      for t in thresholds]

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        ax1.plot(thresholds, das,       'bo-', lw=2, label='DA (%)')
        ax2.plot(thresholds, coverages, 'rs--', lw=2, label='Coverage (%)')
        ax1.axhline(50, color='gray', linestyle=':', lw=1)
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Directional Accuracy (%)', color='blue')
        ax2.set_ylabel('Coverage (%)', color='red')
        ax1.set_title('Confidence Threshold Analysis', fontweight='bold')
        fig.legend(loc='lower right', bbox_to_anchor=(0.88, 0.15))
        ax1.grid(True, alpha=0.3)
        os.makedirs(plots_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'confidence_analysis.png'), dpi=150)
        plt.close()
        print(f"[PLOT] confidence_analysis.png saved")
    except Exception as e:
        print(f"  [WARN] plot_confidence_analysis failed: {e}")
