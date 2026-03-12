"""
utils/metrics.py — Evaluation metrics for direction prediction.
All functions are safe on empty/degenerate inputs.
"""

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score, confusion_matrix)
from scipy.stats import ttest_rel, wilcoxon
from typing import Dict, List, Optional


def compute_metrics(probs: np.ndarray, labels: np.ndarray,
                    conf_mask: Optional[np.ndarray] = None) -> Dict:
    """Compute DA, F1, AUC and confidence-filtered DA."""
    if len(probs) == 0 or len(labels) == 0:
        return {'DA': 50.0, 'F1': 0.5, 'AUC': 0.5,
                'DA_conf': 50.0, 'coverage': 0.0}

    preds = (probs > 0.5).astype(int)

    try:
        da  = float(accuracy_score(labels, preds) * 100)
        f1  = float(f1_score(labels, preds, zero_division=0))
        auc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.5
    except Exception:
        da, f1, auc = 50.0, 0.5, 0.5

    da_conf  = da
    coverage = 100.0
    if conf_mask is not None and conf_mask.sum() > 5:
        try:
            da_conf  = float(accuracy_score(labels[conf_mask], preds[conf_mask]) * 100)
            coverage = float(conf_mask.mean() * 100)
        except Exception:
            pass

    return {
        'DA':       da,
        'F1':       f1,
        'AUC':      auc,
        'DA_conf':  da_conf,
        'coverage': coverage,
    }


def stat_tests(lnn_das: List[float],
               baseline_das: List[float],
               name: str) -> Dict:
    """Paired t-test and Wilcoxon signed-rank test."""
    result = {'model': name, 't_stat': 0.0, 't_pval': 1.0,
              'w_stat': 0.0, 'w_pval': 1.0, 'significant': False}

    a = np.array(lnn_das)
    b = np.array(baseline_das)

    if len(a) < 5 or len(b) < 5 or len(a) != len(b):
        return result

    try:
        t_stat, t_pval = ttest_rel(a, b)
        result['t_stat'] = float(t_stat)
        result['t_pval'] = float(t_pval)
    except Exception:
        pass

    try:
        if not np.all(a == b):
            w_stat, w_pval = wilcoxon(a, b)
            result['w_stat'] = float(w_stat)
            result['w_pval'] = float(w_pval)
    except Exception:
        pass

    result['significant'] = bool(result['t_pval'] < 0.05)
    return result


def numpy_safe(obj):
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (bool, np.bool_)):         return bool(obj)
    if isinstance(obj, np.integer):               return int(obj)
    if isinstance(obj, np.floating):              return float(obj)
    if isinstance(obj, np.ndarray):               return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_safe(i) for i in obj]
    return obj
