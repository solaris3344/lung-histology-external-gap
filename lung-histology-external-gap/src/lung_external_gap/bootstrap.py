from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def bootstrap_auc(y, score, n_boot=2000, seed=2026):
    y = np.asarray(y)
    score = np.asarray(score)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        values.append(roc_auc_score(y[idx], score[idx]))
    est = roc_auc_score(y, score)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def bootstrap_paired_auc_difference(y, score_a, score_b, n_boot=2000, seed=2026):
    y = np.asarray(y)
    a = np.asarray(score_a)
    b = np.asarray(score_b)
    rng = np.random.default_rng(seed)
    values = []
    n = len(y)
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size < 2:
            continue
        values.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    est = roc_auc_score(y, a) - roc_auc_score(y, b)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(est), float(lo), float(hi)
