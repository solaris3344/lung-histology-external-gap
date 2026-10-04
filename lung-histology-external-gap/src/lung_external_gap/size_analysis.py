from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def fit_size_only(log_diameter, y, C=0.1):
    x = np.asarray(log_diameter, dtype=float).reshape(-1, 1)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    model.fit(scaler.transform(x), y)
    return scaler, model


def fit_deep_plus_size(deep_features, log_diameter, y, C=0.1):
    X = np.column_stack([np.asarray(deep_features, dtype=float),
                         np.asarray(log_diameter, dtype=float)])
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
    model.fit(scaler.transform(X), y)
    return scaler, model


def score_by_size_interaction(deep_score, diameter, y):
    """Likelihood-ratio test for manuscript Eq. (10)."""
    s = StandardScaler().fit_transform(np.asarray(deep_score).reshape(-1, 1)).ravel()
    d = StandardScaler().fit_transform(np.log(np.asarray(diameter)).reshape(-1, 1)).ravel()
    y = np.asarray(y)

    X0 = np.column_stack([s, d])
    X1 = np.column_stack([s, d, s * d])

    m0 = LogisticRegression(C=1e6, max_iter=10000).fit(X0, y)
    m1 = LogisticRegression(C=1e6, max_iter=10000).fit(X1, y)

    def ll(model, X):
        p = np.clip(model.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return np.sum(y * np.log(p) + (1-y) * np.log(1-p))

    stat = 2.0 * (ll(m1, X1) - ll(m0, X0))
    p = chi2.sf(stat, 1)
    return float(stat), float(p)


def tertile_auc(score, diameter, y, cut_points=None):
    score = np.asarray(score)
    diameter = np.asarray(diameter)
    y = np.asarray(y)
    if cut_points is None:
        q1, q2 = np.quantile(diameter, [1/3, 2/3])
    else:
        q1, q2 = cut_points
    bins = np.digitize(diameter, [q1, q2], right=True)
    out = []
    for b in range(3):
        m = bins == b
        auc = np.nan
        if m.sum() and np.unique(y[m]).size == 2:
            auc = roc_auc_score(y[m], score[m])
        out.append((b, int(m.sum()), float(auc) if np.isfinite(auc) else np.nan))
    return out
