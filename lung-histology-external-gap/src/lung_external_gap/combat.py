from __future__ import annotations

import numpy as np


class LocationScaleComBat:
    """Study-specific location-scale ComBat-style harmonizer.

    This mirrors the manuscript's non-empirical-Bayes location-scale formulation.
    Histology can be protected during fitting. Unseen groups can be aligned
    transductively using their aggregate feature statistics.

    Notes
    -----
    The unseen-group transformation is a sensitivity analysis, not a prospective
    deployment procedure, because it uses aggregate external test-set statistics.
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def fit(self, X, y, groups):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        groups = np.asarray(groups)
        if not (len(X) == len(y) == len(groups)):
            raise ValueError("row counts differ")

        group_levels = np.unique(groups)
        # Intercept + protected histology covariate.
        D = np.hstack([np.ones((len(X), 1)), y])
        coef, *_ = np.linalg.lstsq(D, X, rcond=None)
        biological = D @ coef
        residual = X - biological

        gamma = {}
        delta = {}
        for g in group_levels:
            m = groups == g
            rg = residual[m]
            gamma[g] = rg.mean(axis=0)
            delta[g] = rg.std(axis=0, ddof=1) if rg.shape[0] > 1 else np.ones(X.shape[1])
            delta[g] = np.maximum(delta[g], self.eps)

        adjusted_residuals = []
        for i, g in enumerate(groups):
            adjusted_residuals.append((residual[i] - gamma[g]) / delta[g])
        adjusted_residuals = np.vstack(adjusted_residuals)

        pooled_sigma = np.maximum(adjusted_residuals.std(axis=0, ddof=1), self.eps)

        self.alpha_ = coef[0]
        self.beta_ = coef[1]
        self.gamma_ = gamma
        self.delta_ = delta
        self.group_levels_ = group_levels
        self.pooled_sigma_ = pooled_sigma

        # Harmonized training statistics used by the transductive unseen-group map.
        Xh = self.transform_training(X, y.ravel(), groups)
        self.harmonized_mean_ = Xh.mean(axis=0)
        self.harmonized_std_ = np.maximum(Xh.std(axis=0, ddof=1), self.eps)
        return self

    def transform_training(self, X, y, groups):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        out = np.empty_like(X)
        for i, g in enumerate(groups):
            bio = self.alpha_ + self.beta_ * y[i]
            r = X[i] - bio
            out[i] = self.pooled_sigma_ * ((r - self.gamma_[g]) / self.delta_[g]) + bio
        return out

    def transform_seen_group_without_label(self, X, groups):
        """Prospective-style seen-group transform without using test labels."""
        X = np.asarray(X, dtype=float)
        groups = np.asarray(groups)
        out = np.empty_like(X)
        for i, g in enumerate(groups):
            if g not in self.gamma_:
                raise KeyError(f"unseen group: {g}")
            r = X[i] - self.alpha_
            out[i] = self.pooled_sigma_ * ((r - self.gamma_[g]) / self.delta_[g]) + self.alpha_
        return out

    def transform_unseen_transductive(self, X):
        """Manuscript Eq. (4), applied to one unseen external group."""
        X = np.asarray(X, dtype=float)
        mu_ext = X.mean(axis=0)
        sd_ext = np.maximum(X.std(axis=0, ddof=1), self.eps)
        return self.harmonized_std_ * ((X - mu_ext) / sd_ext) + self.harmonized_mean_
