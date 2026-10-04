from __future__ import annotations

import numpy as np


class NuisanceCVA:
    """Nuisance-CVA from the manuscript, fitted on training data only.

    Parameters
    ----------
    n_min:
        Minimum class-by-group count required for a group mean.
    mode:
        "hard" removes the estimated nuisance subspace.
        "soft" uses lambda/(lambda+alpha) weights.
    alpha_fraction:
        Soft-mode alpha = alpha_fraction * mean(nonzero lambda).
    eps:
        Numerical tolerance.
    """

    def __init__(self, n_min: int = 3, mode: str = "hard",
                 alpha_fraction: float = 0.01, eps: float = 1e-12):
        if mode not in {"hard", "soft"}:
            raise ValueError("mode must be 'hard' or 'soft'")
        self.n_min = int(n_min)
        self.mode = mode
        self.alpha_fraction = float(alpha_fraction)
        self.eps = float(eps)

    def fit(self, X, y, groups):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        groups = np.asarray(groups)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if not (len(X) == len(y) == len(groups)):
            raise ValueError("X, y and groups must have matching rows")

        classes = np.unique(y)
        group_values = np.unique(groups)

        # Accumulate numerator and count in manuscript Eq. (6).
        numerators = {g: np.zeros(X.shape[1], dtype=float) for g in group_values}
        counts = {g: 0 for g in group_values}

        used_classes = []
        for c in classes:
            eligible = []
            for g in group_values:
                mask = (y == c) & (groups == g)
                n = int(mask.sum())
                if n >= self.n_min:
                    eligible.append((g, n, X[mask].mean(axis=0)))
            if len(eligible) < 2:
                continue

            total_n = sum(n for _, n, _ in eligible)
            m_c = sum(n * m for _, n, m in eligible) / total_n
            used_classes.append(c)

            for g, n, m_cg in eligible:
                numerators[g] += n * (m_cg - m_c)
                counts[g] += n

        shifts = []
        shift_groups = []
        shift_counts = []
        for g in group_values:
            if counts[g] > 0:
                delta_g = numerators[g] / counts[g]
                shifts.append(np.sqrt(counts[g]) * delta_g)
                shift_groups.append(g)
                shift_counts.append(counts[g])

        if not shifts:
            self.V_ = np.zeros((X.shape[1], 0), dtype=float)
            self.weights_ = np.zeros(0, dtype=float)
            self.lambdas_ = np.zeros(0, dtype=float)
            self.P_ = np.eye(X.shape[1])
            self.groups_ = np.asarray([], dtype=groups.dtype)
            self.used_classes_ = np.asarray(used_classes)
            return self

        R = np.vstack(shifts)
        _, s, vt = np.linalg.svd(R, full_matrices=False)
        keep = s > self.eps
        s = s[keep]
        V = vt[keep].T

        n_total = max(int(np.sum(shift_counts)), 1)
        lambdas = (s ** 2) / n_total

        if self.mode == "hard":
            weights = np.ones_like(lambdas)
        else:
            alpha = self.alpha_fraction * float(np.mean(lambdas)) if len(lambdas) else 0.0
            weights = lambdas / (lambdas + alpha + self.eps)

        P = np.eye(X.shape[1]) - (V * weights[None, :]) @ V.T

        self.V_ = V
        self.weights_ = weights
        self.lambdas_ = lambdas
        self.P_ = P
        self.groups_ = np.asarray(shift_groups)
        self.used_classes_ = np.asarray(used_classes)
        return self

    def transform(self, X):
        if not hasattr(self, "P_"):
            raise RuntimeError("fit must be called first")
        X = np.asarray(X, dtype=float)
        return X @ self.P_.T

    def fit_transform(self, X, y, groups):
        return self.fit(X, y, groups).transform(X)


def class_retention_ratio(P, mu_scc, mu_adc, eps: float = 1e-12) -> float:
    """Manuscript Eq. (9): fraction of class mean-difference norm retained."""
    P = np.asarray(P, dtype=float)
    d = np.asarray(mu_scc, dtype=float) - np.asarray(mu_adc, dtype=float)
    return float(np.linalg.norm(P @ d) / max(np.linalg.norm(d), eps))
