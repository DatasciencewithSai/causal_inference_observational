"""Minimal OLS with heteroskedasticity-robust (HC1) or cluster-robust standard errors."""

from __future__ import annotations

import numpy as np


def ols(X: np.ndarray, y: np.ndarray, clusters: np.ndarray | None = None):
    """Return (coefficients, standard errors). X must include an intercept column."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    if n <= k:  # exactly identified: no residual degrees of freedom
        return beta, np.full(k, np.nan)
    if clusters is None:
        meat = (X * resid[:, None]).T @ (X * resid[:, None])
        cov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    else:
        groups = np.unique(clusters)
        g = len(groups)
        meat = np.zeros((k, k))
        for c in groups:
            s = (X[clusters == c] * resid[clusters == c, None]).sum(axis=0)
            meat += np.outer(s, s)
        cov = xtx_inv @ meat @ xtx_inv * (g / (g - 1)) * ((n - 1) / (n - k))
    return beta, np.sqrt(np.diag(cov))
