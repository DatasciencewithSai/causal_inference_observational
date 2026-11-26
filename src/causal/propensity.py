"""Propensity score estimation, inverse probability weighting and balance checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit


def propensity_scores(X, treated) -> np.ndarray:
    """Fit a logistic regression of treatment on covariates X; return P(T=1 | X)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    t = np.asarray(treated, dtype=float)
    Z = np.column_stack([np.ones(len(X)), X])

    def nll(b):
        eta = Z @ b
        return np.sum(np.logaddexp(0, eta) - t * eta)

    def grad(b):
        return Z.T @ (expit(Z @ b) - t)

    res = minimize(nll, np.zeros(Z.shape[1]), jac=grad, method="BFGS")
    return expit(Z @ res.x)


@dataclass(frozen=True)
class IPWResult:
    ate: float
    se: float
    p_value: float
    ci_low: float
    ci_high: float
    n_trimmed: int


def ipw_ate(
    y,
    treated,
    X=None,
    ps=None,
    trim: float = 0.01,
    alpha: float = 0.05,
    n_boot: int = 500,
    seed: int | None = 0,
) -> IPWResult:
    """Average treatment effect by normalized (Hajek) inverse probability weighting.

    Pass covariates X (propensity is re-estimated in each bootstrap draw) or
    precomputed scores ps (bootstrap then treats them as fixed).
    Units with propensity outside [trim, 1 - trim] are dropped.
    Identifying assumptions: no unmeasured confounding and overlap.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(treated, dtype=float)
    if X is None and ps is None:
        raise ValueError("provide covariates X or propensity scores ps")
    Xa = None if X is None else np.asarray(X, dtype=float)

    def estimate(idx):
        p = propensity_scores(Xa[idx], t[idx]) if ps is None else np.asarray(ps)[idx]
        keep = (p >= trim) & (p <= 1 - trim)
        yy, tt, pp = y[idx][keep], t[idx][keep], p[keep]
        w1, w0 = tt / pp, (1 - tt) / (1 - pp)
        return np.sum(w1 * yy) / np.sum(w1) - np.sum(w0 * yy) / np.sum(w0), int((~keep).sum())

    n = len(y)
    ate, n_trimmed = estimate(np.arange(n))
    rng = np.random.default_rng(seed)
    boots = np.array([estimate(rng.integers(0, n, n))[0] for _ in range(n_boot)])
    se = float(boots.std(ddof=1))
    z = ate / se if se > 0 else 0.0
    crit = stats.norm.ppf(1 - alpha / 2)
    return IPWResult(
        ate, se, 2 * stats.norm.sf(abs(z)), ate - crit * se, ate + crit * se, n_trimmed
    )


def standardized_mean_differences(X, treated, weights=None) -> np.ndarray:
    """Standardized mean difference per covariate (|SMD| < 0.1 is the usual balance bar)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    t = np.asarray(treated).astype(bool)
    w = np.ones(len(X)) if weights is None else np.asarray(weights, dtype=float)
    m1 = np.average(X[t], axis=0, weights=w[t])
    m0 = np.average(X[~t], axis=0, weights=w[~t])
    pooled_sd = np.sqrt((X[t].var(axis=0, ddof=1) + X[~t].var(axis=0, ddof=1)) / 2)
    return (m1 - m0) / pooled_sd


def ipw_weights(ps, treated) -> np.ndarray:
    """ATE inverse probability weights: 1/ps for treated, 1/(1-ps) for control."""
    ps = np.asarray(ps, dtype=float)
    t = np.asarray(treated, dtype=float)
    return t / ps + (1 - t) / (1 - ps)
