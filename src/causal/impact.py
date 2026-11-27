"""CausalImpact-style analysis: a structural time-series counterfactual for one treated series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

_DIFFUSE = 1e6  # prior variance of the initial state, on the standardized scale
_GRID_OBS = 31  # grid points for log sigma_obs (coarse-pass mode +/- 1)
_GRID_LEVEL = 40  # grid points for log sigma_level (1e-4 .. 3, standardized scale)


@dataclass(frozen=True)
class CausalImpactResult:
    effect: float  # posterior mean of the average effect per post-period step
    se: float  # posterior SD of the average effect
    p_value: float  # two-sided posterior tail probability of the average effect vs 0
    ci_low: float
    ci_high: float
    cumulative: float  # summed effect over the post-period
    cumulative_ci_low: float
    cumulative_ci_high: float
    relative_effect: float  # effect / average counterfactual
    predicted: np.ndarray  # counterfactual for every step (pre: one-step-ahead, NaN while diffuse)
    predicted_low: np.ndarray
    predicted_high: np.ndarray
    pointwise: np.ndarray  # actual - predicted, every step
    cumulative_path: np.ndarray  # running sum of pointwise effects over the post-period
    cumulative_path_low: np.ndarray
    cumulative_path_high: np.ndarray
    beta: np.ndarray  # posterior mean regression weights on the controls (original scale)
    sigma_obs: float  # posterior mean observation-noise SD (original scale)
    sigma_level: float  # posterior mean random-walk SD of the level (original scale)


def _kalman(y, Z, s2_obs, s2_level):
    """Kalman filter for y_t = Z_t s_t + e_t, run for G variance pairs at once.

    The state is [level, weights...]: the level is a random walk, the weights are static,
    and both start diffuse. s2_obs and s2_level are arrays of shape (G,). Returns the
    log-likelihood (diffuse steps excluded) (G,), one-step predictions and variances
    (G, n), and the filtered state mean (G, m) and covariance (G, m, m) after the last step.
    """
    s2_obs, s2_level = np.atleast_1d(s2_obs), np.atleast_1d(s2_level)
    n, m = Z.shape
    g = len(s2_obs)
    a = np.zeros((g, m))
    P = np.broadcast_to(np.eye(m) * _DIFFUSE, (g, m, m)).copy()
    ll = np.zeros(g)
    pred, pvar = np.empty((g, n)), np.empty((g, n))
    for t in range(n):
        if t > 0:
            P[:, 0, 0] += s2_level
        z = Z[t]
        Pz = P @ z
        f = Pz @ z + s2_obs
        v = y[t] - a @ z
        pred[:, t], pvar[:, t] = a @ z, f
        if t >= m:  # the first m steps only resolve the diffuse prior
            ll -= 0.5 * (np.log(2 * np.pi * f) + v * v / f)
        k = Pz / f[:, None]
        a = a + k * v[:, None]
        P = P - k[:, :, None] * Pz[:, None, :]
        P = (P + P.transpose(0, 2, 1)) / 2
    return ll, pred, pvar, a, P


def _forecast(a, P, Zp, s2_obs, s2_level):
    """Gaussian forecast of the post-period for each variance pair.

    Returns pointwise means and variances (G, h) and the variance of the running sum of
    the first j forecasts (G, h).
    """
    h = len(Zp)
    j = np.arange(1, h + 1)
    mean = a @ Zp.T
    var = np.einsum("tm,gmk,tk->gt", Zp, P, Zp) + s2_level[:, None] * j + s2_obs[:, None]
    C = np.cumsum(Zp, axis=0)  # sum of the first j rows
    # level random walk: Var(sum_{t<=j} sum_{i<=t} eta_i) = sigma^2 * j(j+1)(2j+1)/6
    cvar = np.einsum("tm,gmk,tk->gt", C, P, C)
    cvar = cvar + s2_level[:, None] * j * (j + 1) * (2 * j + 1) / 6 + s2_obs[:, None] * j
    return mean, var, cvar


def _mixture_quantile(w, mu, sd, q):
    """Quantile q of sum_g w_g N(mu_g, sd_g^2), column by column (mu, sd: (G, J))."""
    lo = (mu - 10 * sd).min(axis=0)
    hi = (mu + 10 * sd).max(axis=0)
    for _ in range(80):
        mid = (lo + hi) / 2
        below = w @ stats.norm.cdf((mid - mu) / sd) < q
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return (lo + hi) / 2


def causal_impact(y, X=None, n_pre: int | None = None, alpha: float = 0.05) -> CausalImpactResult:
    """Estimate the effect of an intervention on one time series, CausalImpact style.

    y: outcome series for the treated unit (length n).
    X: optional (n, k) control series that were NOT affected by the intervention.
    n_pre: number of pre-intervention steps; steps n_pre..n-1 are the post-period.

    Model (Brodersen et al. 2015, "Inferring causal impact using Bayesian structural
    time-series models"): y_t = mu_t + x_t' beta + e_t with a random-walk level mu_t.
    It is fitted to the pre-period only and forecast over the post-period as the
    counterfactual; the effect is actual minus counterfactual.

    Inference is Bayesian but exact rather than MCMC: the regression weights have a flat
    prior and are integrated out by the Kalman filter, and the two noise SDs have uniform
    priors and are integrated out on a grid. Credible intervals are quantiles of the
    resulting mixture of Gaussians. Google's CausalImpact additionally uses a
    spike-and-slab prior to select controls and an informative prior on the level SD.

    Identifying assumptions: the controls are unaffected by the intervention; the
    relationship between y and the controls (and the level dynamics) that held before
    the intervention would have continued without it; no other shock hit y at the
    same time.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("y must be one-dimensional")
    n = len(y)
    if X is None:
        X = np.empty((n, 0))
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] != n:
        raise ValueError("X must have one row per element of y")
    if not (np.isfinite(y).all() and np.isfinite(X).all()):
        raise ValueError("y and X must be finite")
    k = X.shape[1]
    if n_pre is None or not (k + 3 <= n_pre < n):
        raise ValueError(f"n_pre must be in [{k + 3}, {n - 1}] (enough pre steps to fit)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    # Standardize on the pre-period for a well-conditioned fit; predictions are unaffected.
    my, sy = y[:n_pre].mean(), y[:n_pre].std()
    if sy == 0:
        raise ValueError("y is constant in the pre-period")
    mx, sx = X[:n_pre].mean(axis=0), X[:n_pre].std(axis=0)
    if (sx == 0).any():
        raise ValueError("a control series is constant in the pre-period")
    ys, Xs = (y - my) / sy, (X - mx) / sx
    Z = np.column_stack([np.ones(n), Xs])
    m = Z.shape[1]
    yp, Zpre, Zp = ys[:n_pre], Z[:n_pre], Z[n_pre:]

    # 1) Coarse grid pass to locate the observation SD (it is well identified).
    log_sl = np.linspace(np.log(1e-4), np.log(3.0), _GRID_LEVEL)

    def log_post(log_so):
        gso, gsl = (v.ravel() for v in np.meshgrid(log_so, log_sl))
        s2o, s2l = np.exp(2 * gso), np.exp(2 * gsl)
        # uniform prior on each SD -> Jacobian exp(log sd) on the log grid
        return gso, gsl, s2o, s2l, _kalman(yp, Zpre, s2o, s2l), gso + gsl

    gso, _, _, _, out, jac = log_post(np.linspace(np.log(1e-3), np.log(3.0), 25))
    centre = gso[np.argmax(out[0] + jac)]

    # 2) Posterior over (sd_obs, sd_level) on a fine log grid.
    gso, gsl, s2o, s2l, out, jac = log_post(centre + np.linspace(-1, 1, _GRID_OBS))
    ll, pred_pre, var_pre, a, P = out
    logw = ll + jac
    w = np.exp(logw - logw.max())
    keep = w > 1e-10 * w.sum()
    w = w[keep] / w[keep].sum()
    s2o, s2l, a, P = s2o[keep], s2l[keep], a[keep], P[keep]
    pred_pre, var_pre = pred_pre[keep], var_pre[keep]

    # 3) Mixture forecast of the post-period, back on the original scale.
    mean, var, cvar = _forecast(a, P, Zp, s2o, s2l)
    h = n - n_pre
    mu = np.concatenate([pred_pre, mean], axis=1) * sy + my
    sd = np.sqrt(np.concatenate([var_pre, var], axis=1)) * sy
    lo_q, hi_q = alpha / 2, 1 - alpha / 2

    predicted = w @ mu
    predicted_low = _mixture_quantile(w, mu, sd, lo_q)
    predicted_high = _mixture_quantile(w, mu, sd, hi_q)
    for arr in (predicted, predicted_low, predicted_high):
        arr[:m] = np.nan  # one-step predictions are meaningless while the prior is diffuse

    cum_mu = np.cumsum(y[n_pre:]) - np.cumsum(mean * sy + my, axis=1)  # (G, h)
    cum_sd = np.sqrt(cvar) * sy
    cum_path = w @ cum_mu
    cum_low = _mixture_quantile(w, cum_mu, cum_sd, lo_q)
    cum_high = _mixture_quantile(w, cum_mu, cum_sd, hi_q)

    eff_mu, eff_sd = cum_mu[:, -1:] / h, cum_sd[:, -1:] / h
    effect = cum_path[-1] / h
    se = np.sqrt(w @ (eff_sd[:, 0] ** 2 + eff_mu[:, 0] ** 2) - effect**2)
    below0 = w @ stats.norm.cdf(-eff_mu[:, 0] / eff_sd[:, 0])
    p = min(1.0, 2 * min(below0, 1 - below0))
    counterfactual = (w @ mean).mean() * sy + my
    return CausalImpactResult(
        effect=float(effect),
        se=float(se),
        p_value=float(p),
        ci_low=float(cum_low[-1] / h),
        ci_high=float(cum_high[-1] / h),
        cumulative=float(cum_path[-1]),
        cumulative_ci_low=float(cum_low[-1]),
        cumulative_ci_high=float(cum_high[-1]),
        relative_effect=float(effect / counterfactual),
        predicted=predicted,
        predicted_low=predicted_low,
        predicted_high=predicted_high,
        pointwise=y - predicted,
        cumulative_path=cum_path,
        cumulative_path_low=cum_low,
        cumulative_path_high=cum_high,
        beta=(w @ a[:, 1:]) * sy / sx,
        sigma_obs=float(w @ np.sqrt(s2o) * sy),
        sigma_level=float(w @ np.sqrt(s2l) * sy),
    )
