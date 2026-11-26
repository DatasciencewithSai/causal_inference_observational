import numpy as np

from causal import ipw_ate, ipw_weights, propensity_scores, standardized_mean_differences


def _confounded(n=4000, effect=2.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    p = 1 / (1 + np.exp(-(0.8 * X[:, 0] - 0.5 * X[:, 1])))
    t = (rng.random(n) < p).astype(int)
    y = effect * t + 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(size=n)
    return X, t, y, p


def test_propensity_close_to_truth():
    X, t, _, p = _confounded()
    assert np.abs(propensity_scores(X, t) - p).mean() < 0.03


def test_ipw_removes_confounding_bias():
    X, t, y, _ = _confounded()
    naive = y[t == 1].mean() - y[t == 0].mean()
    r = ipw_ate(y, t, X=X, n_boot=200)
    assert abs(naive - 2.0) > 1.0  # naive comparison is badly biased
    assert r.ci_low < 2.0 < r.ci_high
    assert abs(r.ate - 2.0) < 0.3


def test_weighting_improves_balance():
    X, t, _, _ = _confounded()
    ps = propensity_scores(X, t)
    before = np.abs(standardized_mean_differences(X, t))
    after = np.abs(standardized_mean_differences(X, t, ipw_weights(ps, t)))
    assert (before > 0.2).any()
    assert (after < 0.1).all()
