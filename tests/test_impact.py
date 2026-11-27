import warnings

import numpy as np
import pytest

from causal import causal_impact
from causal.impact import _forecast, _kalman


def _series(effect, n=100, n_pre=70, seed=1):
    """Treated series driven by two trending controls plus a slowly drifting level."""
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(0.3, 1, (n, 2)), axis=0) + 50  # controls trend upward
    y = 10 + 0.8 * X[:, 0] + 0.3 * X[:, 1] + np.cumsum(rng.normal(0, 0.3, n))
    y = y + rng.normal(0, 1, n)
    y[n_pre:] += effect
    return y, X, n_pre


def test_filter_and_forecast_match_statsmodels():
    # Reference: statsmodels' local level + regression with the weights in the state.
    sm = pytest.importorskip("statsmodels.tsa.statespace.structural")
    y, X, n_pre = _series(effect=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.UnobservedComponents(
            y[:n_pre], level="llevel", exog=X[:n_pre], mle_regression=False
        )
        ref = model.fit(disp=False)
    s2o, s2l = np.array([ref.params[0]]), np.array([ref.params[1]])
    Z = np.column_stack([np.ones(len(y)), X])
    ll, _, _, a, P = _kalman(y[:n_pre], Z[:n_pre], s2o, s2l)
    assert np.isclose(ll[0], ref.llf, atol=1e-4)
    mean, var, cvar = _forecast(a, P, Z[n_pre:], s2o, s2l)
    f = ref.get_forecast(len(y) - n_pre, exog=X[n_pre:])
    assert np.allclose(mean[0], f.predicted_mean, atol=1e-3)
    assert np.allclose(var[0], f.var_pred_mean, rtol=1e-4)
    # Variance of running sums: compare with the full forecast covariance from simulation
    # of the same Gaussian (closed form vs explicit matrix).
    h = len(y) - n_pre
    L = np.tril(np.ones((h, h)))
    Zp = Z[n_pre:]
    cov = Zp @ P[0] @ Zp.T + s2l[0] * L @ L.T + s2o[0] * np.eye(h)
    assert np.allclose(cvar[0], np.cumsum(np.cumsum(cov, axis=0), axis=1).diagonal())


def test_intervals_are_calibrated():
    # 95% intervals for a null effect should cover 0 about 95% of the time.
    covered = [
        r.ci_low < 0 < r.ci_high for r in (causal_impact(*_series(0.0, seed=s)) for s in range(60))
    ]
    assert np.mean(covered) >= 0.88


def test_recovers_effect_where_naive_pre_post_is_biased():
    y, X, n_pre = _series(effect=5.0)
    r = causal_impact(y, X, n_pre)
    assert r.ci_low < 5.0 < r.ci_high
    assert r.p_value < 0.01
    # Controls trend upward, so a plain before/after comparison credits the trend too.
    naive = y[n_pre:].mean() - y[:n_pre].mean()
    assert naive > 10.0


def test_null_effect_not_significant():
    y, X, n_pre = _series(effect=0.0, seed=4)
    r = causal_impact(y, X, n_pre)
    assert r.ci_low < 0 < r.ci_high
    assert r.cumulative_ci_low < 0 < r.cumulative_ci_high


def test_internal_consistency():
    y, X, n_pre = _series(effect=3.0)
    r = causal_impact(y, X, n_pre)
    h = len(y) - n_pre
    assert np.isclose(r.cumulative, r.effect * h)
    assert np.isclose(r.cumulative_path[-1], r.cumulative)
    assert np.isclose(r.pointwise[n_pre:].sum(), r.cumulative)
    assert np.isclose(r.cumulative_ci_high - r.cumulative, (r.ci_high - r.effect) * h)
    assert np.isnan(r.predicted[:3]).all()  # diffuse start: intercept + 2 weights
    assert np.isfinite(r.predicted[3:]).all()
    widths = r.cumulative_path_high - r.cumulative_path_low
    assert (np.diff(widths) > 0).all()  # uncertainty accumulates


def test_level_only_without_controls():
    rng = np.random.default_rng(7)
    y = 100 + np.cumsum(rng.normal(0, 0.2, 80)) + rng.normal(0, 1, 80)
    y[60:] += 4
    r = causal_impact(y, n_pre=60)
    assert r.ci_low < 4.0 < r.ci_high
    assert r.beta.shape == (0,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_pre": None},
        {"n_pre": 2},
        {"n_pre": 100},
        {"alpha": 1.5, "n_pre": 70},
    ],
)
def test_rejects_bad_arguments(kwargs):
    y, X, _ = _series(effect=0.0)
    with pytest.raises(ValueError):
        causal_impact(y, X, **kwargs)


def test_rejects_bad_data():
    y, X, n_pre = _series(effect=0.0)
    with pytest.raises(ValueError):
        causal_impact(y, X[:-1], n_pre)
    bad = y.copy()
    bad[5] = np.nan
    with pytest.raises(ValueError):
        causal_impact(bad, X, n_pre)
    X2 = X.copy()
    X2[:n_pre, 0] = 1.0
    with pytest.raises(ValueError):
        causal_impact(y, X2, n_pre)
