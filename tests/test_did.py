import numpy as np
import pytest

from causal import diff_in_diff


def _panel(effect, n_units=400, seed=0):
    rng = np.random.default_rng(seed)
    unit = np.repeat(np.arange(n_units), 2)
    post = np.tile([0, 1], n_units)
    treated = (unit < n_units // 2).astype(int)
    unit_fe = rng.normal(0, 2, n_units)[unit]
    y = 5 + 1.5 * treated + 2.0 * post + effect * treated * post + unit_fe
    y = y + rng.normal(0, 1, len(y))
    return y, treated, post, unit


def test_exact_cell_means():
    # Noise-free: ATT equals the double difference of cell means.
    y = np.array([1.0, 2.0, 3.0, 7.0])
    r = diff_in_diff(y, treated=[0, 0, 1, 1], post=[0, 1, 0, 1], clusters=None)
    assert np.isclose(r.att, (7 - 3) - (2 - 1))


def test_recovers_effect_with_clusters():
    y, d, t, unit = _panel(effect=3.0)
    r = diff_in_diff(y, d, t, clusters=unit)
    assert r.ci_low < 3.0 < r.ci_high
    assert r.p_value < 0.001


def test_null_effect_not_significant():
    y, d, t, unit = _panel(effect=0.0, seed=3)
    r = diff_in_diff(y, d, t, clusters=unit)
    assert r.ci_low < 0 < r.ci_high


def test_requires_all_cells():
    with pytest.raises(ValueError):
        diff_in_diff([1, 2, 3], [0, 0, 1], [0, 1, 0])
