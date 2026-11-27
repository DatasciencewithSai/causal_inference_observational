"""Numbers for causal_impact.html: a simulated TV campaign analysed with causal.causal_impact.

Story: a TV campaign runs in Denver from day 71; Austin, Phoenix and Seattle get no campaign.
All four markets share weekly seasonality and growth. True effect: +30 sign-ups / day.

Run: .venv/bin/python visuals/causal_impact/impact_numbers.py
"""

import json

import numpy as np

from causal import causal_impact

rng = np.random.default_rng(20261169)
N, N_PRE, TRUE_EFFECT = 100, 70, 30.0
day = np.arange(1, N + 1)
weekly = np.array([0.0, 4, 6, 5, 2, -9, -8])[day % 7]  # weekend dip
trend = 0.9 * day  # growth shared by every market

controls = np.column_stack(
    [
        base + 0.5 * trend + 1.5 * weekly + np.cumsum(rng.normal(0, 1.5, N)) + rng.normal(0, 4, N)
        for base in (220.0, 160.0, 130.0)
    ]
)
X = controls
y = 40 + 0.6 * X[:, 0] + 0.5 * X[:, 1] + 0.4 * X[:, 2] + np.cumsum(rng.normal(0, 0.8, N))
y = y + rng.normal(0, 5, N)
y[N_PRE:] += TRUE_EFFECT

r = causal_impact(y, X, N_PRE)
naive = y[N_PRE:].mean() - y[:N_PRE].mean()
placebo = causal_impact(y[:N_PRE], X[:N_PRE], 50)  # fake launch at day 51, pre-period only

print(
    f"avg actual after {y[N_PRE:].mean():.1f}  avg counterfactual {r.predicted[N_PRE:].mean():.1f}"
)
print(f"naive before/after {naive:+.1f}  (avg before {y[:N_PRE].mean():.1f})")
print(f"effect {r.effect:+.1f}/day  95% CI [{r.ci_low:.1f}, {r.ci_high:.1f}]  p {r.p_value:.2g}")
print(f"cumulative {r.cumulative:+.0f}  CI [{r.cumulative_ci_low:.0f}, {r.cumulative_ci_high:.0f}]")
print(f"relative {r.relative_effect:+.1%}")
print(
    f"weights {np.round(r.beta, 2)}  sigma_obs {r.sigma_obs:.2f}  sigma_level {r.sigma_level:.2f}"
)
for name, c in zip(("Austin", "Phoenix", "Seattle"), X.T, strict=True):
    print(f"{name} (no campaign) before/after {c[N_PRE:].mean() / c[:N_PRE].mean() - 1:+.1%}")
print(f"Denver before/after {y[N_PRE:].mean() / y[:N_PRE].mean() - 1:+.1%}")
print(f"naive / effect {naive / r.effect:.1f}x")
print(f"placebo (fake launch day 51) {placebo.effect:+.1f}/day  p {placebo.p_value:.2f}")


def r1(a):
    return [None if not np.isfinite(v) else round(float(v), 1) for v in a]


print(
    json.dumps(
        {
            "y": r1(y),
            "X": [r1(c) for c in X.T],
            "pred": r1(r.predicted),
            "lo": r1(r.predicted_low),
            "hi": r1(r.predicted_high),
            "cum": r1(r.cumulative_path),
            "cum_lo": r1(r.cumulative_path_low),
            "cum_hi": r1(r.cumulative_path_high),
        }
    )
)
