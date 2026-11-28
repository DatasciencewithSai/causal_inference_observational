"""Numbers for causal_impact.html: simulated stories analysed with causal.causal_impact.

Each story simulates 100 days (launch on day 71) for one treated series and three control
series, runs causal_impact, checks that the rounded on-screen arithmetic adds up, and writes
<story>.data.js, which the page loads (?story=<name>). No number on screen is typed by hand.

Stories:
  tv       - a TV ad runs in Denver only; Austin, Phoenix and Seattle get no ad.
  feature  - one-tap checkout ships to every iOS user in an App Store release (no A/B test);
             Android, mobile web and desktop don't have it yet. A sitewide sale hits every
             platform on days 84-87.

Run: .venv/bin/python visuals/causal_impact/impact_numbers.py
"""

import json
import math
from pathlib import Path

import numpy as np

from causal import causal_impact

N, N_PRE = 100, 70
HERE = Path(__file__).parent


def simulate_tv():
    rng = np.random.default_rng(20261169)
    day = np.arange(1, N + 1)
    weekly = np.array([0.0, 4, 6, 5, 2, -9, -8])[day % 7]  # weekend dip
    trend = 0.9 * day  # growth shared by every market
    X = np.column_stack(
        [
            b + 0.5 * trend + 1.5 * weekly + np.cumsum(rng.normal(0, 1.5, N)) + rng.normal(0, 4, N)
            for b in (220.0, 160.0, 130.0)
        ]
    )
    y = 40 + 0.6 * X[:, 0] + 0.5 * X[:, 1] + 0.4 * X[:, 2] + np.cumsum(rng.normal(0, 0.8, N))
    y = y + rng.normal(0, 5, N)
    y[N_PRE:] += 30.0  # true effect
    return y, X, 30.0


def simulate_feature(seed=20261166):
    rng = np.random.default_rng(seed)
    day = np.arange(1, N + 1)
    weekly = np.array([0.0, -0.02, -0.03, -0.02, 0.01, 0.06, 0.07])[day % 7]  # weekend shopping
    growth = 0.0015 * day
    sale = np.where((day >= 84) & (day <= 87), 0.22, 0.0)  # sitewide sale, every platform
    common = 1 + growth + weekly + sale
    X = np.column_stack(
        [
            b * common + np.cumsum(rng.normal(0, 6, N)) + rng.normal(0, 18, N)
            for b in (2400.0, 1500.0, 1100.0)
        ]
    )
    y = 150 + 0.35 * X[:, 0] + 0.4 * X[:, 1] + 0.3 * X[:, 2] + np.cumsum(rng.normal(0, 3, N))
    y = y + rng.normal(0, 14, N)
    y[N_PRE:] += 95.0  # true effect
    return y, X, 95.0


STORIES = {
    "tv": dict(
        simulate=simulate_tv,
        dp=1,
        title="Did the TV ad work?",
        subtitle="Ad in Denver only  ·  Metric: daily sign-ups  ·  100 days",
        pill_c="No ad · 3 cities",
        pill_t="TV ad · Denver",
        icon_t="tv",
        icon_c="city",
        treated="Denver",
        controls=["Austin", "Phoenix", "Seattle"],
        tag_t="TV ad",
        sub_q="One market, no A/B test",
        sub_controls="Cities the ad never reached",
        tag_c="no ad",
        launch="TV ad · day 71",
        series="Denver sign-ups / day",
        question="how many sign-ups did the ad cause?",
        question_sub="We need Denver without the ad, which nobody can observe.",
        trap_head="DENVER · AVG SIGN-UPS / DAY",
        trap_warn="✗ Growth + seasonality get credited to the ad",
        model_label="without the ad (model)",
        pre_note="pre-period fit tracks the weekly wiggles",
        post_note="band widens",
        unit="sign-ups / day",
        cum_unit="in 30 days",
        rel_unit="vs no-ad forecast",
        verdict="The TV ad worked",
        footer="No A/B test?  Borrow controls  ·  Forecast the counterfactual  ·  Run a placebo",
        aria="One-page animated explainer of CausalImpact: measuring a TV ad with a "
        "time-series counterfactual",
    ),
    "feature": dict(
        simulate=simulate_feature,
        dp=1,
        title="Did one-tap checkout work?",
        subtitle="Shipped to all iOS users  ·  Metric: daily checkouts  ·  100 days",
        pill_c="Android + web",
        pill_t="iOS · one-tap",
        icon_t="phone",
        icon_c="devices",
        treated="iOS app",
        controls=["Android", "Mobile web", "Desktop"],
        tag_t="one-tap",
        sub_q="One platform, no A/B test",
        sub_controls="Platforms without it",
        tag_c="no feature",
        launch="iOS release · day 71",
        series="iOS checkouts / day",
        question="how many checkouts did it add?",
        question_sub="It went to every iOS user at once, so there is no control group.",
        trap_head="iOS · AVG CHECKOUTS / DAY",
        trap_warn="✗ Growth + a sitewide sale get credited to the feature",
        model_label="without one-tap (model)",
        pre_note="pre-period fit tracks the weekly cycle",
        post_note="sale: in forecast too",
        unit="checkouts / day",
        cum_unit="in 30 days",
        rel_unit="vs counterfactual",
        verdict="One-tap checkout worked",
        footer="Can't A/B test?  Borrow controls  ·  Forecast the counterfactual  ·  Run a placebo",
        aria="One-page animated explainer of CausalImpact: measuring a product feature "
        "launch with a time-series counterfactual",
    ),
}


def nice_ticks(lo, hi, n=3):
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10)), key=lambda s: abs(s - raw))
    first = math.ceil(lo / step) * step
    ticks = [first + i * step for i in range(int((hi - first) / step) + 1)]
    return [int(t) if float(t).is_integer() else round(t, 2) for t in ticks]


def axis(values, pad=0.06, n=3, top=None):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    lo, hi = v.min(), v.max()
    span = hi - lo
    lo, hi = lo - pad * span, hi + (pad if top is None else top) * span
    return {"lo": round(float(lo), 1), "hi": round(float(hi), 1), "ticks": nice_ticks(lo, hi, n)}


def analyse(name, cfg):
    y, X, truth = cfg["simulate"]()
    r = causal_impact(y, X, N_PRE)
    placebo = causal_impact(y[:N_PRE], X[:N_PRE], 50)  # fake launch at day 51, pre-period only
    dp = cfg["dp"]

    def f(v, d=dp, sign=False):
        return f"{v:{'+' if sign else ''},.{d}f}"

    before, after = y[:N_PRE].mean(), y[N_PRE:].mean()
    cf = r.predicted[N_PRE:].mean()
    naive = after - before
    h = N - N_PRE
    # The arithmetic a viewer can redo from the screen must add up after rounding.
    rd = lambda v: round(float(v), dp)  # noqa: E731
    checks = {
        "after - before = naive": rd(rd(after) - rd(before)) == rd(naive),
        "actual - counterfactual = effect": rd(rd(after) - rd(cf)) == rd(r.effect),
        "effect x 30 = cumulative": round(rd(r.effect) * h) == round(r.cumulative),
        "effect / counterfactual = relative": round(rd(r.effect) / rd(cf) * 100, 1)
        == round(r.relative_effect * 100, 1),
    }
    growth = [(after / before - 1) * 100] + [
        (c[N_PRE:].mean() / c[:N_PRE].mean() - 1) * 100 for c in X.T
    ]

    print(f"== {name}: true effect {truth:+g}/{cfg['unit']}")
    print(f"avg before {before:.2f}  after {after:.2f}  counterfactual {cf:.2f}")
    print(f"naive before/after {naive:+.2f}")
    print(f"effect {r.effect:+.2f}  95% CI [{r.ci_low:.2f}, {r.ci_high:.2f}]  p {r.p_value:.2g}")
    lo, hi = r.cumulative_ci_low, r.cumulative_ci_high
    print(f"cumulative {r.cumulative:+.1f}  CI [{lo:.1f}, {hi:.1f}]")
    print(f"relative {r.relative_effect:+.2%}  naive/effect {naive / r.effect:.2f}x")
    print(f"weights {np.round(r.beta, 3)}  growth % {np.round(growth, 1)}")
    print(f"placebo (fake launch day 51) {placebo.effect:+.2f}  p {placebo.p_value:.2f}")
    for k, ok in checks.items():
        print(f"  {'ok ' if ok else 'BAD'} {k}")

    def r1(a):
        return [None if not np.isfinite(v) else round(float(v), 1) for v in a]

    eff = y - r.predicted
    eff_lo, eff_hi = y - r.predicted_high, y - r.predicted_low
    w = r.beta
    formula = f"{cfg['treated']} ≈ level + " + " + ".join(
        f"{b:.2f}·{c}" for b, c in zip(w, cfg["controls"], strict=True)
    )
    p_txt = "p < 0.001" if r.p_value < 0.001 else f"p = {r.p_value:.3f}"
    story = {
        **{k: v for k, v in cfg.items() if k != "simulate"},
        "before": f(before),
        "after": f(after),
        "naive": f(naive, sign=True),
        "cf": f(cf),
        "growth": [round(float(g), 1) for g in growth],
        "weights": [round(float(b), 2) for b in w],
        "formula": formula,
        "effect": f(r.effect, sign=True),
        "effect_v": round(float(r.effect), 1),
        "effect_short": f(r.effect, 0, sign=True),
        "ci": f"CI {f(r.ci_low)} – {f(r.ci_high)}",
        "cum": f(r.cumulative, 0, sign=True),
        "cum_v": round(float(r.cumulative), 1),
        "cum_ci": f"{f(r.cumulative_ci_low, 0)}–{f(r.cumulative_ci_high, 0)}",
        "cum_ci_tile": f"CI {f(r.cumulative_ci_low, 0)} – {f(r.cumulative_ci_high, 0)}",
        "rel": f"{r.relative_effect * 100:+.1f}%",
        "p": p_txt,
        "placebo": f"placebo launch at day 51: p = {placebo.p_value:.2f}",
        "naive_warn": f"Naive before/after said {f(naive, sign=True)}: "
        f"{naive / r.effect:.1f}× too high",
        "ax_q": axis(y, n=3, top=0.2),  # headroom for the launch label
        "ax_f": axis(np.r_[y, r.predicted_low[10:], r.predicted_high[10:]], n=3, top=0.15),
        "ax_pw": axis(np.r_[eff_lo[10:], eff_hi[10:]], n=2),
        "ax_cum": axis(np.r_[0, r.cumulative_path_high], pad=0.04, n=2),
        "ax_ctrl": [axis(c) for c in X.T],
        "D": {
            "y": r1(y),
            "X": [r1(c) for c in X.T],
            "pred": r1(r.predicted),
            "lo": r1(r.predicted_low),
            "hi": r1(r.predicted_high),
            "eff": r1(eff),
            "eff_lo": r1(eff_lo),
            "eff_hi": r1(eff_hi),
            "cum": r1(r.cumulative_path),
            "cum_lo": r1(r.cumulative_path_low),
            "cum_hi": r1(r.cumulative_path_high),
        },
    }
    if not all(checks.values()):
        raise SystemExit(f"{name}: rounded on-screen arithmetic does not add up; pick another seed")
    out = HERE / f"{name}.data.js"
    header = "// Generated by impact_numbers.py; do not edit by hand.\n"
    out.write_text(header + f"window.STORY = {json.dumps(story, ensure_ascii=False)};\n")
    print(f"  wrote {out.name}")


if __name__ == "__main__":
    for name, cfg in STORIES.items():
        analyse(name, cfg)
