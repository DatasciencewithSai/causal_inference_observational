# causal_inference_observational

Causal inference for **uncontrolled experiments**: when you can't randomize, estimate effects from
observational data and quasi-experiments.

Sister repos: [experimentation_ab_testing](https://github.com/DatasciencewithSai/experimentation_ab_testing)
(randomized A/B tests) and [Experimentation_guide](https://github.com/DatasciencewithSai/Experimentation_guide) (docs and reading list).

## Methods

| Module | Method | Key assumption |
|---|---|---|
| `causal.did` | Difference-in-differences (2x2), cluster-robust SEs | Parallel trends |
| `causal.propensity` | Propensity scores, IPW average treatment effect, covariate balance (SMD) | No unmeasured confounding, overlap |
| `causal.impact` | CausalImpact-style time-series counterfactual (local level + control series, exact Bayesian intervals) | Controls unaffected; pre-period relationship holds |

Planned (see [docs/ROADMAP.md](docs/ROADMAP.md)): event-study DiD, synthetic control, market matching,
regression discontinuity, matching estimators.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from causal import (
    causal_impact,
    diff_in_diff,
    ipw_ate,
    propensity_scores,
    ipw_weights,
    standardized_mean_differences,
)

# Difference-in-differences on a panel (one row per unit x period)
diff_in_diff(y, treated=is_treated_group, post=is_after_launch, clusters=unit_id)

# Propensity score weighting
ipw_ate(y, treated, X=covariates)  # ATE with bootstrap CI
ps = propensity_scores(covariates, treated)
standardized_mean_differences(covariates, treated, ipw_weights(ps, treated))  # balance check

# CausalImpact-style: one treated series, control series the intervention didn't touch
r = causal_impact(y_market, X=control_markets, n_pre=70)  # first 70 steps are pre-period
r.effect, r.ci_low, r.ci_high, r.cumulative, r.relative_effect
```

## Visual explainers

`visuals/did_pre_post/` is a one-minute animated explainer of pre/post analysis with a DiD
regression (`did_pre_post.mp4`). Every number on screen comes from `did_numbers.py`, which
simulates the store panel and runs `diff_in_diff`. Re-render with `node render.js` from that
folder (Playwright + ffmpeg).

`visuals/causal_impact/` is a one-minute animated explainer of CausalImpact (`causal_impact.mp4`).
Every number on screen comes from `impact_numbers.py`, which simulates the campaign and runs
`causal_impact`. Re-render with `node render.js` from that folder (Playwright + ffmpeg).

## Development

```bash
pytest
ruff check . && ruff format --check .
```

CI runs both on every push and pull request (Python 3.10–3.13).
