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

Planned (see [docs/ROADMAP.md](docs/ROADMAP.md)): event-study DiD, synthetic control, market matching,
regression discontinuity, Bayesian structural time series (CausalImpact-style), matching estimators.

## Install

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from causal import (
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
```

## Development

```bash
pytest
ruff check . && ruff format --check .
```

CI runs both on every push and pull request (Python 3.10–3.13).
