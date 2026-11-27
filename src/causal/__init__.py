"""causal: causal inference for observational data and quasi-experiments."""

from causal.did import diff_in_diff
from causal.impact import CausalImpactResult, causal_impact
from causal.propensity import (
    ipw_ate,
    ipw_weights,
    propensity_scores,
    standardized_mean_differences,
)

__all__ = [
    "CausalImpactResult",
    "causal_impact",
    "diff_in_diff",
    "ipw_ate",
    "ipw_weights",
    "propensity_scores",
    "standardized_mean_differences",
]

__version__ = "0.1.0"
