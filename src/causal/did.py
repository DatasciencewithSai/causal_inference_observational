"""Difference-in-differences (2x2 design)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from causal._ols import ols


@dataclass(frozen=True)
class DiDResult:
    att: float  # average treatment effect on the treated
    se: float
    p_value: float
    ci_low: float
    ci_high: float


def diff_in_diff(
    y,
    treated,
    post,
    clusters=None,
    alpha: float = 0.05,
) -> DiDResult:
    """Estimate the 2x2 DiD effect via y ~ treated + post + treated:post.

    y: outcome per observation.
    treated: 1 if the unit is in the treatment group, else 0.
    post: 1 if the observation is after the intervention, else 0.
    clusters: optional unit ids for cluster-robust SEs (recommended for panels).

    Identifying assumption: parallel trends between groups absent treatment.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(treated, dtype=float)
    t = np.asarray(post, dtype=float)
    if not (len(y) == len(d) == len(t)):
        raise ValueError("y, treated and post must have the same length")
    for name, v in (("treated", d), ("post", t)):
        if not np.isin(v, (0, 1)).all():
            raise ValueError(f"{name} must be 0/1")
    cells = {(a, b) for a, b in zip(d, t, strict=True)}
    if len(cells) < 4:
        raise ValueError("need observations in all four treated x post cells")
    X = np.column_stack([np.ones_like(y), d, t, d * t])
    cl = None if clusters is None else np.asarray(clusters)
    beta, se = ols(X, y, cl)
    att, s = beta[3], se[3]
    z = att / s
    p = 2 * stats.norm.sf(abs(z))
    crit = stats.norm.ppf(1 - alpha / 2)
    return DiDResult(att, s, p, att - crit * s, att + crit * s)
