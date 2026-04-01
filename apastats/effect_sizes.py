"""
Effect size calculations following APA 7th edition and JAP norms.

Provides Cohen's *d*, *f*², *R*², partial eta-squared, and Cohen's
benchmark interpretations.  Confidence intervals for effect sizes
use the noncentral distribution approach where applicable.

References
----------
Cohen, J. (1988). *Statistical power analysis for the behavioral
    sciences* (2nd ed.). Erlbaum.
Lakens, D. (2013). Calculating and reporting effect sizes to facilitate
    cumulative science. *Frontiers in Psychology*, 4, 863.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats as sp_stats

from apastats.formatting import fmt_number, fmt_ci


# ═══════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EffectSizeResult:
    """A single effect-size estimate with CI and interpretation."""

    name: str
    value: float
    ci_lower: float
    ci_upper: float
    interpretation: str  # "small", "medium", "large"
    ci_level: float = 0.95

    def __repr__(self) -> str:
        v = fmt_number(self.value, stat_type="d" if "d" in self.name.lower() else "r2")
        ci = fmt_ci(self.ci_lower, self.ci_upper)
        return f"{self.name} = {v}, 95% CI {ci} ({self.interpretation})"


# ═══════════════════════════════════════════════════════════════════════════
# Cohen's d
# ═══════════════════════════════════════════════════════════════════════════

def _cohens_d_benchmark(d: float) -> str:
    """Cohen's (1988) benchmarks for d."""
    d_abs = abs(d)
    if d_abs < 0.20:
        return "negligible"
    if d_abs < 0.50:
        return "small"
    if d_abs < 0.80:
        return "medium"
    return "large"


def cohens_d(
    group1: np.ndarray,
    group2: np.ndarray,
    pooled: bool = True,
    ci_level: float = 0.95,
) -> EffectSizeResult:
    """Compute Cohen's *d* for two independent groups.

    Parameters
    ----------
    group1, group2 : array-like
        Sample data for each group.
    pooled : bool
        If True (default), use pooled *SD*; if False, use the control
        group (group2) *SD* (Glass's delta).
    ci_level : float
        Confidence level for the CI (default .95).

    Returns
    -------
    EffectSizeResult
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    n1, n2 = len(g1), len(g2)

    m1, m2 = g1.mean(), g2.mean()

    if pooled:
        s = math.sqrt(
            ((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1))
            / (n1 + n2 - 2)
        )
    else:
        s = g2.std(ddof=1)

    d = (m1 - m2) / s if s > 0 else 0.0

    # CI via noncentral t distribution
    se_d = math.sqrt(1 / n1 + 1 / n2 + d ** 2 / (2 * (n1 + n2)))
    alpha = 1 - ci_level
    t_crit = sp_stats.t.ppf(1 - alpha / 2, n1 + n2 - 2)
    ci_lo = d - t_crit * se_d
    ci_hi = d + t_crit * se_d

    return EffectSizeResult(
        name="Cohen's d",
        value=d,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        interpretation=_cohens_d_benchmark(d),
        ci_level=ci_level,
    )


# ═══════════════════════════════════════════════════════════════════════════
# f² (for regression)
# ═══════════════════════════════════════════════════════════════════════════

def _f2_benchmark(f2: float) -> str:
    """Cohen's (1988) benchmarks for f²."""
    if f2 < 0.02:
        return "negligible"
    if f2 < 0.15:
        return "small"
    if f2 < 0.35:
        return "medium"
    return "large"


def cohens_f2(
    r2_full: float,
    r2_reduced: float = 0.0,
) -> EffectSizeResult:
    """Compute Cohen's *f*² for a set of predictors in regression.

    ``f² = (R²_full - R²_reduced) / (1 - R²_full)``

    For the overall model, set ``r2_reduced = 0``.
    For an incremental set of predictors (e.g. the interaction term),
    pass the *R*² without those predictors as ``r2_reduced``.

    Parameters
    ----------
    r2_full : float
        *R*² of the full model.
    r2_reduced : float
        *R*² of the reduced model (default 0 = overall effect).

    Returns
    -------
    EffectSizeResult
        Note: CI is not computed for *f*² (set to NaN).
    """
    denom = 1 - r2_full
    f2 = (r2_full - r2_reduced) / denom if denom > 0 else 0.0

    return EffectSizeResult(
        name="Cohen's f\u00b2",
        value=f2,
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        interpretation=_f2_benchmark(f2),
    )


# ═══════════════════════════════════════════════════════════════════════════
# R² benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def _r2_benchmark(r2: float) -> str:
    """Cohen's (1988) benchmarks for R²."""
    if r2 < 0.02:
        return "negligible"
    if r2 < 0.13:
        return "small"
    if r2 < 0.26:
        return "medium"
    return "large"


def r2_effect(r2: float) -> EffectSizeResult:
    """Wrap an *R*² value with Cohen's benchmark interpretation."""
    return EffectSizeResult(
        name="R\u00b2",
        value=r2,
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        interpretation=_r2_benchmark(r2),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Partial eta-squared
# ═══════════════════════════════════════════════════════════════════════════

def _eta2_benchmark(eta2: float) -> str:
    """Cohen's benchmarks for partial eta-squared (same as R²)."""
    return _r2_benchmark(eta2)


def partial_eta_squared(
    ss_effect: float,
    ss_error: float,
) -> EffectSizeResult:
    """Compute partial eta-squared from sums of squares.

    ``η²_p = SS_effect / (SS_effect + SS_error)``
    """
    total = ss_effect + ss_error
    eta2 = ss_effect / total if total > 0 else 0.0

    return EffectSizeResult(
        name="partial \u03b7\u00b2",
        value=eta2,
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        interpretation=_eta2_benchmark(eta2),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Kappa-squared (for mediation)
# ═══════════════════════════════════════════════════════════════════════════

def kappa_squared(
    ab: float,
    x_sd: float,
    y_sd: float,
) -> EffectSizeResult:
    """Compute kappa-squared for an indirect effect.

    ``κ² = ab × (SD_x / SD_y)`` — the standardised indirect effect,
    which can be interpreted on the correlation metric.

    Note: Preacher & Kelley (2011) introduced κ² but it has since been
    criticised.  The **completely standardised indirect effect**
    (ab × SD_x / SD_y) is the more commonly reported metric in JAP.
    """
    kappa2 = abs(ab * x_sd / y_sd) if y_sd > 0 else 0.0

    return EffectSizeResult(
        name="Standardised indirect effect",
        value=ab * x_sd / y_sd if y_sd > 0 else 0.0,
        ci_lower=float("nan"),
        ci_upper=float("nan"),
        interpretation=_r2_benchmark(kappa2),  # rough benchmark
    )
