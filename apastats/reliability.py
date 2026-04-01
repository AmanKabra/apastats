"""
Scale reliability analysis following JAP / APA 7th edition norms.

Provides:
  - Cronbach's alpha (delegates to :func:`apastats.descriptives.cronbach_alpha`)
  - McDonald's omega-total (from one-factor EFA loadings)
  - Composite reliability (CR)
  - Average variance extracted (AVE)
  - Corrected item-total correlations (CITC)
  - Alpha-if-item-deleted

Conventions
-----------
* Omega-total for congeneric one-factor models:
  ω_t = (Σλ)² / [(Σλ)² + Σ(1 − λ²)]
* CR uses the same formula (identical for a congeneric model).
* AVE = Σλ² / k.
* CITC threshold: ≥ .30 acceptable, ≥ .40 preferred.
* Report omega alongside alpha (McNeish, 2018).

References
----------
McNeish, D. (2018). Thanks coefficient alpha, we'll take it from here.
    *Psychological Methods*, 23, 412–433.
Flora, D. B. (2020). Your coefficient alpha is probably wrong, but which
    coefficient omega is right? *Advances in Methods and Practices in
    Psychological Science*, 3, 484–501.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from apastats.descriptives import cronbach_alpha
from apastats.formatting import fmt_number


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReliabilityResult:
    """Full scale-reliability results.

    Attributes
    ----------
    alpha : float
        Cronbach's alpha.
    omega_total : float
        McDonald's omega-total.
    cr : float
        Composite reliability.
    ave : float
        Average variance extracted.
    citc : pd.Series
        Corrected item-total correlations (indexed by item name).
    alpha_if_deleted : pd.Series
        Alpha with each item removed (indexed by item name).
    factor_loadings : pd.Series
        Standardised one-factor loadings (indexed by item name).
    n_items : int
    n_obs : int
    item_names : list[str]
    table_str : str
    """

    alpha: float
    omega_total: float
    cr: float
    ave: float
    citc: pd.Series
    alpha_if_deleted: pd.Series
    factor_loadings: pd.Series
    n_items: int
    n_obs: int
    item_names: List[str]
    table_str: str = ""

    def __repr__(self) -> str:
        return self.table_str

    def report(self) -> str:
        """Return copy-paste APA in-text reliability string."""
        return (
            f"\u03b1 = {fmt_number(self.alpha, 'alpha')}, "
            f"\u03c9 = {fmt_number(self.omega_total, 'alpha')}, "
            f"CR = {fmt_number(self.cr, 'alpha')}, "
            f"AVE = {fmt_number(self.ave, 'alpha')}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Computations
# ═══════════════════════════════════════════════════════════════════════════

def _one_factor_loadings(item_scores: np.ndarray) -> np.ndarray:
    """Extract standardised one-factor loadings via principal axis factoring.

    Falls back to a correlation-based approach if factor_analyzer is not
    available.
    """
    try:
        from factor_analyzer import FactorAnalyzer
        fa = FactorAnalyzer(n_factors=1, method="principal", rotation=None)
        fa.fit(item_scores)
        loadings = fa.loadings_[:, 0]
    except ImportError:
        # Fallback: loadings = correlation of each item with total
        # (approximation, not true factor loadings)
        total = item_scores.sum(axis=1)
        corr_mat = np.corrcoef(item_scores.T)
        eigenvalues, eigenvectors = np.linalg.eigh(corr_mat)
        # Largest eigenvalue
        idx = np.argmax(eigenvalues)
        loadings = eigenvectors[:, idx] * np.sqrt(eigenvalues[idx])
    return loadings


def _omega_total(loadings: np.ndarray) -> float:
    """McDonald's omega-total from one-factor loadings.

    ω_t = (Σλ)² / [(Σλ)² + Σ(1 − λ²)]
    """
    sum_lambda = np.sum(loadings)
    sum_uniqueness = np.sum(1 - loadings ** 2)
    numerator = sum_lambda ** 2
    return float(numerator / (numerator + sum_uniqueness))


def _composite_reliability(loadings: np.ndarray) -> float:
    """Composite reliability (CR). Same formula as omega-total."""
    return _omega_total(loadings)


def _ave(loadings: np.ndarray) -> float:
    """Average variance extracted."""
    return float(np.mean(loadings ** 2))


def _citc(item_scores: np.ndarray) -> np.ndarray:
    """Corrected item-total correlations.

    For each item, correlate the item with the sum of all *other* items.
    """
    n_items = item_scores.shape[1]
    citc_vals = np.empty(n_items)
    total = item_scores.sum(axis=1)
    for j in range(n_items):
        rest_total = total - item_scores[:, j]
        r = np.corrcoef(item_scores[:, j], rest_total)[0, 1]
        citc_vals[j] = r
    return citc_vals


def _alpha_if_deleted(item_scores: np.ndarray) -> np.ndarray:
    """Cronbach's alpha when each item is removed in turn."""
    n_items = item_scores.shape[1]
    alpha_vals = np.empty(n_items)
    for j in range(n_items):
        remaining = np.delete(item_scores, j, axis=1)
        if remaining.shape[1] < 2:
            alpha_vals[j] = float("nan")
        else:
            alpha_vals[j] = cronbach_alpha(remaining)
    return alpha_vals


# ═══════════════════════════════════════════════════════════════════════════
# Table formatter
# ═══════════════════════════════════════════════════════════════════════════

def _format_table(result: ReliabilityResult, decimals: int = 2) -> str:
    lines: list[str] = []
    lines.append("Scale Reliability Analysis")
    rule = "\u2500" * 65
    lines.append(rule)

    header = (
        "Item".ljust(24)
        + "Loading".rjust(10)
        + "CITC".rjust(10)
        + "\u03b1 if deleted".rjust(14)
    )
    lines.append(header)
    lines.append(rule)

    for item in result.item_names:
        loading = fmt_number(result.factor_loadings[item], "alpha", decimals)
        citc = fmt_number(result.citc[item], "r", decimals)
        aid = fmt_number(result.alpha_if_deleted[item], "alpha", decimals)
        lines.append(
            item.ljust(24) + loading.rjust(10) + citc.rjust(10) + aid.rjust(14)
        )

    lines.append(rule)
    lines.append(
        f"Scale: \u03b1 = {fmt_number(result.alpha, 'alpha', decimals)}, "
        f"\u03c9 = {fmt_number(result.omega_total, 'alpha', decimals)}, "
        f"CR = {fmt_number(result.cr, 'alpha', decimals)}, "
        f"AVE = {fmt_number(result.ave, 'alpha', decimals)}"
    )
    lines.append(f"N = {result.n_obs}, k = {result.n_items} items")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def scale_reliability(
    data: pd.DataFrame,
    items: Sequence[str],
    decimals: int = 2,
) -> ReliabilityResult:
    """Compute comprehensive scale reliability statistics.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataset.
    items : sequence of str
        Column names of the scale items.

    Returns
    -------
    ReliabilityResult
    """
    items = list(items)
    missing = [c for c in items if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")
    if len(items) < 2:
        raise ValueError("At least 2 items are required for reliability analysis.")

    item_scores = data[items].dropna().values.astype(float)
    n_obs, n_items = item_scores.shape

    # Alpha
    alpha = cronbach_alpha(item_scores)

    # One-factor loadings
    loadings = _one_factor_loadings(item_scores)
    # Ensure loadings are positive (flip sign if majority negative)
    if np.sum(loadings < 0) > np.sum(loadings > 0):
        loadings = -loadings

    # Omega, CR, AVE
    omega = _omega_total(loadings)
    cr = _composite_reliability(loadings)
    ave = _ave(loadings)

    # CITC
    citc_vals = _citc(item_scores)

    # Alpha-if-deleted
    aid_vals = _alpha_if_deleted(item_scores)

    # Build Series
    loading_series = pd.Series(loadings, index=items, name="Loading")
    citc_series = pd.Series(citc_vals, index=items, name="CITC")
    aid_series = pd.Series(aid_vals, index=items, name="Alpha_if_deleted")

    result = ReliabilityResult(
        alpha=alpha,
        omega_total=omega,
        cr=cr,
        ave=ave,
        citc=citc_series,
        alpha_if_deleted=aid_series,
        factor_loadings=loading_series,
        n_items=n_items,
        n_obs=n_obs,
        item_names=items,
    )
    result.table_str = _format_table(result, decimals)
    return result
