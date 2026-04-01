"""
Confirmatory factor analysis (CFA) following JAP / APA 7th edition norms.

Provides:
  - CFA model fitting via **semopy** backend
  - Fit indices: chi2, df, p, CFI, TLI, RMSEA (with 90 % CI), SRMR
  - Fit interpretation against Hu & Bentler (1999) cutoffs
  - Standardised factor loadings table
  - Composite reliability (CR) and AVE per factor
  - Discriminant validity: Fornell–Larcker matrix, HTMT
  - Model comparison (chi-square difference test)

Conventions
-----------
* Fit cutoffs (Hu & Bentler, 1999): CFI >= .95, TLI >= .95,
  RMSEA <= .06, SRMR <= .08 for good fit.
* Factor loadings >= .40 acceptable, >= .70 ideal.
* Discriminant validity: sqrt(AVE) > inter-factor correlation
  (Fornell & Larcker, 1981); HTMT < .85 (Henseler et al., 2015).

References
----------
Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes.
    *Structural Equation Modeling*, 6, 1–55.
Fornell, C., & Larcker, D. F. (1981). Evaluating structural equation
    models. *Journal of Marketing Research*, 18, 39–50.
Henseler, J., Ringle, C. M., & Sarstedt, M. (2015). A new criterion
    for assessing discriminant validity. *JAMS*, 43, 115–135.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from apastats.formatting import fmt_number, fmt_p, fmt_ci, fmt_r2


# ═══════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FitIndices:
    """CFA model fit indices."""

    chi2: float
    df: int
    p: float
    cfi: float
    tli: float
    rmsea: float
    rmsea_ci_lower: float
    rmsea_ci_upper: float
    srmr: float
    aic: float = float("nan")
    bic: float = float("nan")

    def interpretation(self) -> str:
        """Evaluate fit against Hu & Bentler (1999) cutoffs."""
        good = (
            self.cfi >= 0.95 and self.tli >= 0.95
            and self.rmsea <= 0.06 and self.srmr <= 0.08
        )
        acceptable = (
            self.cfi >= 0.90 and self.tli >= 0.90
            and self.rmsea <= 0.08 and self.srmr <= 0.10
        )
        if good:
            return "good"
        if acceptable:
            return "acceptable"
        return "poor"

    def report(self) -> str:
        """APA in-text string for fit indices."""
        return (
            f"\u03c7\u00b2({self.df}) = {fmt_number(self.chi2, 'f')}, "
            f"p {fmt_p(self.p)}, "
            f"CFI = {fmt_number(self.cfi, 'alpha')}, "
            f"TLI = {fmt_number(self.tli, 'alpha')}, "
            f"RMSEA = {fmt_number(self.rmsea, 'alpha')}, "
            f"90% CI {fmt_ci(self.rmsea_ci_lower, self.rmsea_ci_upper, stat_type='alpha')}, "
            f"SRMR = {fmt_number(self.srmr, 'alpha')}"
        )

    def __repr__(self) -> str:
        return self.report()


@dataclass
class CFAResult:
    """Full CFA results.

    Attributes
    ----------
    fit : FitIndices
    loadings_df : pd.DataFrame
        Standardised factor loadings (items x factors).
    cr : dict[str, float]
        Composite reliability per factor.
    ave : dict[str, float]
        Average variance extracted per factor.
    fornell_larcker_df : pd.DataFrame
        sqrt(AVE) on diagonal, inter-factor correlations off-diagonal.
    htmt_df : pd.DataFrame
        Heterotrait-monotrait ratio matrix.
    factor_names : list[str]
    n : int
    table_str : str
    """

    fit: FitIndices
    loadings_df: pd.DataFrame
    cr: Dict[str, float]
    ave: Dict[str, float]
    fornell_larcker_df: pd.DataFrame
    htmt_df: pd.DataFrame
    factor_names: List[str]
    n: int
    table_str: str = ""

    def __repr__(self) -> str:
        return self.table_str

    def report(self) -> str:
        """APA in-text report."""
        lines = [self.fit.report()]
        for f in self.factor_names:
            lines.append(
                f"  {f}: CR = {fmt_number(self.cr[f], 'alpha')}, "
                f"AVE = {fmt_number(self.ave[f], 'alpha')}"
            )
        return "\n".join(lines)

    def compare(self, other: CFAResult) -> dict:
        """Chi-square difference test between two nested models.

        Returns dict with delta_chi2, delta_df, p.
        """
        d_chi2 = abs(self.fit.chi2 - other.fit.chi2)
        d_df = abs(self.fit.df - other.fit.df)
        if d_df == 0:
            p = 1.0
        else:
            p = float(sp_stats.chi2.sf(d_chi2, d_df))
        return {
            "delta_chi2": d_chi2,
            "delta_df": d_df,
            "p": p,
            "significant": p < 0.05,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _rmsea_ci(chi2: float, df: int, n: int, ci_level: float = 0.90):
    """Compute RMSEA confidence interval via noncentral chi-square."""
    if df <= 0:
        return (0.0, 0.0)

    # Point estimate
    rmsea = math.sqrt(max((chi2 - df) / (df * (n - 1)), 0.0))

    alpha = 1 - ci_level

    # Lower bound: find noncentrality where chi2 is at upper alpha/2
    # Upper bound: find noncentrality where chi2 is at lower alpha/2
    try:
        # Lower: find lambda such that P(chi2 > observed | lambda) = alpha/2
        if chi2 > df:
            ncp_lower = sp_stats.chi2.isf(1 - alpha / 2, df, loc=0, scale=1)
            # Use iterative approach
            from scipy.optimize import brentq

            def _lower_func(ncp):
                return sp_stats.ncx2.sf(chi2, df, ncp) - (1 - alpha / 2)

            try:
                ncp_lo = brentq(_lower_func, 0, max(chi2 * 3, 100))
                ci_lo = math.sqrt(max(ncp_lo / (df * (n - 1)), 0.0))
            except (ValueError, RuntimeError):
                ci_lo = 0.0
        else:
            ci_lo = 0.0

        def _upper_func(ncp):
            return sp_stats.ncx2.sf(chi2, df, ncp) - (alpha / 2)

        try:
            ncp_hi = brentq(_upper_func, 0, max(chi2 * 5, 200))
            ci_hi = math.sqrt(max(ncp_hi / (df * (n - 1)), 0.0))
        except (ValueError, RuntimeError):
            ci_hi = rmsea * 1.5  # fallback rough estimate

    except Exception:
        ci_lo = max(rmsea - 0.02, 0.0)
        ci_hi = rmsea + 0.02

    return (ci_lo, ci_hi)


def _compute_htmt(data: pd.DataFrame, factor_items: dict[str, list[str]]) -> pd.DataFrame:
    """Compute the HTMT matrix.

    HTMT(i,j) = mean(between-trait correlations) /
                sqrt(mean(within-trait-i correlations) * mean(within-trait-j correlations))
    """
    factor_names = list(factor_items.keys())
    k = len(factor_names)
    corr = data.corr()

    htmt_mat = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            items_i = factor_items[factor_names[i]]
            items_j = factor_items[factor_names[j]]

            # Between-trait correlations (items of i correlated with items of j)
            between = []
            for ii in items_i:
                for jj in items_j:
                    between.append(abs(corr.loc[ii, jj]))
            mean_between = np.mean(between)

            # Within-trait correlations (off-diagonal within each factor)
            within_i = []
            for a in range(len(items_i)):
                for b in range(a + 1, len(items_i)):
                    within_i.append(abs(corr.loc[items_i[a], items_i[b]]))
            mean_within_i = np.mean(within_i) if within_i else 1.0

            within_j = []
            for a in range(len(items_j)):
                for b in range(a + 1, len(items_j)):
                    within_j.append(abs(corr.loc[items_j[a], items_j[b]]))
            mean_within_j = np.mean(within_j) if within_j else 1.0

            denom = math.sqrt(mean_within_i * mean_within_j)
            htmt_val = mean_between / denom if denom > 0 else float("nan")
            htmt_mat[i, j] = htmt_mat[j, i] = htmt_val

    return pd.DataFrame(htmt_mat, index=factor_names, columns=factor_names)


def _model_syntax_from_dict(factor_items: dict[str, list[str]]) -> str:
    """Convert a {factor: [items]} dict to semopy/lavaan model syntax."""
    lines = []
    for factor, items in factor_items.items():
        lines.append(f"{factor} =~ {' + '.join(items)}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Table formatter
# ═══════════════════════════════════════════════════════════════════════════

def _format_table(result: CFAResult, decimals: int = 2) -> str:
    lines: list[str] = []
    lines.append("Confirmatory Factor Analysis Results")
    rule = "\u2500" * 72
    lines.append(rule)

    # Fit indices
    lines.append(f"Fit: {result.fit.report()} ({result.fit.interpretation()} fit)")
    lines.append("")

    # Factor loadings
    lines.append("Standardised Factor Loadings:")
    lines.append(rule)

    # Header
    cols = list(result.loadings_df.columns)
    header = "Item".ljust(24)
    for c in cols:
        header += c.rjust(12)
    lines.append(header)
    lines.append(rule)

    for item in result.loadings_df.index:
        row = str(item).ljust(24)
        for c in cols:
            val = result.loadings_df.loc[item, c]
            if abs(val) > 0.001:
                row += fmt_number(val, "alpha", decimals).rjust(12)
            else:
                row += "".rjust(12)
        lines.append(row)

    lines.append(rule)

    # CR and AVE
    cr_row = "CR".ljust(24)
    ave_row = "AVE".ljust(24)
    for c in cols:
        cr_row += fmt_number(result.cr.get(c, 0), "alpha", decimals).rjust(12)
        ave_row += fmt_number(result.ave.get(c, 0), "alpha", decimals).rjust(12)
    lines.append(cr_row)
    lines.append(ave_row)

    lines.append(rule)

    # Discriminant validity
    lines.append("")
    lines.append("Fornell-Larcker Discriminant Validity:")
    fl = result.fornell_larcker_df
    fl_header = "".ljust(16)
    for c in fl.columns:
        fl_header += c.rjust(12)
    lines.append(fl_header)
    for i, row_name in enumerate(fl.index):
        row = str(row_name).ljust(16)
        for j, col_name in enumerate(fl.columns):
            val = fl.iloc[i, j]
            if i == j:
                row += fmt_number(val, "alpha", decimals).rjust(12)
            elif j < i:
                row += fmt_number(val, "r", decimals).rjust(12)
            else:
                row += "".rjust(12)
        lines.append(row)

    lines.append("")
    lines.append(f"Note. N = {result.n}. Diagonal = \u221aAVE. "
                 "Off-diagonal = inter-factor correlations.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def cfa(
    data: pd.DataFrame,
    factors: dict[str, list[str]],
    decimals: int = 2,
) -> CFAResult:
    """Run a confirmatory factor analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataset containing all indicator variables.
    factors : dict[str, list[str]]
        Mapping of factor names to lists of indicator column names.
        Example: ``{"POS": ["pos1", "pos2", "pos3"],
                     "Commit": ["com1", "com2", "com3"]}``
    decimals : int
        Decimal places for formatted output.

    Returns
    -------
    CFAResult

    Raises
    ------
    ImportError
        If semopy is not installed (``pip install semopy``).
    """
    try:
        import semopy
    except ImportError:
        raise ImportError(
            "semopy is required for CFA. Install it with: pip install semopy"
        )

    # --- Validate ---
    all_items = []
    for factor, items in factors.items():
        if len(items) < 2:
            raise ValueError(
                f"Factor '{factor}' must have at least 2 indicators, got {len(items)}."
            )
        all_items.extend(items)
    if len(all_items) != len(set(all_items)):
        raise ValueError("Items must not be shared across factors.")
    missing = [c for c in all_items if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")

    df = data[all_items].dropna()
    n = len(df)

    # --- Fit model ---
    syntax = _model_syntax_from_dict(factors)
    model = semopy.Model(syntax)
    model.fit(df)

    # --- Fit indices ---
    stats = semopy.calc_stats(model)

    chi2 = float(stats.loc["Value", "chi2"])
    dof = int(stats.loc["Value", "DoF"])
    p_chi2 = float(stats.loc["Value", "chi2 p-value"]) if "chi2 p-value" in stats.columns else float(sp_stats.chi2.sf(chi2, dof))
    cfi = float(stats.loc["Value", "CFI"])
    tli = float(stats.loc["Value", "TLI"]) if "TLI" in stats.columns else float("nan")
    rmsea = float(stats.loc["Value", "RMSEA"])
    aic = float(stats.loc["Value", "AIC"]) if "AIC" in stats.columns else float("nan")
    bic = float(stats.loc["Value", "BIC"]) if "BIC" in stats.columns else float("nan")

    # SRMR: not in semopy calc_stats; compute manually
    # SRMR = sqrt(mean of squared standardised residuals in lower triangle)
    try:
        sigma_obs = np.array(df.cov())
        sigma_model = np.array(model.calc_sigma()[0]) if hasattr(model, "calc_sigma") else sigma_obs
        p_vars = sigma_obs.shape[0]
        # Standardised residual matrix
        D = np.diag(1.0 / np.sqrt(np.diag(sigma_obs)))
        std_resid = D @ (sigma_obs - sigma_model) @ D
        # Lower triangle including diagonal
        lower = std_resid[np.tril_indices(p_vars)]
        srmr = float(np.sqrt(np.mean(lower ** 2)))
    except Exception:
        srmr = float("nan")

    rmsea_lo, rmsea_hi = _rmsea_ci(chi2, dof, n)

    fit = FitIndices(
        chi2=chi2, df=dof, p=p_chi2,
        cfi=cfi, tli=tli,
        rmsea=rmsea, rmsea_ci_lower=rmsea_lo, rmsea_ci_upper=rmsea_hi,
        srmr=srmr, aic=aic, bic=bic,
    )

    # --- Standardised loadings ---
    inspect = model.inspect(std_est=True)
    # In semopy, measurement paths are op == "~" with:
    #   lval = indicator (observed), rval = latent variable
    loading_rows = inspect[inspect["op"] == "~"].copy()

    factor_names = list(factors.keys())
    loadings_dict: dict[str, dict[str, float]] = {f: {} for f in factor_names}

    for _, row in loading_rows.iterrows():
        item = row["lval"]   # indicator (observed variable)
        lv = row["rval"]     # latent variable (factor)
        std_val = row["Est. Std"] if "Est. Std" in row.index else row["Estimate"]
        if lv in loadings_dict:
            loadings_dict[lv][item] = float(std_val)

    # Build loadings DataFrame
    loadings_df = pd.DataFrame(0.0, index=all_items, columns=factor_names)
    for factor, item_loadings in loadings_dict.items():
        for item, val in item_loadings.items():
            if item in loadings_df.index:
                loadings_df.loc[item, factor] = val

    # --- CR and AVE per factor ---
    cr_dict: dict[str, float] = {}
    ave_dict: dict[str, float] = {}
    for factor in factor_names:
        lambdas = loadings_df[factor].values
        lambdas = lambdas[np.abs(lambdas) > 0.001]  # only assigned items
        sum_l = np.sum(lambdas)
        sum_uniq = np.sum(1 - lambdas ** 2)
        cr_val = sum_l ** 2 / (sum_l ** 2 + sum_uniq) if (sum_l ** 2 + sum_uniq) > 0 else 0
        ave_val = np.mean(lambdas ** 2) if len(lambdas) > 0 else 0
        cr_dict[factor] = float(cr_val)
        ave_dict[factor] = float(ave_val)

    # --- Fornell-Larcker matrix ---
    # Diagonal = sqrt(AVE), off-diagonal = inter-factor correlations
    k = len(factor_names)
    fl_mat = np.zeros((k, k))

    # Inter-factor correlations from semopy
    factor_corr_rows = inspect[
        (inspect["op"] == "~~")
        & (inspect["lval"] != inspect["rval"])
        & (inspect["lval"].isin(factor_names))
        & (inspect["rval"].isin(factor_names))
    ]

    for _, row in factor_corr_rows.iterrows():
        lv1 = row["lval"]
        lv2 = row["rval"]
        est = row["Est. Std"] if "Est. Std" in row.index else row["Estimate"]
        i = factor_names.index(lv1)
        j = factor_names.index(lv2)
        fl_mat[i, j] = fl_mat[j, i] = float(est)

    for i in range(k):
        fl_mat[i, i] = math.sqrt(ave_dict[factor_names[i]])

    fl_df = pd.DataFrame(fl_mat, index=factor_names, columns=factor_names)

    # --- HTMT ---
    htmt_df = _compute_htmt(df, factors)

    # --- Build result ---
    result = CFAResult(
        fit=fit,
        loadings_df=loadings_df,
        cr=cr_dict,
        ave=ave_dict,
        fornell_larcker_df=fl_df,
        htmt_df=htmt_df,
        factor_names=factor_names,
        n=n,
    )
    result.table_str = _format_table(result, decimals)
    return result
