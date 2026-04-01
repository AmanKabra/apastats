"""
Conditional process analysis (moderated mediation) following JAP norms.

Implements Hayes PROCESS macro equivalents:
  - **Model 7**: First-stage moderated mediation (X→M moderated by W)
  - **Model 8**: Model 7 + direct-effect moderation
  - **Model 14**: Second-stage moderated mediation (M→Y moderated by W)
  - **Model 15**: Model 14 + direct-effect moderation

Key conventions
---------------
* All variables **mean-centred** before computing product terms.
* **Percentile bootstrap** CIs for indirect effects and the index of
  moderated mediation (10 000 resamples by default).
* Conditional indirect effects probed at −1 *SD*, mean, +1 *SD* of *W*.
* **Unstandardised coefficients** throughout.

References
----------
Hayes, A. F. (2022). *Introduction to mediation, moderation, and
    conditional process analysis* (3rd ed.). Guilford.
Hayes, A. F. (2015). An index and test of linear moderated mediation.
    *Multivariate Behavioral Research*, 50, 1–22.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats

from apastats.formatting import (
    fmt_number,
    fmt_p,
    fmt_r2,
    fmt_ci,
    significance_stars,
    report_regression_coeff,
    report_indirect_effect,
)


# ═══════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConditionalIndirectEffect:
    """Indirect effect of X on Y through M at a specific level of W."""

    w_label: str       # e.g. "−1 SD", "Mean", "+1 SD"
    w_value: float     # raw centred value of W
    ab: float          # point estimate
    boot_se: float
    ci_lower: float
    ci_upper: float
    significant: bool  # CI excludes zero


@dataclass
class ConditionalProcessResult:
    """Full results from a conditional process analysis.

    Attributes
    ----------
    model : int
        PROCESS model number (7, 8, 14, or 15).
    mediator_model : sm.OLS fitted result
    outcome_model : sm.OLS fitted result
    mediator_coeffs : dict
        Named coefficients from the mediator regression.
    outcome_coeffs : dict
        Named coefficients from the outcome regression.
    conditional_indirect : list of ConditionalIndirectEffect
        Indirect effects at probed levels of W.
    imm : float
        Index of moderated mediation (point estimate).
    imm_boot_se : float
    imm_ci_lower : float
    imm_ci_upper : float
    imm_significant : bool
    direct_effect : float
        Direct effect (c'). For Models 8/15 this is at the mean of W.
    conditional_direct : list or None
        For Models 8/15, direct effects at probed levels of W.
    r2_m : float
    r2_y : float
    n : int
    n_boot : int
    table_str : str
    x_name : str
    m_name : str
    y_name : str
    w_name : str
    """

    model: int
    mediator_model: object
    outcome_model: object
    mediator_coeffs: Dict[str, float]
    outcome_coeffs: Dict[str, float]
    conditional_indirect: List[ConditionalIndirectEffect]
    imm: float
    imm_boot_se: float
    imm_ci_lower: float
    imm_ci_upper: float
    imm_significant: bool
    direct_effect: float
    conditional_direct: Optional[List[dict]]
    r2_m: float
    r2_y: float
    n: int
    n_boot: int
    table_str: str
    x_name: str
    m_name: str
    y_name: str
    w_name: str

    def __repr__(self) -> str:
        return self.table_str

    def report(self) -> str:
        """Copy-paste APA in-text strings."""
        lines: list[str] = []
        lines.append(f"PROCESS Model {self.model} (N = {self.n})")

        # Index of moderated mediation
        ci = fmt_ci(self.imm_ci_lower, self.imm_ci_upper, stat_type="b")
        sig = "significant" if self.imm_significant else "not significant"
        lines.append(
            f"Index of moderated mediation: "
            f"{fmt_number(self.imm, 'b')}, SE = {fmt_number(self.imm_boot_se, 'se')}, "
            f"95% CI {ci} ({sig})"
        )

        # Conditional indirect effects
        lines.append("Conditional indirect effects:")
        for cie in self.conditional_indirect:
            ci = fmt_ci(cie.ci_lower, cie.ci_upper, stat_type="b")
            sig = "significant" if cie.significant else "not significant"
            lines.append(
                f"  At {cie.w_label} of {self.w_name}: "
                f"ab = {fmt_number(cie.ab, 'b')}, 95% CI {ci} ({sig})"
            )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ols(y, X):
    X_c = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_c).fit()


def _build_model_matrices(
    df: pd.DataFrame,
    x_c: str, w_c: str, m_col: str,
    model: int,
    covariates: list[str],
):
    """Return (X_mediator, X_outcome) design matrices for the given PROCESS model."""

    xw = f"{x_c}_x_{w_c}"
    mw = f"{m_col}_x_{w_c}"
    df[xw] = df[x_c] * df[w_c]
    df[mw] = df[m_col] * df[w_c]

    if model == 7:
        # M = a1*X + a2*W + a3*X*W  (+covariates)
        # Y = c'*X + b*M  (+covariates)
        med_preds = [x_c, w_c, xw] + covariates
        out_preds = [x_c, m_col] + covariates

    elif model == 8:
        # M = a1*X + a2*W + a3*X*W  (+covariates)
        # Y = c1'*X + c2'*W + c3'*X*W + b1*M  (+covariates)
        med_preds = [x_c, w_c, xw] + covariates
        out_preds = [x_c, w_c, xw, m_col] + covariates

    elif model == 14:
        # M = a*X  (+covariates)
        # Y = c'*X + b1*M + b2*W + b3*M*W  (+covariates)
        med_preds = [x_c] + covariates
        out_preds = [x_c, m_col, w_c, mw] + covariates

    elif model == 15:
        # M = a*X  (+covariates)
        # Y = c1'*X + c2'*W + c3'*X*W + b1*M + b2*M*W  (+covariates)
        med_preds = [x_c] + covariates
        out_preds = [x_c, w_c, xw, m_col, mw] + covariates

    else:
        raise ValueError(f"Unsupported PROCESS model: {model}")

    return med_preds, out_preds


def _extract_coefficients(model_fit, pred_names):
    """Extract named coefficients dict from a fitted OLS model."""
    names = ["const"] + pred_names
    return {name: float(model_fit.params.iloc[i]) for i, name in enumerate(names)}


def _compute_conditional_indirect(
    med_coeffs: dict,
    out_coeffs: dict,
    w_val: float,
    model: int,
    x_c: str, w_c: str, m_col: str,
) -> float:
    """Point estimate of conditional indirect effect at a given w."""
    xw = f"{x_c}_x_{w_c}"
    mw = f"{m_col}_x_{w_c}"

    if model in (7, 8):
        # theta(w) = (a1 + a3*w) * b
        a1 = med_coeffs[x_c]
        a3 = med_coeffs[xw]
        b_key = m_col
        b = out_coeffs[b_key]
        return (a1 + a3 * w_val) * b

    elif model in (14, 15):
        # theta(w) = a * (b1 + b3*w)
        a = med_coeffs[x_c]
        b1 = out_coeffs[m_col]
        b3 = out_coeffs[mw]
        return a * (b1 + b3 * w_val)

    return 0.0


def _compute_imm(
    med_coeffs: dict,
    out_coeffs: dict,
    model: int,
    x_c: str, w_c: str, m_col: str,
) -> float:
    """Point estimate of the index of moderated mediation."""
    xw = f"{x_c}_x_{w_c}"
    mw = f"{m_col}_x_{w_c}"

    if model in (7, 8):
        # IMM = a3 * b
        a3 = med_coeffs[xw]
        b = out_coeffs[m_col]
        return a3 * b

    elif model in (14, 15):
        # IMM = a * b3
        a = med_coeffs[x_c]
        b3 = out_coeffs[mw]
        return a * b3

    return 0.0


def _bootstrap_conditional(
    data_arrays: dict,
    model: int,
    x_c: str, w_c: str, m_col: str,
    med_preds: list[str],
    out_preds: list[str],
    w_probe_vals: list[float],
    n_boot: int,
    seed: Optional[int],
):
    """Bootstrap IMM and conditional indirect effects.

    Returns
    -------
    imm_boots : ndarray (n_boot,)
    cie_boots : ndarray (n_boot, len(w_probe_vals))
    """
    rng = np.random.default_rng(seed)
    n = len(data_arrays["y"])
    n_w = len(w_probe_vals)

    imm_boots = np.empty(n_boot)
    cie_boots = np.empty((n_boot, n_w))

    # Pre-build column arrays
    col_names = list(data_arrays.keys())
    all_data = np.column_stack([data_arrays[c] for c in col_names])

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = {c: all_data[idx, i] for i, c in enumerate(col_names)}

        # Recompute product terms on bootstrap sample
        xw_key = f"{x_c}_x_{w_c}"
        mw_key = f"{m_col}_x_{w_c}"
        boot[xw_key] = boot[x_c] * boot[w_c]
        boot[mw_key] = boot[m_col] * boot[w_c]

        # Fit mediator model
        X_med = np.column_stack([boot[p] for p in med_preds])
        X_med_c = sm.add_constant(X_med, has_constant="add")
        try:
            med_fit = sm.OLS(boot[m_col], X_med_c).fit()
        except Exception:
            imm_boots[b] = np.nan
            cie_boots[b, :] = np.nan
            continue

        med_c = {name: float(med_fit.params[i])
                 for i, name in enumerate(["const"] + med_preds)}

        # Fit outcome model
        X_out = np.column_stack([boot[p] for p in out_preds])
        X_out_c = sm.add_constant(X_out, has_constant="add")
        try:
            out_fit = sm.OLS(boot["y"], X_out_c).fit()
        except Exception:
            imm_boots[b] = np.nan
            cie_boots[b, :] = np.nan
            continue

        out_c = {name: float(out_fit.params[i])
                 for i, name in enumerate(["const"] + out_preds)}

        imm_boots[b] = _compute_imm(med_c, out_c, model, x_c, w_c, m_col)
        for j, wv in enumerate(w_probe_vals):
            cie_boots[b, j] = _compute_conditional_indirect(
                med_c, out_c, wv, model, x_c, w_c, m_col
            )

    return imm_boots, cie_boots


# ═══════════════════════════════════════════════════════════════════════════
# Table formatter
# ═══════════════════════════════════════════════════════════════════════════

def _format_table(result: ConditionalProcessResult, decimals: int = 2) -> str:
    lines: list[str] = []
    lines.append(f"Conditional Process Analysis (PROCESS Model {result.model})")
    lines.append(
        f"Effect of {result.x_name} on {result.y_name} through {result.m_name}, "
        f"moderated by {result.w_name}"
    )

    rule = "\u2500" * 80
    lines.append(rule)

    # Mediator model
    lines.append(f"Mediator model: {result.m_name} (R\u00b2 = {fmt_r2(result.r2_m)})")
    for k, v in result.mediator_coeffs.items():
        if k == "const":
            continue
        lines.append(f"  {k:30s}  b = {fmt_number(v, 'b', decimals)}")

    lines.append("")

    # Outcome model
    lines.append(f"Outcome model: {result.y_name} (R\u00b2 = {fmt_r2(result.r2_y)})")
    for k, v in result.outcome_coeffs.items():
        if k == "const":
            continue
        lines.append(f"  {k:30s}  b = {fmt_number(v, 'b', decimals)}")

    lines.append(rule)

    # IMM
    ci = fmt_ci(result.imm_ci_lower, result.imm_ci_upper, decimals, stat_type="b")
    sig = "*" if result.imm_significant else ""
    lines.append(
        f"Index of moderated mediation{sig}: "
        f"{fmt_number(result.imm, 'b', decimals)}, "
        f"SE = {fmt_number(result.imm_boot_se, 'se', decimals)}, "
        f"95% CI {ci}"
    )

    lines.append("")
    lines.append("Conditional indirect effects of X on Y through M:")
    col_w = {"w": 12, "ab": 10, "se": 10, "ci": 22}
    header = (
        result.w_name.ljust(col_w["w"])
        + "Effect".rjust(col_w["ab"])
        + "Boot SE".rjust(col_w["se"])
        + "95% CI".rjust(col_w["ci"])
    )
    lines.append(header)

    for cie in result.conditional_indirect:
        ci = fmt_ci(cie.ci_lower, cie.ci_upper, decimals, stat_type="b")
        row = (
            cie.w_label.ljust(col_w["w"])
            + fmt_number(cie.ab, "b", decimals).rjust(col_w["ab"])
            + fmt_number(cie.boot_se, "se", decimals).rjust(col_w["se"])
            + ci.rjust(col_w["ci"])
        )
        lines.append(row)

    lines.append(rule)

    lines.append(
        f"Note. N = {result.n}. Bootstrap sample size = {result.n_boot:,}. "
        "Percentile bootstrap confidence intervals. "
        "Variables were mean-centred."
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def conditional_process(
    data: pd.DataFrame,
    x: str,
    m: str,
    y: str,
    w: str,
    model: int = 7,
    covariates: Optional[Sequence[str]] = None,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    decimals: int = 2,
) -> ConditionalProcessResult:
    """Run a conditional process analysis (moderated mediation).

    Implements Hayes' PROCESS Models 7, 8, 14, and 15.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Predictor column.
    m : str
        Mediator column.
    y : str
        Outcome column.
    w : str
        Moderator column.
    model : {7, 8, 14, 15}
        PROCESS model number.
    covariates : sequence of str, optional
    n_boot : int
        Bootstrap resamples (default 10 000).
    ci_level : float
        CI level (default .95).
    seed : int, optional
        Random seed for reproducibility.
    decimals : int
        Decimal places for formatted output.

    Returns
    -------
    ConditionalProcessResult
    """
    if model not in (7, 8, 14, 15):
        raise ValueError(f"Supported models: 7, 8, 14, 15. Got {model}.")

    covariates = list(covariates) if covariates else []

    # --- Validate ---
    all_vars = [x, m, y, w] + covariates
    if len(all_vars) != len(set(all_vars)):
        raise ValueError("All variables must be distinct.")
    needed = list(set(all_vars))
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")

    n_original = len(data)
    df = data[needed].dropna().copy()
    n = len(df)
    if n_original - n > 0:
        warnings.warn(
            f"{n_original - n} observation(s) dropped due to missing values "
            f"(N reduced from {n_original} to {n}).",
            stacklevel=2,
        )

    # --- Mean-centre ---
    x_c = f"{x}_c"
    w_c = f"{w}_c"
    df[x_c] = df[x] - df[x].mean()
    df[w_c] = df[w] - df[w].mean()

    # Build design matrices
    med_preds, out_preds = _build_model_matrices(
        df, x_c, w_c, m, model, covariates
    )

    y_arr = df[y].values.astype(float)
    m_arr = df[m].values.astype(float)

    # --- Fit OLS models ---
    X_med = df[med_preds].astype(float)
    med_fit = _ols(m_arr, X_med)
    med_coeffs = _extract_coefficients(med_fit, med_preds)

    X_out = df[out_preds].astype(float)
    out_fit = _ols(y_arr, X_out)
    out_coeffs = _extract_coefficients(out_fit, out_preds)

    # --- Probe values of W (centred: -1SD, 0, +1SD) ---
    w_sd = df[w_c].std(ddof=1)
    w_probe_labels = ["\u22121 SD", "Mean", "+1 SD"]
    w_probe_vals = [-w_sd, 0.0, w_sd]

    # --- Point estimates ---
    imm_point = _compute_imm(med_coeffs, out_coeffs, model, x_c, w_c, m)
    cie_points = [
        _compute_conditional_indirect(med_coeffs, out_coeffs, wv, model, x_c, w_c, m)
        for wv in w_probe_vals
    ]

    # Direct effect (at mean of W = 0 when centred)
    if model in (7, 14):
        direct = out_coeffs[x_c]
    else:
        # Models 8, 15: direct is conditional, report at mean
        direct = out_coeffs[x_c]  # c1' at w=0

    # Conditional direct for Models 8, 15
    conditional_direct = None
    if model in (8, 15):
        xw_key = f"{x_c}_x_{w_c}"
        c1 = out_coeffs[x_c]
        c3 = out_coeffs[xw_key]
        conditional_direct = [
            {"w_label": lab, "w_value": wv, "direct": c1 + c3 * wv}
            for lab, wv in zip(w_probe_labels, w_probe_vals)
        ]

    # --- Bootstrap ---
    alpha = 1 - ci_level
    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100

    # Prepare data arrays for bootstrap
    data_arrays = {
        x_c: df[x_c].values,
        w_c: df[w_c].values,
        m: m_arr,
        "y": y_arr,
    }
    for cov in covariates:
        data_arrays[cov] = df[cov].values.astype(float)

    imm_boots, cie_boots = _bootstrap_conditional(
        data_arrays, model, x_c, w_c, m,
        med_preds, out_preds, w_probe_vals, n_boot, seed,
    )

    # IMM CI
    imm_ci_lo = float(np.nanpercentile(imm_boots, lo_pct))
    imm_ci_hi = float(np.nanpercentile(imm_boots, hi_pct))
    imm_boot_se = float(np.nanstd(imm_boots, ddof=1))
    imm_sig = not (imm_ci_lo <= 0 <= imm_ci_hi)

    # Conditional indirect CIs
    cond_effects: list[ConditionalIndirectEffect] = []
    for j, (lab, wv) in enumerate(zip(w_probe_labels, w_probe_vals)):
        boot_j = cie_boots[:, j]
        ci_lo = float(np.nanpercentile(boot_j, lo_pct))
        ci_hi = float(np.nanpercentile(boot_j, hi_pct))
        boot_se = float(np.nanstd(boot_j, ddof=1))
        sig = not (ci_lo <= 0 <= ci_hi)
        cond_effects.append(ConditionalIndirectEffect(
            w_label=lab, w_value=wv, ab=cie_points[j],
            boot_se=boot_se, ci_lower=ci_lo, ci_upper=ci_hi,
            significant=sig,
        ))

    # --- Build result ---
    result = ConditionalProcessResult(
        model=model,
        mediator_model=med_fit,
        outcome_model=out_fit,
        mediator_coeffs=med_coeffs,
        outcome_coeffs=out_coeffs,
        conditional_indirect=cond_effects,
        imm=imm_point,
        imm_boot_se=imm_boot_se,
        imm_ci_lower=imm_ci_lo,
        imm_ci_upper=imm_ci_hi,
        imm_significant=imm_sig,
        direct_effect=direct,
        conditional_direct=conditional_direct,
        r2_m=med_fit.rsquared,
        r2_y=out_fit.rsquared,
        n=n,
        n_boot=n_boot,
        table_str="",
        x_name=x,
        m_name=m,
        y_name=y,
        w_name=w,
    )

    result.table_str = _format_table(result, decimals)
    return result
