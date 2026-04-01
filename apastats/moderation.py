"""
Moderation (interaction) analysis following JAP / APA 7th edition norms.

Implements:
  1. Hierarchical OLS regression (controls → main effects → interaction).
  2. Simple-slopes analysis at ±1 *SD* of the moderator (Aiken & West, 1991).
  3. Johnson–Neyman regions-of-significance technique.
  4. APA-formatted interaction plots.
  5. APA-formatted regression tables.

Key conventions
---------------
* Predictor and moderator are **mean-centred** before computing the
  product term (Aiken & West, 1991).
* **Unstandardised coefficients** (*b*) are reported when an interaction
  term is present — standardised betas for interaction terms are
  mis-scaled (Aiken & West, 1991; Hayes, 2022).
* Significance: ``*`` *p* < .05, ``**`` *p* < .01.
* Tables use horizontal rules only (no vertical lines).

References
----------
Aiken, L. S., & West, S. G. (1991). *Multiple regression: Testing and
    interpreting interactions*. Sage.
Bauer, D. J., & Curran, P. J. (2005). Probing interactions in fixed and
    multilevel regression. *Multivariate Behavioral Research*, 40, 373–400.
Hayes, A. F. (2022). *Introduction to mediation, moderation, and
    conditional process analysis* (3rd ed.). Guilford.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from apastats.formatting import (
    fmt_number,
    fmt_p,
    fmt_r2,
    significance_stars,
    fmt_ci,
)


# ═══════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimpleSlope:
    """One conditional effect of X on Y at a given level of W."""

    w_label: str          # e.g. "-1 SD", "Mean", "+1 SD"
    w_value: float        # raw (centred) value of W used
    b: float              # unstandardised slope
    se: float
    t: float
    df: float
    p: float
    ci_lower: float
    ci_upper: float


@dataclass
class JohnsonNeymanResult:
    """Johnson–Neyman regions of significance."""

    boundaries: List[float]
    w_range: tuple[float, float]
    pct_below: List[float]
    pct_above: List[float]
    conditional_effects: pd.DataFrame  # columns: w, b, se, t, p, ci_lo, ci_hi


@dataclass
class ModerationResult:
    """Full moderation analysis results.

    Attributes
    ----------
    steps : list of statsmodels RegressionResultsWrapper
        Fitted OLS models for each hierarchical step.
    step_summaries : list of pd.DataFrame
        Coefficient tables for each step.
    r2 : list of float
        *R*² for each step.
    delta_r2 : list of float
        Δ*R*² relative to previous step (0 for step 1).
    delta_f : list of tuple[float, float]]
        (Δ*F*, *p*) for each Δ*R*² test.
    simple_slopes : list of SimpleSlope
        Conditional effects at −1 *SD*, mean, +1 *SD* of the moderator.
    jn : JohnsonNeymanResult or None
        Johnson–Neyman output (``None`` when not requested).
    table_str : str
        APA-formatted plain-text regression table.
    interaction_b : float
        Unstandardised interaction coefficient.
    interaction_p : float
        *p* value for the interaction term.
    n : int
        Sample size.
    x_name : str
    w_name : str
    y_name : str
    """

    steps: list
    step_summaries: List[pd.DataFrame]
    r2: List[float]
    delta_r2: List[float]
    delta_f: List[tuple]
    simple_slopes: List[SimpleSlope]
    jn: Optional[JohnsonNeymanResult]
    table_str: str
    interaction_b: float
    interaction_p: float
    n: int
    x_name: str
    w_name: str
    y_name: str
    _param_names: List[str] = field(default_factory=list, repr=False)

    def __repr__(self) -> str:  # noqa: D105
        return self.table_str

    # Convenience ---------------------------------------------------------

    def plot(self, **kwargs) -> plt.Figure:
        """Draw a standard 2-way interaction plot (±1 *SD*)."""
        return plot_interaction(self, **kwargs)

    def plot_jn(self, **kwargs) -> plt.Figure:
        """Draw a Johnson–Neyman plot."""
        if self.jn is None:
            raise ValueError("Johnson–Neyman was not computed. Re-run with jn=True.")
        return plot_johnson_neyman(self, **kwargs)

    def report(self) -> str:
        """Return copy-paste APA in-text strings for the moderation analysis."""
        from apastats.formatting import (
            report_regression_coeff, report_model_fit,
            report_r2_change, report_simple_slope,
        )
        lines: list[str] = []

        # Final model fit
        final = self.steps[-1]
        lines.append("Model fit:")
        lines.append(f"  {report_model_fit(self.r2[-1], final.fvalue, final.df_model, final.df_resid, sp_stats.f.sf(final.fvalue, final.df_model, final.df_resid))}")

        # Interaction term
        idx_xw = self._param_names.index(f"{self.x_name}_c \u00d7 {self.w_name}_c")
        lines.append("Interaction:")
        lines.append(f"  {report_regression_coeff(self.interaction_b, float(final.bse.iloc[idx_xw]), float(final.tvalues.iloc[idx_xw]), self.interaction_p, final.df_resid)}")

        # ΔR² for interaction step
        if len(self.delta_r2) > 1:
            dr2 = self.delta_r2[-1]
            df_val, df_p = self.delta_f[-1]
            lines.append(f"  {report_r2_change(dr2, df_val, 1, final.df_resid, df_p)}")

        # Simple slopes
        lines.append("Simple slopes:")
        for ss in self.simple_slopes:
            lines.append(f"  {report_simple_slope(ss.w_label, ss.b, ss.se, ss.t, ss.df, ss.p, ss.ci_lower, ss.ci_upper)}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ols(y, X) -> sm.OLS:
    """Fit OLS with a constant.  *X* may be a DataFrame (preferred)."""
    X_c = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_c).fit()


def _delta_f_test(
    r2_full: float,
    r2_reduced: float,
    df_full_resid: int,
    p_added: int,
    n: int,
) -> tuple[float, float]:
    """F-test for ΔR²."""
    if p_added == 0:
        return (0.0, 1.0)
    delta_r2 = r2_full - r2_reduced
    f_val = (delta_r2 / p_added) / ((1 - r2_full) / df_full_resid)
    p_val = 1 - sp_stats.f.cdf(f_val, p_added, df_full_resid)
    return (f_val, p_val)


def _simple_slope(
    b_x: float,
    b_xw: float,
    w_val: float,
    cov_matrix: np.ndarray,
    idx_x: int,
    idx_xw: int,
    df_resid: float,
    label: str,
) -> SimpleSlope:
    """Compute one simple slope of X on Y at a given value of W."""
    b = b_x + b_xw * w_val
    # Var(b_x + b_xw * w) = Var(b_x) + w²·Var(b_xw) + 2w·Cov(b_x, b_xw)
    var_slope = (
        cov_matrix[idx_x, idx_x]
        + w_val ** 2 * cov_matrix[idx_xw, idx_xw]
        + 2 * w_val * cov_matrix[idx_x, idx_xw]
    )
    se = np.sqrt(max(var_slope, 0.0))
    t_val = b / se if se > 0 else np.inf
    p_val = 2 * sp_stats.t.sf(abs(t_val), df_resid)
    t_crit = sp_stats.t.ppf(0.975, df_resid)
    ci_lo = b - t_crit * se
    ci_hi = b + t_crit * se
    return SimpleSlope(
        w_label=label,
        w_value=w_val,
        b=b,
        se=se,
        t=t_val,
        df=df_resid,
        p=p_val,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
    )


def _johnson_neyman(
    b_x: float,
    b_xw: float,
    cov_matrix: np.ndarray,
    idx_x: int,
    idx_xw: int,
    df_resid: float,
    w_values: np.ndarray,
) -> JohnsonNeymanResult:
    """Compute Johnson–Neyman regions of significance.

    Identifies the value(s) of *W* where the conditional effect of *X*
    on *Y* transitions between significant and non-significant at
    α = .05 (two-tailed).
    """
    t_crit = sp_stats.t.ppf(0.975, df_resid)

    w_min, w_max = float(w_values.min()), float(w_values.max())
    w_grid = np.linspace(w_min, w_max, 1000)

    effects = []
    for w in w_grid:
        ss = _simple_slope(b_x, b_xw, w, cov_matrix, idx_x, idx_xw, df_resid, "")
        effects.append({
            "w": w, "b": ss.b, "se": ss.se, "t": ss.t,
            "p": ss.p, "ci_lo": ss.ci_lower, "ci_hi": ss.ci_upper,
        })
    effects_df = pd.DataFrame(effects)

    # Find boundaries where |t| crosses t_crit
    sig = np.abs(effects_df["t"].values)
    crossings = np.where(np.diff(np.sign(sig - t_crit)))[0]
    boundaries: list[float] = []
    for c in crossings:
        # Linear interpolation between grid points
        w1, w2 = w_grid[c], w_grid[c + 1]
        s1, s2 = sig[c] - t_crit, sig[c + 1] - t_crit
        if s2 - s1 != 0:
            w_boundary = w1 - s1 * (w2 - w1) / (s2 - s1)
        else:
            w_boundary = (w1 + w2) / 2
        boundaries.append(float(w_boundary))

    n_total = len(w_values)
    pct_below = [float(np.mean(w_values < b) * 100) for b in boundaries]
    pct_above = [float(np.mean(w_values > b) * 100) for b in boundaries]

    return JohnsonNeymanResult(
        boundaries=boundaries,
        w_range=(w_min, w_max),
        pct_below=pct_below,
        pct_above=pct_above,
        conditional_effects=effects_df,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Table formatter
# ═══════════════════════════════════════════════════════════════════════════

def _format_regression_table(
    result: ModerationResult,
    decimals: int = 2,
) -> str:
    """Build a plain-text APA hierarchical regression table."""
    n_steps = len(result.steps)
    lines: list[str] = []

    lines.append("Table 2")
    lines.append(
        f"Hierarchical Regression Results for the Moderating Effect of "
        f"{result.w_name} on the {result.x_name}\u2013{result.y_name} Relationship"
    )

    # Collect all predictor names across steps
    all_preds: list[str] = []
    for step_df in result.step_summaries:
        for p in step_df.index:
            if p not in all_preds and p != "const":
                all_preds.append(p)

    # Column widths
    var_w = max(24, max(len(p) for p in all_preds) + 2)
    col_w = 10

    # Build header
    step_labels = [f"Step {i + 1}" for i in range(n_steps)]
    # Each step has b and SE sub-columns
    total_w = var_w + n_steps * 2 * col_w
    rule = "\u2500" * total_w

    lines.append(rule)

    # First header row: step labels spanning two columns each
    header1 = " " * var_w
    for sl in step_labels:
        header1 += sl.center(2 * col_w)
    lines.append(header1)

    # Second header row: b / SE
    header2 = " " * var_w
    for _ in range(n_steps):
        header2 += "b".rjust(col_w) + "SE".rjust(col_w)
    lines.append(header2)
    lines.append(rule)

    # Data rows
    for pred in all_preds:
        row = pred.ljust(var_w)
        for i, step_df in enumerate(result.step_summaries):
            if pred in step_df.index:
                b_val = step_df.loc[pred, "b"]
                se_val = step_df.loc[pred, "se"]
                p_val = step_df.loc[pred, "p"]
                stars = significance_stars(p_val)
                b_str = fmt_number(b_val, stat_type="b", decimals=decimals) + stars
                se_str = fmt_number(se_val, stat_type="se", decimals=decimals)
                row += b_str.rjust(col_w) + se_str.rjust(col_w)
            else:
                row += " " * (2 * col_w)
        lines.append(row)

    # R², ΔR², F, ΔF rows
    lines.append("")
    row_r2 = "R\u00b2".ljust(var_w)
    row_dr2 = "\u0394R\u00b2".ljust(var_w)
    row_f = "F".ljust(var_w)
    row_df = "\u0394F".ljust(var_w)

    for i in range(n_steps):
        model = result.steps[i]
        r2_str = fmt_r2(result.r2[i]) + significance_stars(
            sp_stats.f.sf(model.fvalue, model.df_model, model.df_resid)
        )
        row_r2 += r2_str.rjust(2 * col_w)

        if i == 0:
            row_dr2 += " ".rjust(2 * col_w)
            row_df += " ".rjust(2 * col_w)
        else:
            dr2_str = fmt_r2(result.delta_r2[i]) + significance_stars(result.delta_f[i][1])
            row_dr2 += dr2_str.rjust(2 * col_w)
            df_str = fmt_number(result.delta_f[i][0], stat_type="f", decimals=decimals)
            df_p_stars = significance_stars(result.delta_f[i][1])
            row_df += (df_str + df_p_stars).rjust(2 * col_w)

        f_str = fmt_number(model.fvalue, stat_type="f", decimals=decimals)
        f_p_stars = significance_stars(
            sp_stats.f.sf(model.fvalue, model.df_model, model.df_resid)
        )
        row_f += (f_str + f_p_stars).rjust(2 * col_w)

    lines.append(row_r2)
    lines.append(row_dr2)
    lines.append(row_f)
    lines.append(row_df)

    lines.append(rule)

    # Note
    lines.append(
        f"Note. N = {result.n}. Unstandardised regression coefficients are"
        " reported. Variables were mean-centred before computing the"
        " interaction term."
    )
    lines.append("*p < .05. **p < .01.")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Interaction plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_interaction(
    result: ModerationResult,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    w_label: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (6.5, 4.5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Two-way interaction plot at ±1 *SD* of the moderator.

    Follows APA 7th edition figure formatting:
      - Bold "Figure X" number, italic title
      - Clean axes, no unnecessary gridlines
      - Direct line labels or legend
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    model = result.steps[-1]  # final step with interaction
    params = model.params
    param_names = result._param_names

    x_name_c = result.x_name + "_c"
    w_name_c = result.w_name + "_c"
    xw_name = f"{result.x_name}_c × {result.w_name}_c"

    b0 = float(params.iloc[param_names.index("const")])
    b_x = float(params.iloc[param_names.index(x_name_c)])
    b_w = float(params.iloc[param_names.index(w_name_c)])
    b_xw = float(params.iloc[param_names.index(xw_name)])

    # Controls are at their means (= 0 since centred), so only
    # intercept + main effects + interaction contribute.

    # Reconstruct SDs from the data used in the model
    exog = np.asarray(model.model.exog)
    x_idx = param_names.index(x_name_c)
    w_idx = param_names.index(w_name_c)
    x_vals = exog[:, x_idx]
    w_vals = exog[:, w_idx]

    x_sd = np.std(x_vals, ddof=1)
    w_sd = np.std(w_vals, ddof=1)

    x_lo, x_hi = -x_sd, x_sd
    x_range = np.array([x_lo, x_hi])

    # Plot lines for W at -1 SD, mean, +1 SD
    w_levels = [(-w_sd, f"\u22121 SD {result.w_name}"),
                (0, f"Mean {result.w_name}"),
                (w_sd, f"+1 SD {result.w_name}")]

    # Controls are evaluated at their means (= 0 when centred),
    # so they contribute nothing to the predicted lines.

    line_styles = ["--", "-", "-."]
    colors = ["#2166ac", "#333333", "#b2182b"]  # blue, black, red

    for (w_val, w_lbl), ls, color in zip(w_levels, line_styles, colors):
        y_hat = b0 + b_x * x_range + b_w * w_val + b_xw * x_range * w_val
        ax.plot(x_range, y_hat, ls, color=color, linewidth=2, label=w_lbl)

    # Formatting
    ax.set_xlabel(x_label or result.x_name, fontsize=11)
    ax.set_ylabel(y_label or result.y_name, fontsize=11)

    ax.set_xticks([-x_sd, 0, x_sd])
    ax.set_xticklabels(["\u22121 SD", "Mean", "+1 SD"])

    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title:
        fig.suptitle(title, fontstyle="italic", fontsize=12, y=1.02)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Johnson–Neyman plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_johnson_neyman(
    result: ModerationResult,
    w_label: Optional[str] = None,
    figsize: tuple[float, float] = (7, 4.5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """Johnson–Neyman plot: conditional effect across all values of *W*."""
    if result.jn is None:
        raise ValueError("Johnson–Neyman was not computed.")

    jn = result.jn
    df = jn.conditional_effects

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(df["w"], df["b"], color="#333333", linewidth=2, label="Conditional effect")
    ax.fill_between(
        df["w"], df["ci_lo"], df["ci_hi"],
        alpha=0.15, color="#4393c3", label="95% CI",
    )
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")

    # Shade regions of significance
    for bnd in jn.boundaries:
        ax.axvline(bnd, color="#d6604d", linewidth=1.2, linestyle=":")

    ax.set_xlabel(w_label or result.w_name, fontsize=11)
    ax.set_ylabel(f"Conditional effect of {result.x_name} on {result.y_name}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def moderation_analysis(
    data: pd.DataFrame,
    x: str,
    w: str,
    y: str,
    controls: Optional[Sequence[str]] = None,
    jn: bool = True,
    decimals: int = 2,
) -> ModerationResult:
    """Run a full moderation (interaction) analysis.

    Performs hierarchical OLS regression in three steps:

      1. Control variables only (skipped when *controls* is empty).
      2. Mean-centred predictor (*X*) and moderator (*W*).
      3. Product term (*X* × *W*).

    Then probes the interaction via simple slopes at ±1 *SD* of *W* and,
    optionally, the Johnson–Neyman technique.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataset.
    x : str
        Column name of the predictor (independent variable).
    w : str
        Column name of the moderator.
    y : str
        Column name of the outcome (dependent variable).
    controls : sequence of str, optional
        Column names of control / covariate variables entered in Step 1.
    jn : bool
        Whether to compute the Johnson–Neyman regions of significance
        (default ``True``).
    decimals : int
        Decimal places for the formatted table (default 2).

    Returns
    -------
    ModerationResult
        Full results including models, simple slopes, Johnson–Neyman,
        formatted table, and plotting methods.
    """
    controls = list(controls) if controls else []

    # --- Validate --------------------------------------------------------
    if x == w:
        raise ValueError(
            f"Predictor and moderator must be different variables, got x={x!r} and w={w!r}."
        )
    if x == y or w == y:
        raise ValueError("Outcome variable must differ from predictor and moderator.")

    needed = [x, w, y] + controls
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")

    n_original = len(data)
    df = data[needed].dropna().copy()
    n = len(df)
    n_dropped = n_original - n
    if n_dropped > 0:
        warnings.warn(
            f"{n_dropped} observation(s) dropped due to missing values "
            f"(N reduced from {n_original} to {n}).",
            stacklevel=2,
        )
    if n < len(controls) + 4:
        raise ValueError(
            f"Insufficient observations (N = {n}) after dropping missing values."
        )

    y_arr = df[y].values.astype(float)

    # --- Mean-centre all continuous predictors (Aiken & West, 1991) --------
    x_mean = df[x].mean()
    w_mean = df[w].mean()
    df[f"{x}_c"] = df[x] - x_mean
    df[f"{w}_c"] = df[w] - w_mean
    df[f"{x}_c × {w}_c"] = df[f"{x}_c"] * df[f"{w}_c"]

    x_c = f"{x}_c"
    w_c = f"{w}_c"
    xw = f"{x}_c × {w}_c"

    # Mean-centre continuous control variables for interpretability.
    # Binary (0/1) controls are left uncentred so the intercept remains
    # interpretable as the expected value when the binary variable is 0.
    controls_c: list[str] = []
    for c in controls:
        vals = df[c]
        unique = vals.dropna().unique()
        is_binary = (len(unique) <= 2) and set(unique).issubset({0, 0.0, 1, 1.0})
        if is_binary:
            controls_c.append(c)  # keep original column
        else:
            cname = f"{c}_c"
            df[cname] = vals - vals.mean()
            controls_c.append(cname)

    # --- Hierarchical steps ----------------------------------------------
    steps_models = []
    step_summaries: list[pd.DataFrame] = []
    r2_list: list[float] = []
    delta_r2_list: list[float] = []
    delta_f_list: list[tuple] = []

    # Determine step compositions
    step_preds: list[list[str]] = []
    if controls_c:
        step_preds.append(controls_c)                          # Step 1: controls
        step_preds.append(controls_c + [x_c, w_c])             # Step 2: main effects
        step_preds.append(controls_c + [x_c, w_c, xw])         # Step 3: interaction
    else:
        step_preds.append([x_c, w_c])                           # Step 1: main effects
        step_preds.append([x_c, w_c, xw])                       # Step 2: interaction

    prev_r2 = 0.0
    prev_df_model = 0
    for i, preds in enumerate(step_preds):
        X_df = df[preds].astype(float)
        model = _ols(y_arr, X_df)
        steps_models.append(model)

        # Coefficient summary
        param_names = ["const"] + preds
        summary_rows = []
        for j, pname in enumerate(param_names):
            summary_rows.append({
                "predictor": pname,
                "b": model.params.iloc[j],
                "se": model.bse.iloc[j],
                "t": model.tvalues.iloc[j],
                "p": model.pvalues.iloc[j],
            })
        sdf = pd.DataFrame(summary_rows).set_index("predictor")
        step_summaries.append(sdf)

        r2 = model.rsquared
        r2_list.append(r2)

        dr2 = r2 - prev_r2
        delta_r2_list.append(dr2)

        p_added = len(preds) - (len(step_preds[i - 1]) if i > 0 else 0)
        df_test = _delta_f_test(r2, prev_r2, int(model.df_resid), p_added, n)
        delta_f_list.append(df_test)

        prev_r2 = r2
        prev_df_model = int(model.df_model)

    # --- Simple slopes at -1 SD, mean, +1 SD of W -----------------------
    final_model = steps_models[-1]
    final_preds = step_preds[-1]
    param_names_final = ["const"] + final_preds
    # Convert to numpy for numeric indexing in simple-slope / J-N code
    cov = np.asarray(final_model.cov_params())

    idx_x = param_names_final.index(x_c)
    idx_xw = param_names_final.index(xw)

    b_x = float(final_model.params.iloc[idx_x])
    b_xw = float(final_model.params.iloc[idx_xw])

    w_sd = df[w_c].std(ddof=1)
    df_resid = final_model.df_resid

    slopes = [
        _simple_slope(b_x, b_xw, -w_sd, cov, idx_x, idx_xw, df_resid, "\u22121 SD"),
        _simple_slope(b_x, b_xw, 0.0, cov, idx_x, idx_xw, df_resid, "Mean"),
        _simple_slope(b_x, b_xw, w_sd, cov, idx_x, idx_xw, df_resid, "+1 SD"),
    ]

    # --- Johnson–Neyman --------------------------------------------------
    jn_result = None
    if jn:
        jn_result = _johnson_neyman(
            b_x, b_xw, cov, idx_x, idx_xw, df_resid, df[w_c].values,
        )

    # --- Build result object ---------------------------------------------
    interaction_b = b_xw
    interaction_p = float(final_model.pvalues.iloc[idx_xw])

    mod_result = ModerationResult(
        steps=steps_models,
        step_summaries=step_summaries,
        r2=r2_list,
        delta_r2=delta_r2_list,
        delta_f=delta_f_list,
        simple_slopes=slopes,
        jn=jn_result,
        table_str="",  # filled below
        interaction_b=interaction_b,
        interaction_p=interaction_p,
        n=n,
        x_name=x,
        w_name=w,
        y_name=y,
        _param_names=param_names_final,
    )

    mod_result.table_str = _format_regression_table(mod_result, decimals=decimals)
    return mod_result
