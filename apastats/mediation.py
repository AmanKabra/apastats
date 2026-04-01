"""
Mediation analysis following JAP / APA 7th edition norms.

Implements:
  1. Single-mediator model (PROCESS Model 4 equivalent).
  2. Parallel multiple-mediator model.
  3. Percentile bootstrap confidence intervals for indirect effects.
  4. APA-formatted results table.
  5. Path-diagram figure.

Key conventions
---------------
* Bootstrap with 10 000 resamples by default (JAP standard).
* **Percentile** bootstrap CIs (recommended over bias-corrected in
  PROCESS v3+; see Hayes, 2022, p. 131).
* The indirect effect is significant when the CI **excludes zero** —
  no *t* or *p* is reported for the indirect effect.
* Unstandardised coefficients throughout.

References
----------
Hayes, A. F. (2022). *Introduction to mediation, moderation, and
    conditional process analysis* (3rd ed.). Guilford.
Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling
    strategies for assessing and comparing indirect effects in multiple
    mediator models. *Behavior Research Methods*, 40, 879–891.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from apastats.formatting import (
    fmt_number,
    fmt_p,
    fmt_ci,
    significance_stars,
)


# ═══════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PathEstimate:
    """A single OLS path coefficient."""

    label: str
    b: float
    se: float
    t: float
    p: float
    ci_lower: float
    ci_upper: float


@dataclass
class IndirectEffect:
    """Bootstrap indirect effect for one mediator."""

    mediator: str
    ab: float
    boot_se: float
    ci_lower: float
    ci_upper: float
    significant: bool  # CI excludes zero


@dataclass
class MediationResult:
    """Full mediation analysis results.

    Attributes
    ----------
    paths : dict[str, PathEstimate]
        Named path estimates.  Keys follow the convention
        ``"a1"``, ``"b1"``, ``"c"``, ``"c_prime"`` (and ``"a2"``,
        ``"b2"`` for parallel mediators).
    indirect_effects : list[IndirectEffect]
        One per mediator.
    total_indirect : IndirectEffect
        Sum of all specific indirect effects (same as the single
        indirect effect when there is only one mediator).
    direct_effect : PathEstimate
        The *c'* path.
    total_effect : PathEstimate
        The *c* path.
    r2_m : dict[str, float]
        *R*² for each mediator regression.
    r2_y : float
        *R*² for the outcome regression (full model).
    n : int
        Sample size.
    n_boot : int
        Number of bootstrap resamples.
    table_str : str
        APA-formatted plain-text results table.
    x_name : str
    y_name : str
    m_names : list[str]
    """

    paths: Dict[str, PathEstimate]
    indirect_effects: List[IndirectEffect]
    total_indirect: IndirectEffect
    direct_effect: PathEstimate
    total_effect: PathEstimate
    r2_m: Dict[str, float]
    r2_y: float
    n: int
    n_boot: int
    table_str: str
    x_name: str
    y_name: str
    m_names: List[str]

    def __repr__(self) -> str:  # noqa: D105
        return self.table_str

    def plot(self, **kwargs) -> plt.Figure:
        """Draw a path diagram."""
        return plot_path_diagram(self, **kwargs)

    def report(self) -> str:
        """Return copy-paste APA in-text strings for the mediation analysis."""
        from apastats.formatting import (
            report_regression_coeff, report_indirect_effect,
        )
        lines: list[str] = []

        # Path coefficients
        for key, pe in self.paths.items():
            lines.append(f"Path {key}: {report_regression_coeff(pe.b, pe.se, pe.t, pe.p, 0, pe.ci_lower, pe.ci_upper).replace('t(0)', f't({self.n - 2})')}")

        # Direct effect
        pe = self.direct_effect
        lines.append(f"Direct effect (c'): {report_regression_coeff(pe.b, pe.se, pe.t, pe.p, 0, pe.ci_lower, pe.ci_upper).replace('t(0)', f't({self.n - 2})')}")

        # Total effect
        pe = self.total_effect
        lines.append(f"Total effect (c): {report_regression_coeff(pe.b, pe.se, pe.t, pe.p, 0, pe.ci_lower, pe.ci_upper).replace('t(0)', f't({self.n - 2})')}")

        # Indirect effects
        for ie in self.indirect_effects:
            lines.append(report_indirect_effect(ie.ab, ie.boot_se, ie.ci_lower, ie.ci_upper, ie.mediator))
        if len(self.indirect_effects) > 1:
            ie = self.total_indirect
            lines.append(f"Total {report_indirect_effect(ie.ab, ie.boot_se, ie.ci_lower, ie.ci_upper)}")

        lines.append(f"Bootstrap samples: {self.n_boot:,}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _ols_fit(y: np.ndarray, X: np.ndarray):
    """OLS with constant; returns fitted model."""
    X_c = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_c).fit()


def _path_from_model(model, idx: int, label: str) -> PathEstimate:
    """Extract a PathEstimate from a statsmodels result at parameter index."""
    ci = model.conf_int(alpha=0.05)
    return PathEstimate(
        label=label,
        b=model.params[idx],
        se=model.bse[idx],
        t=model.tvalues[idx],
        p=model.pvalues[idx],
        ci_lower=ci[idx, 0],
        ci_upper=ci[idx, 1],
    )


def _bootstrap_indirect(
    data_x: np.ndarray,
    data_m: np.ndarray,       # shape (n, k_mediators)
    data_y: np.ndarray,
    covariates: Optional[np.ndarray],
    n_boot: int,
    seed: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap specific and total indirect effects (percentile method).

    Each resample fits the *a* and *b* path regressions and records the
    product *ab*.  Column order in the *b*-path design matrix is always
    ``[X, M1, …, Mk, cov1, …, covp]``, so ``params[2 + j]`` correctly
    indexes mediator *j* regardless of the number of covariates.

    Returns
    -------
    specific_boots : ndarray, shape (n_boot, k)
        Bootstrap distribution for each mediator's indirect effect.
    total_boots : ndarray, shape (n_boot,)
        Bootstrap distribution for the total indirect effect.
    """
    rng = np.random.default_rng(seed)
    n = len(data_x)
    k = data_m.shape[1] if data_m.ndim == 2 else 1
    if data_m.ndim == 1:
        data_m = data_m.reshape(-1, 1)

    specific_boots = np.empty((n_boot, k))
    total_boots = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        x_b = data_x[idx]
        m_b = data_m[idx]
        y_b = data_y[idx]
        cov_b = covariates[idx] if covariates is not None else None

        ab_total = 0.0
        for j in range(k):
            # a path: X (+ covariates) -> M_j
            if cov_b is not None:
                Xa = np.column_stack([x_b, cov_b])
            else:
                Xa = x_b.reshape(-1, 1)
            Xa_c = sm.add_constant(Xa, has_constant="add")
            try:
                a_model = sm.OLS(m_b[:, j], Xa_c).fit()
                a_coef = a_model.params[1]  # X coefficient
            except Exception:
                a_coef = np.nan

            # b path: M_j (+ X + covariates) -> Y
            if cov_b is not None:
                Xb = np.column_stack([x_b, m_b, cov_b])
            else:
                Xb = np.column_stack([x_b, m_b])
            Xb_c = sm.add_constant(Xb, has_constant="add")
            try:
                b_model = sm.OLS(y_b, Xb_c).fit()
                # b coef for M_j: index is 2 + j (const=0, x=1, m1=2, m2=3, ...)
                b_coef = b_model.params[2 + j]
            except Exception:
                b_coef = np.nan

            ab = a_coef * b_coef
            specific_boots[b, j] = ab
            ab_total += ab

        total_boots[b] = ab_total

    return specific_boots, total_boots


# ═══════════════════════════════════════════════════════════════════════════
# Table formatter
# ═══════════════════════════════════════════════════════════════════════════

def _format_mediation_table(result: MediationResult, decimals: int = 2) -> str:
    """Build a plain-text APA mediation results table."""
    lines: list[str] = []

    m_through = ", ".join(result.m_names)
    lines.append("Table 3")
    lines.append(
        f"Mediation Analysis Results: Indirect Effect of {result.x_name} on "
        f"{result.y_name} Through {m_through}"
    )

    col_w = {"path": 36, "b": 10, "se": 10, "t": 10, "p": 10, "ci": 20}
    total_w = sum(col_w.values())
    rule = "\u2500" * total_w

    lines.append(rule)
    header = (
        "Path / Effect".ljust(col_w["path"])
        + "b".rjust(col_w["b"])
        + "SE".rjust(col_w["se"])
        + "t".rjust(col_w["t"])
        + "p".rjust(col_w["p"])
        + "95% CI".rjust(col_w["ci"])
    )
    lines.append(header)
    lines.append(rule)

    def _row(label: str, pe: PathEstimate, suppress_t_p: bool = False) -> str:
        b_str = fmt_number(pe.b, stat_type="b", decimals=decimals)
        se_str = fmt_number(pe.se, stat_type="se", decimals=decimals)
        if suppress_t_p:
            t_str = "\u2014"
            p_str = "\u2014"
        else:
            t_str = fmt_number(pe.t, stat_type="t", decimals=decimals)
            p_str = fmt_p(pe.p)
        ci_str = fmt_ci(pe.ci_lower, pe.ci_upper, decimals=decimals, stat_type="b")
        return (
            label.ljust(col_w["path"])
            + b_str.rjust(col_w["b"])
            + se_str.rjust(col_w["se"])
            + t_str.rjust(col_w["t"])
            + p_str.rjust(col_w["p"])
            + ci_str.rjust(col_w["ci"])
        )

    # a paths
    for i, m in enumerate(result.m_names):
        key = f"a{i + 1}" if len(result.m_names) > 1 else "a"
        pe = result.paths[key]
        label = f"Path a{'_' + str(i+1) if len(result.m_names) > 1 else ''} ({result.x_name} \u2192 {m})"
        lines.append(_row(label, pe))

    # b paths
    for i, m in enumerate(result.m_names):
        key = f"b{i + 1}" if len(result.m_names) > 1 else "b"
        pe = result.paths[key]
        label = f"Path b{'_' + str(i+1) if len(result.m_names) > 1 else ''} ({m} \u2192 {result.y_name})"
        lines.append(_row(label, pe))

    # Direct effect c'
    lines.append(_row(
        f"Direct effect c\u2032 ({result.x_name} \u2192 {result.y_name})",
        result.direct_effect,
    ))

    # Total effect c
    lines.append(_row(
        f"Total effect c ({result.x_name} \u2192 {result.y_name})",
        result.total_effect,
    ))

    # Indirect effects (no t or p — significance by CI)
    lines.append("")
    for ie in result.indirect_effects:
        sig_marker = " *" if ie.significant else ""
        ci_str = fmt_ci(ie.ci_lower, ie.ci_upper, decimals=decimals, stat_type="b")
        b_str = fmt_number(ie.ab, stat_type="b", decimals=decimals)
        se_str = fmt_number(ie.boot_se, stat_type="se", decimals=decimals)
        label = f"Indirect via {ie.mediator}"
        row = (
            label.ljust(col_w["path"])
            + b_str.rjust(col_w["b"])
            + se_str.rjust(col_w["se"])
            + "\u2014".rjust(col_w["t"])
            + "\u2014".rjust(col_w["p"])
            + ci_str.rjust(col_w["ci"])
        )
        lines.append(row)

    if len(result.indirect_effects) > 1:
        ie = result.total_indirect
        ci_str = fmt_ci(ie.ci_lower, ie.ci_upper, decimals=decimals, stat_type="b")
        b_str = fmt_number(ie.ab, stat_type="b", decimals=decimals)
        se_str = fmt_number(ie.boot_se, stat_type="se", decimals=decimals)
        row = (
            "Total indirect effect".ljust(col_w["path"])
            + b_str.rjust(col_w["b"])
            + se_str.rjust(col_w["se"])
            + "\u2014".rjust(col_w["t"])
            + "\u2014".rjust(col_w["p"])
            + ci_str.rjust(col_w["ci"])
        )
        lines.append(row)

    lines.append(rule)
    lines.append(
        f"Note. N = {result.n}. Bootstrap sample size = {result.n_boot:,}. "
        "CI = confidence interval. Confidence intervals for indirect "
        "effects are percentile bootstrap confidence intervals."
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# Path diagram
# ═══════════════════════════════════════════════════════════════════════════

def plot_path_diagram(
    result: MediationResult,
    figsize: Optional[tuple[float, float]] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """Draw a mediation path diagram with coefficients.

    For single-mediator models, draws the classic triangular layout.
    For parallel mediators, fans them out vertically.
    """
    k = len(result.m_names)
    if figsize is None:
        figsize = (8, 3 + 1.5 * max(0, k - 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 10.5)
    y_top = 4 + (k - 1)
    ax.set_ylim(-1, y_top + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # Box style
    box_kw = dict(
        boxstyle="round,pad=0.4", edgecolor="#333333",
        facecolor="#f7f7f7", linewidth=1.5,
    )

    # Positions
    x_pos = (1.0, 2.0)    # X box centre
    y_pos = (9.0, 2.0)    # Y box centre

    # Mediator positions (spread vertically)
    m_positions = []
    if k == 1:
        m_positions.append((5.0, y_top - 0.5))
    else:
        for i in range(k):
            m_y = y_top - 0.5 - i * 1.5
            m_positions.append((5.0, m_y))

    # Draw boxes
    ax.text(*x_pos, result.x_name, ha="center", va="center", fontsize=11,
            fontweight="bold", bbox=box_kw)
    ax.text(*y_pos, result.y_name, ha="center", va="center", fontsize=11,
            fontweight="bold", bbox=box_kw)
    for i, m_name in enumerate(result.m_names):
        ax.text(*m_positions[i], m_name, ha="center", va="center",
                fontsize=10, fontweight="bold", bbox=box_kw)

    # Arrow style
    arrow_kw = dict(
        arrowstyle="->,head_width=0.25,head_length=0.15",
        color="#333333", linewidth=1.5,
    )

    # a paths (X -> M)
    for i in range(k):
        key = f"a{i + 1}" if k > 1 else "a"
        pe = result.paths[key]
        stars = significance_stars(pe.p)
        label = f"{fmt_number(pe.b, 'b')}{stars}"
        mid_x = (x_pos[0] + m_positions[i][0]) / 2
        mid_y = (x_pos[1] + m_positions[i][1]) / 2 + 0.3
        ax.annotate("", xy=(m_positions[i][0] - 0.8, m_positions[i][1]),
                     xytext=(x_pos[0] + 0.8, x_pos[1]),
                     arrowprops=arrow_kw)
        ax.text(mid_x, mid_y, f"a = {label}", ha="center", fontsize=9)

    # b paths (M -> Y)
    for i in range(k):
        key = f"b{i + 1}" if k > 1 else "b"
        pe = result.paths[key]
        stars = significance_stars(pe.p)
        label = f"{fmt_number(pe.b, 'b')}{stars}"
        mid_x = (m_positions[i][0] + y_pos[0]) / 2
        mid_y = (m_positions[i][1] + y_pos[1]) / 2 + 0.3
        ax.annotate("", xy=(y_pos[0] - 0.8, y_pos[1]),
                     xytext=(m_positions[i][0] + 0.8, m_positions[i][1]),
                     arrowprops=arrow_kw)
        ax.text(mid_x, mid_y, f"b = {label}", ha="center", fontsize=9)

    # c / c' path (X -> Y, direct)
    pe_c = result.total_effect
    pe_cp = result.direct_effect
    stars_c = significance_stars(pe_c.p)
    stars_cp = significance_stars(pe_cp.p)
    c_label = fmt_number(pe_c.b, "b") + stars_c
    cp_label = fmt_number(pe_cp.b, "b") + stars_cp

    ax.annotate("", xy=(y_pos[0] - 0.8, y_pos[1]),
                xytext=(x_pos[0] + 0.8, x_pos[1]),
                arrowprops=dict(**arrow_kw, linestyle="--"))
    ax.text(5.0, y_pos[1] - 0.5,
            f"c = {c_label}  (c\u2032 = {cp_label})",
            ha="center", fontsize=9)

    # Indirect effect annotation
    ie_label_parts = []
    for ie in result.indirect_effects:
        ci = fmt_ci(ie.ci_lower, ie.ci_upper, stat_type="b")
        ie_label_parts.append(
            f"Indirect{' via ' + ie.mediator if k > 1 else ''}: "
            f"ab = {fmt_number(ie.ab, 'b')}, 95% CI {ci}"
        )
    ie_text = "\n".join(ie_label_parts)
    ax.text(5.0, -0.5, ie_text, ha="center", fontsize=8.5, fontstyle="italic")

    if title:
        fig.suptitle(title, fontstyle="italic", fontsize=12, y=1.02)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def mediation_analysis(
    data: pd.DataFrame,
    x: str,
    m: str | Sequence[str],
    y: str,
    covariates: Optional[Sequence[str]] = None,
    n_boot: int = 10_000,
    ci_level: float = 0.95,
    seed: Optional[int] = None,
    decimals: int = 2,
) -> MediationResult:
    """Run a bootstrap mediation analysis (PROCESS Model 4 equivalent).

    Supports single and parallel multiple mediators.  Uses percentile
    bootstrap confidence intervals for the indirect effect(s).

    Parameters
    ----------
    data : pd.DataFrame
        Source dataset.
    x : str
        Column name of the predictor.
    m : str or sequence of str
        Column name(s) of mediator(s).  A single string gives a
        single-mediator model; a list gives a parallel-mediator model.
    y : str
        Column name of the outcome.
    covariates : sequence of str, optional
        Column names of covariates entered in all regressions.
    n_boot : int
        Number of bootstrap resamples (default 10 000).
    ci_level : float
        Confidence level for bootstrap CIs (default .95).
    seed : int, optional
        Random seed for reproducibility.
    decimals : int
        Decimal places for the formatted table (default 2).

    Returns
    -------
    MediationResult
        Full results with paths, indirect effects, formatted table, and
        path-diagram plotting method.
    """
    if isinstance(m, str):
        m_names = [m]
    else:
        m_names = list(m)
    covariates = list(covariates) if covariates else []
    k = len(m_names)

    # --- Validate --------------------------------------------------------
    if len(m_names) != len(set(m_names)):
        raise ValueError(f"Duplicate mediator names: {m_names}")
    all_vars = [x, y] + m_names + covariates
    if len(all_vars) != len(set(all_vars)):
        raise ValueError(
            "Predictor, outcome, mediators, and covariates must all be distinct variables."
        )

    needed = [x, y] + m_names + covariates
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
    if n < len(covariates) + k + 3:
        raise ValueError(
            f"Insufficient observations (N = {n}) after dropping missing values."
        )

    x_arr = df[x].values.astype(float)
    y_arr = df[y].values.astype(float)
    m_arr = df[m_names].values.astype(float)  # (n, k)
    if m_arr.ndim == 1:
        m_arr = m_arr.reshape(-1, 1)
    cov_arr = df[covariates].values.astype(float) if covariates else None

    # --- Path a regressions: X (+cov) -> each M --------------------------
    paths: Dict[str, PathEstimate] = {}
    r2_m: Dict[str, float] = {}

    for j in range(k):
        if cov_arr is not None:
            Xa = np.column_stack([x_arr, cov_arr])
        else:
            Xa = x_arr.reshape(-1, 1)
        model_a = _ols_fit(m_arr[:, j], Xa)
        key = f"a{j + 1}" if k > 1 else "a"
        paths[key] = _path_from_model(model_a, 1, key)  # idx 1 = X
        r2_m[m_names[j]] = model_a.rsquared

    # --- Path b + c' regression: X + all M (+cov) -> Y --------------------
    if cov_arr is not None:
        Xb = np.column_stack([x_arr, m_arr, cov_arr])
    else:
        Xb = np.column_stack([x_arr, m_arr])
    model_b = _ols_fit(y_arr, Xb)

    for j in range(k):
        key = f"b{j + 1}" if k > 1 else "b"
        paths[key] = _path_from_model(model_b, 2 + j, key)  # idx 2+j = M_j

    # Direct effect (c')
    direct = _path_from_model(model_b, 1, "c_prime")  # idx 1 = X
    r2_y = model_b.rsquared

    # --- Total effect: X (+cov) -> Y (without mediators) ------------------
    if cov_arr is not None:
        Xc = np.column_stack([x_arr, cov_arr])
    else:
        Xc = x_arr.reshape(-1, 1)
    model_c = _ols_fit(y_arr, Xc)
    total = _path_from_model(model_c, 1, "c")

    # --- Point estimates of indirect effects ------------------------------
    indirect_point = []
    for j in range(k):
        a_key = f"a{j + 1}" if k > 1 else "a"
        b_key = f"b{j + 1}" if k > 1 else "b"
        ab = paths[a_key].b * paths[b_key].b
        indirect_point.append(ab)
    total_ab_point = sum(indirect_point)

    # --- Bootstrap --------------------------------------------------------
    alpha = 1 - ci_level
    lo_pct = (alpha / 2) * 100
    hi_pct = (1 - alpha / 2) * 100

    specific_boots, total_boots = _bootstrap_indirect(
        x_arr, m_arr, y_arr, cov_arr, n_boot, seed,
    )

    indirect_effects: list[IndirectEffect] = []
    for j in range(k):
        boot_j = specific_boots[:, j]
        ci_lo = float(np.percentile(boot_j, lo_pct))
        ci_hi = float(np.percentile(boot_j, hi_pct))
        boot_se = float(np.std(boot_j, ddof=1))
        sig = not (ci_lo <= 0 <= ci_hi)
        indirect_effects.append(IndirectEffect(
            mediator=m_names[j],
            ab=indirect_point[j],
            boot_se=boot_se,
            ci_lower=ci_lo,
            ci_upper=ci_hi,
            significant=sig,
        ))

    # Total indirect
    total_ci_lo = float(np.percentile(total_boots, lo_pct))
    total_ci_hi = float(np.percentile(total_boots, hi_pct))
    total_boot_se = float(np.std(total_boots, ddof=1))
    total_indirect = IndirectEffect(
        mediator="Total",
        ab=total_ab_point,
        boot_se=total_boot_se,
        ci_lower=total_ci_lo,
        ci_upper=total_ci_hi,
        significant=not (total_ci_lo <= 0 <= total_ci_hi),
    )

    # --- Build result -----------------------------------------------------
    med_result = MediationResult(
        paths=paths,
        indirect_effects=indirect_effects,
        total_indirect=total_indirect,
        direct_effect=direct,
        total_effect=total,
        r2_m=r2_m,
        r2_y=r2_y,
        n=n,
        n_boot=n_boot,
        table_str="",
        x_name=x,
        y_name=y,
        m_names=m_names,
    )

    med_result.table_str = _format_mediation_table(med_result, decimals=decimals)
    return med_result
