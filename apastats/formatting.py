"""
APA 7th edition number and table formatting utilities.

Rules implemented
-----------------
* **No leading zero** for statistics bounded between -1 and +1:
  correlations (*r*), reliability (alpha), standardised coefficients (beta),
  *p* values, *R*².
* **Leading zero** for statistics that *can* exceed ±1:
  *M*, *SD*, *b* (unstandardised), *t*, *F*, *SE*, Cohen's *d*.
* Two decimal places by default throughout.
* Significance stars: ``*`` *p* < .05, ``**`` *p* < .01.
* *p* values reported exact to three decimals; very small values as "< .001".
* 95 % CI formatted as ``[lower, upper]``.

References
----------
APA (2020). *Publication Manual of the APA* (7th ed.).
"""

from __future__ import annotations

import math
from typing import Optional

# ---------------------------------------------------------------------------
# Core number formatters
# ---------------------------------------------------------------------------

# Statistics that are bounded between -1 and +1 (no leading zero)
_BOUNDED_STATS = {"r", "alpha", "beta", "p", "r2", "delta_r2", "sr2"}

# Statistics that can exceed ±1 (leading zero)
_UNBOUNDED_STATS = {"m", "sd", "b", "se", "t", "f", "d", "ci"}


def fmt_number(
    value: float,
    stat_type: str = "unbounded",
    decimals: int = 2,
) -> str:
    """Format a number following APA 7th edition rules.

    Parameters
    ----------
    value : float
        The numeric value to format.
    stat_type : str
        Either a named statistic (e.g. ``"r"``, ``"p"``, ``"m"``, ``"b"``)
        or the shorthand ``"bounded"`` / ``"unbounded"``.
    decimals : int
        Number of decimal places (default 2).

    Returns
    -------
    str
        APA-formatted string.
    """
    if math.isnan(value):
        return ""
    if math.isinf(value):
        return "\u221e" if value > 0 else "-\u221e"
    if decimals < 0:
        raise ValueError(f"decimals must be >= 0, got {decimals}")

    # Normalise negative zero to positive zero
    if value == 0.0:
        value = 0.0

    stat_lower = stat_type.lower()
    bounded = stat_lower in _BOUNDED_STATS or stat_lower == "bounded"

    formatted = f"{value:.{decimals}f}"

    if bounded:
        # Strip leading zero: "0.54" -> ".54", "-0.54" -> "-.54"
        if formatted.startswith("0."):
            formatted = formatted[1:]
        elif formatted.startswith("-0."):
            formatted = "-" + formatted[2:]

    return formatted


def fmt_p(p_value: float) -> str:
    """Format a *p* value following APA 7th edition.

    - Exact to three decimals for values >= .001.
    - ``"< .001"`` for very small values.
    - Never ``".000"`` or ``"0.000"``.
    """
    if math.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "< .001"
    return fmt_number(p_value, stat_type="p", decimals=3)


def significance_stars(p_value: float) -> str:
    """Return significance stars per JAP convention.

    ``**`` for *p* < .01, ``*`` for *p* < .05, else empty string.
    """
    if math.isnan(p_value):
        return ""
    if p_value < .01:
        return "**"
    if p_value < .05:
        return "*"
    return ""


def fmt_ci(lower: float, upper: float, decimals: int = 2, stat_type: str = "unbounded") -> str:
    """Format a confidence interval as ``[lower, upper]`` per APA 7th."""
    lo = fmt_number(lower, stat_type=stat_type, decimals=decimals)
    hi = fmt_number(upper, stat_type=stat_type, decimals=decimals)
    return f"[{lo}, {hi}]"


def fmt_r2(value: float, decimals: int = 2) -> str:
    """Format *R*² (bounded, no leading zero)."""
    return fmt_number(value, stat_type="r2", decimals=decimals)


def fmt_correlation(r: float, p: float, decimals: int = 2) -> str:
    """Format a correlation with significance stars (for Table 1)."""
    stars = significance_stars(p)
    return f"{fmt_number(r, stat_type='r', decimals=decimals)}{stars}"


# ---------------------------------------------------------------------------
# Table-level helpers
# ---------------------------------------------------------------------------

_TABLE_NOTE_STARS = "*p < .05. **p < .01."


def table_note(
    n: int,
    extra_lines: Optional[list[str]] = None,
    alpha_on_diagonal: bool = True,
) -> str:
    """Build a standard APA table note block.

    Parameters
    ----------
    n : int
        Sample size.
    extra_lines : list[str], optional
        Additional note lines inserted between the general note and the
        probability note.
    alpha_on_diagonal : bool
        If True, includes the standard reliability note.

    Returns
    -------
    str
        Multi-line note string.
    """
    parts: list[str] = []
    general = f"Note. N = {n}."
    if alpha_on_diagonal:
        general += (
            " Reliability coefficients (Cronbach's alpha) appear on the"
            " diagonal in parentheses."
        )
    parts.append(general)
    if extra_lines:
        parts.extend(extra_lines)
    parts.append(_TABLE_NOTE_STARS)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# In-text APA reporting strings
# ---------------------------------------------------------------------------

def report_regression_coeff(
    b: float,
    se: float,
    t: float,
    p: float,
    df: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
    standardised: bool = False,
) -> str:
    """Copy-paste APA string for a regression coefficient.

    Examples
    --------
    >>> report_regression_coeff(0.25, 0.08, 3.13, .002, 225, 0.09, 0.41)
    'b = 0.25, SE = 0.08, t(225) = 3.13, p = .002, 95% CI [0.09, 0.41]'
    """
    if standardised:
        coeff_str = f"\u03b2 = {fmt_number(b, 'beta')}"
    else:
        coeff_str = f"b = {fmt_number(b, 'b')}"
    parts = [
        coeff_str,
        f"SE = {fmt_number(se, 'se')}",
        f"t({df:.0f}) = {fmt_number(t, 't')}",
        f"p {fmt_p(p)}",
    ]
    if ci_lower is not None and ci_upper is not None:
        stat = "beta" if standardised else "b"
        parts.append(f"95% CI {fmt_ci(ci_lower, ci_upper, stat_type=stat)}")
    return ", ".join(parts)


def report_model_fit(
    r2: float,
    f_val: float,
    df_model: float,
    df_resid: float,
    p: float,
) -> str:
    """Copy-paste APA string for overall model fit.

    Example: ``R² = .22, F(3, 225) = 21.15, p < .001``
    """
    return (
        f"R\u00b2 = {fmt_r2(r2)}, "
        f"F({df_model:.0f}, {df_resid:.0f}) = {fmt_number(f_val, 'f')}, "
        f"p {fmt_p(p)}"
    )


def report_r2_change(
    delta_r2: float,
    delta_f: float,
    df_num: float,
    df_denom: float,
    p: float,
) -> str:
    """Copy-paste APA string for R² change.

    Example: ``ΔR² = .03, ΔF(1, 224) = 8.92, p = .003``
    """
    return (
        f"\u0394R\u00b2 = {fmt_r2(delta_r2)}, "
        f"\u0394F({df_num:.0f}, {df_denom:.0f}) = {fmt_number(delta_f, 'f')}, "
        f"p {fmt_p(p)}"
    )


def report_indirect_effect(
    ab: float,
    se: float,
    ci_lower: float,
    ci_upper: float,
    mediator: Optional[str] = None,
) -> str:
    """Copy-paste APA string for a bootstrap indirect effect.

    Example: ``ab = 0.17, SE = 0.07, 95% CI [0.04, 0.32]``
    """
    prefix = f"indirect effect via {mediator}: " if mediator else ""
    return (
        f"{prefix}ab = {fmt_number(ab, 'b')}, "
        f"SE = {fmt_number(se, 'se')}, "
        f"95% CI {fmt_ci(ci_lower, ci_upper, stat_type='b')}"
    )


def report_simple_slope(
    w_label: str,
    b: float,
    se: float,
    t: float,
    df: float,
    p: float,
    ci_lower: float,
    ci_upper: float,
) -> str:
    """Copy-paste APA string for a simple slope.

    Example: ``At +1 SD of W: b = 0.45, SE = 0.06, t(394) = 7.20,
    p < .001, 95% CI [0.33, 0.57]``
    """
    return (
        f"At {w_label}: "
        f"b = {fmt_number(b, 'b')}, "
        f"SE = {fmt_number(se, 'se')}, "
        f"t({df:.0f}) = {fmt_number(t, 't')}, "
        f"p {fmt_p(p)}, "
        f"95% CI {fmt_ci(ci_lower, ci_upper, stat_type='b')}"
    )
