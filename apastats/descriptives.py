"""
Descriptive statistics and intercorrelation table ("Table 1").

Produces the standard Journal of Applied Psychology Table 1 with:
  - Numbered variable rows
  - *M* and *SD* columns (leading zero, 2 decimals)
  - Lower-triangular intercorrelation matrix
  - Cronbach's alpha on the diagonal in parentheses
  - Significance stars on correlations (* p < .05, ** p < .01)
  - APA 7th edition horizontal-rule-only formatting

References
----------
APA (2020). *Publication Manual of the APA* (7th ed.).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from apastats.formatting import (
    fmt_number,
    fmt_correlation,
    significance_stars,
    table_note,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DescriptivesResult:
    """Container for descriptive-statistics results.

    Attributes
    ----------
    means : pd.Series
        Variable means.
    sds : pd.Series
        Variable standard deviations.
    correlations : pd.DataFrame
        Full symmetric correlation matrix (*r* values).
    p_values : pd.DataFrame
        Two-tailed *p* values for each correlation.
    n : int
        Sample size used for correlations.
    alphas : dict[str, float | None]
        Cronbach's alpha for each variable (``None`` when not applicable).
    variable_labels : list[str]
        Display labels for the variables.
    table_str : str
        Ready-to-use APA-formatted plain-text table.
    table_df : pd.DataFrame
        Machine-readable DataFrame of the formatted table (useful for
        export to Word / LaTeX).
    """

    means: pd.Series
    sds: pd.Series
    correlations: pd.DataFrame
    p_values: pd.DataFrame
    n: int
    alphas: Dict[str, Optional[float]]
    variable_labels: List[str]
    table_str: str = ""
    table_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __repr__(self) -> str:  # noqa: D105
        return self.table_str


# ---------------------------------------------------------------------------
# Cronbach's alpha
# ---------------------------------------------------------------------------

def cronbach_alpha(item_scores: np.ndarray) -> float:
    """Compute Cronbach's alpha for a set of item scores.

    Parameters
    ----------
    item_scores : array-like, shape (n_observations, n_items)
        Matrix where each column is an item.

    Returns
    -------
    float
        Cronbach's alpha coefficient.
    """
    item_scores = np.asarray(item_scores, dtype=float)
    # Drop rows with any NaN
    mask = ~np.isnan(item_scores).any(axis=1)
    item_scores = item_scores[mask]
    n_items = item_scores.shape[1]
    if n_items < 2:
        raise ValueError("Cronbach's alpha requires at least 2 items.")
    item_vars = item_scores.var(axis=0, ddof=1)
    total_var = item_scores.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return float("nan")
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha)


# ---------------------------------------------------------------------------
# Pairwise correlations with p-values
# ---------------------------------------------------------------------------

def _pairwise_corr(
    df: pd.DataFrame,
    method: str = "pearson",
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Compute pairwise correlations and two-tailed *p* values.

    Uses **pairwise complete observations** (listwise deletion would
    discard too much data in practice).  Returns the minimum pairwise *N*
    as the reported sample size.
    """
    cols = df.columns
    k = len(cols)
    r_mat = np.ones((k, k))
    p_mat = np.zeros((k, k))
    n_max = len(df)
    n_min = n_max
    n_pairs: dict[tuple[int, int], int] = {}

    for i in range(k):
        for j in range(i + 1, k):
            pair = df[[cols[i], cols[j]]].dropna()
            n_pair = len(pair)
            n_pairs[(i, j)] = n_pair
            n_min = min(n_min, n_pair)
            if n_pair < 3:
                r_mat[i, j] = r_mat[j, i] = np.nan
                p_mat[i, j] = p_mat[j, i] = np.nan
                warnings.warn(
                    f"Correlation between '{cols[i]}' and '{cols[j]}' could not "
                    f"be computed (pairwise N = {n_pair} < 3).",
                    stacklevel=3,
                )
                continue
            if method == "pearson":
                r, p = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
            elif method == "spearman":
                r, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            else:
                raise ValueError(f"Unknown method: {method}")
            r_mat[i, j] = r_mat[j, i] = r
            p_mat[i, j] = p_mat[j, i] = p

    # Warn if any pairwise N differs from the full sample size
    if n_pairs and n_min < n_max:
        n_vals = list(n_pairs.values())
        warnings.warn(
            f"Pairwise N ranges from {min(n_vals)} to {max(n_vals)} "
            f"(full sample N = {n_max}). The table reports the minimum "
            f"(N = {n_min}). Consider noting variable pairwise Ns in "
            "the table footnote.",
            stacklevel=2,
        )

    r_df = pd.DataFrame(r_mat, index=cols, columns=cols)
    p_df = pd.DataFrame(p_mat, index=cols, columns=cols)
    return r_df, p_df, n_min


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------

def _build_table(
    means: pd.Series,
    sds: pd.Series,
    r_df: pd.DataFrame,
    p_df: pd.DataFrame,
    alphas: Dict[str, Optional[float]],
    labels: List[str],
    n: int,
    decimals: int,
    note_extra: Optional[List[str]],
) -> tuple[str, pd.DataFrame]:
    """Build the APA-formatted plain-text table and a DataFrame version."""
    k = len(labels)

    # --- Build DataFrame version ---
    rows: list[dict] = []
    for i in range(k):
        var = r_df.index[i]
        row: dict = {
            "Variable": f"{i + 1}. {labels[i]}",
            "M": fmt_number(means[var], stat_type="m", decimals=decimals),
            "SD": fmt_number(sds[var], stat_type="sd", decimals=decimals),
        }
        for j in range(k):
            col_label = str(j + 1)
            if j == i:
                # Diagonal: alpha in parentheses or em-dash
                a = alphas.get(var)
                if a is not None:
                    row[col_label] = f"({fmt_number(a, stat_type='alpha', decimals=decimals)})"
                else:
                    row[col_label] = "\u2014"  # em-dash
            elif j < i:
                # Lower triangle: correlation with stars
                row[col_label] = fmt_correlation(
                    r_df.iloc[i, j], p_df.iloc[i, j], decimals=decimals
                )
            else:
                # Upper triangle: blank
                row[col_label] = ""
        rows.append(row)

    table_df = pd.DataFrame(rows)

    # --- Build plain-text version ---
    # Determine column widths
    col_keys = list(table_df.columns)
    widths: dict[str, int] = {}
    for c in col_keys:
        max_w = max(len(c), table_df[c].astype(str).str.len().max())
        widths[c] = max_w + 2  # padding

    # Ensure minimum widths for readability
    widths["Variable"] = max(widths["Variable"], 24)
    for c in col_keys[1:]:
        widths[c] = max(widths[c], 8)

    total_width = sum(widths.values())
    rule = "\u2500" * total_width  # thin horizontal rule

    lines: list[str] = []
    lines.append("Table 1")
    lines.append(
        "Means, Standard Deviations, and Intercorrelations Among Study Variables"
    )
    lines.append(rule)

    # Header row
    header = ""
    for c in col_keys:
        header += c.rjust(widths[c]) if c != "Variable" else c.ljust(widths[c])
    lines.append(header)
    lines.append(rule)

    # Data rows
    for _, r in table_df.iterrows():
        line = ""
        for c in col_keys:
            val = str(r[c])
            if c == "Variable":
                line += val.ljust(widths[c])
            else:
                line += val.rjust(widths[c])
        lines.append(line)

    lines.append(rule)

    # Notes
    has_alpha = any(v is not None for v in alphas.values())
    lines.append(table_note(n, extra_lines=note_extra, alpha_on_diagonal=has_alpha))

    return "\n".join(lines), table_df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def descriptives_table(
    data: pd.DataFrame,
    variables: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    alphas: Optional[Dict[str, Union[float, np.ndarray, pd.DataFrame, List[str]]]] = None,
    method: str = "pearson",
    decimals: int = 2,
    note_extra: Optional[List[str]] = None,
) -> DescriptivesResult:
    """Build a JAP-style descriptive statistics and intercorrelation table.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataset.
    variables : sequence of str
        Column names to include in the table, in desired display order.
    labels : sequence of str, optional
        Human-readable labels for each variable.  Defaults to the column
        names.
    alphas : dict, optional
        Cronbach's alpha for each variable.  Values may be:

        * A **float** — used directly as the pre-computed alpha.
        * A **2-D array / DataFrame** of item-level responses — alpha is
          computed automatically via :func:`cronbach_alpha`.
        * A **list of column names** present in *data* — item scores are
          extracted and alpha is computed.
        * ``None`` for variables without a reliability estimate (an
          em-dash is shown on the diagonal).

        Variables absent from the dict default to ``None`` (em-dash).
    method : {"pearson", "spearman"}
        Correlation method.
    decimals : int
        Decimal places for *M*, *SD*, *r*, and alpha (default 2).
    note_extra : list of str, optional
        Additional lines for the table note (inserted before the
        probability note).

    Returns
    -------
    DescriptivesResult
        Full results object with formatted table and raw statistics.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> from apastats import descriptives_table
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "satisfaction": rng.normal(3.5, 0.8, 200),
    ...     "commitment":  rng.normal(3.8, 0.9, 200),
    ...     "turnover":    rng.normal(2.1, 1.0, 200),
    ... })
    >>> res = descriptives_table(df, ["satisfaction", "commitment", "turnover"])
    >>> print(res)
    """
    variables = list(variables)
    if labels is None:
        labels = list(variables)
    else:
        labels = list(labels)
    if len(labels) != len(variables):
        raise ValueError("Length of labels must match length of variables.")

    # Subset and validate
    missing = [v for v in variables if v not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in data: {missing}")
    df = data[variables].copy()

    # Compute descriptives
    means = df.mean()
    sds = df.std(ddof=1)

    # Compute correlations
    r_df, p_df, n_min = _pairwise_corr(df, method=method)

    # Resolve alphas
    resolved_alphas: Dict[str, Optional[float]] = {}
    alphas = alphas or {}
    for var in variables:
        raw = alphas.get(var)
        if raw is None:
            resolved_alphas[var] = None
        elif isinstance(raw, (int, float)):
            resolved_alphas[var] = float(raw)
        elif isinstance(raw, list):
            # List of column names -> extract items and compute
            item_data = data[raw].values
            resolved_alphas[var] = cronbach_alpha(item_data)
        elif isinstance(raw, (np.ndarray, pd.DataFrame)):
            arr = np.asarray(raw)
            resolved_alphas[var] = cronbach_alpha(arr)
        else:
            raise TypeError(
                f"Unsupported alpha value type for '{var}': {type(raw)}"
            )

    table_str, table_df = _build_table(
        means, sds, r_df, p_df, resolved_alphas, labels, n_min, decimals, note_extra
    )

    return DescriptivesResult(
        means=means,
        sds=sds,
        correlations=r_df,
        p_values=p_df,
        n=n_min,
        alphas=resolved_alphas,
        variable_labels=labels,
        table_str=table_str,
        table_df=table_df,
    )
