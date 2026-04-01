# apastats

**APA 7th edition compliant statistical analyses for organizational science research.**

Python is increasingly the language of choice for analytical pipelines across the social sciences, yet there is no Python package purpose-built for the statistical reporting conventions that journals like the *Journal of Applied Psychology* (JAP) demand. Researchers who want to stay in Python are forced to cobble together output from statsmodels, scipy, and manual formatting — or switch to SPSS/R just for the final tables. `apastats` eliminates that gap entirely.

Beyond the missing-tooling problem, there is a deeper methodological concern: **transparency**. Most statistical software applies default settings silently — bootstrap resamples, confidence interval methods, centering decisions, estimation procedures — and scholars have no visibility into what was actually run. When defaults go unreported, readers implicitly treat one set of assumptions as interchangeable with another. But assumption 1 is not assumption 2: different bootstrap methods, different centering choices, and different CI procedures can produce diverging results. Science built on unreported defaults is science that cannot be properly evaluated or replicated. `apastats` addresses this by explicitly surfacing all parameter settings in every analysis output, so that methods sections can be complete and precise.

Finally, an enormous amount of researcher time is wasted on formatting. Every decimal place, leading zero, significance star, and table border must follow exact APA 7th edition rules — and getting any of it wrong invites revision or desk rejection. `apastats` handles all of this programmatically so that researchers can focus on writing content, not formatting tables. Every analysis returns both a formatted plain-text table and a structured result object with raw statistics, a `.report()` method for copy-paste in-text reporting strings, and export to Word, LaTeX, or CSV.

## Installation

```bash
pip install apastats
```

For CFA and scale reliability features, install with optional dependencies:

```bash
pip install apastats[all]
```

## What's Included

| Module | What it does |
|---|---|
| **Descriptives** | JAP "Table 1" — means, SDs, lower-triangular correlations, Cronbach's alpha on the diagonal, significance stars |
| **Moderation** | Hierarchical regression, simple slopes at +/-1 SD, Johnson-Neyman regions of significance, interaction plots |
| **Mediation** | Bootstrap indirect effects (10,000 resamples), single and parallel mediators, path diagrams |
| **Conditional Process** | Moderated mediation (PROCESS Models 7, 8, 14, 15), index of moderated mediation with bootstrap CI |
| **CFA** | Confirmatory factor analysis via semopy — fit indices (chi-sq, CFI, TLI, RMSEA with 90% CI, SRMR), standardised loadings, CR, AVE, Fornell-Larcker, HTMT |
| **Scale Reliability** | Cronbach's alpha, McDonald's omega, composite reliability, AVE, corrected item-total correlations, alpha-if-deleted |
| **Effect Sizes** | Cohen's d (with CI), f-squared, R-squared interpretation, partial eta-squared |
| **Export** | APA-formatted Word (.docx), LaTeX (booktabs), CSV |

## Quick Start

### Descriptives and Correlations (Table 1)

```python
from apastats import descriptives_table

result = descriptives_table(
    data=df,
    variables=["pos", "commitment", "performance", "age"],
    labels=["Perceived org. support", "Commitment", "Performance", "Age"],
    alphas={
        "pos": ["pos_item1", "pos_item2", "pos_item3"],
        "commitment": 0.91,
        # age has no alpha — em-dash on diagonal
    },
)
print(result)
```

### Moderation Analysis

```python
from apastats import moderation_analysis

result = moderation_analysis(
    data=df, x="pos", w="empowerment", y="performance",
    controls=["age", "tenure"],
)
print(result)          # APA regression table
print(result.report()) # Copy-paste in-text strings
result.plot()          # Interaction plot at +/-1 SD
result.plot_jn()       # Johnson-Neyman plot
```

### Mediation Analysis

```python
from apastats import mediation_analysis

result = mediation_analysis(
    data=df, x="pos", m="engagement", y="performance",
    n_boot=10_000, seed=42,
)
print(result)          # APA mediation table
print(result.report()) # In-text strings with bootstrap CIs
result.plot()          # Path diagram
```

### Conditional Process Analysis (Moderated Mediation)

```python
from apastats import conditional_process

result = conditional_process(
    data=df, x="pos", m="engagement", y="performance", w="empowerment",
    model=7, n_boot=10_000, seed=42,
)
print(result)          # Table with IMM and conditional indirect effects
print(result.report()) # Copy-paste APA strings
```

### CFA and Scale Reliability

```python
from apastats import cfa, scale_reliability

# Scale reliability
rel = scale_reliability(df, items=["pos1", "pos2", "pos3", "pos4"])
print(rel)             # Item-level table with loadings, CITC, alpha-if-deleted
print(rel.report())    # "alpha = .89, omega = .90, CR = .90, AVE = .62"

# Confirmatory factor analysis
result = cfa(
    data=df,
    factors={
        "POS": ["pos1", "pos2", "pos3", "pos4"],
        "Commitment": ["com1", "com2", "com3", "com4"],
    },
)
print(result)          # Fit indices, loadings, CR, AVE, Fornell-Larcker
print(result.report()) # Copy-paste fit string
```

### Exporting to Word

```python
from apastats import descriptives_table, to_docx

result = descriptives_table(df, variables=["pos", "commitment", "performance"])
to_docx(
    result.table_df,
    "Table1.docx",
    title="Table 1",
    subtitle="Means, Standard Deviations, and Intercorrelations Among Study Variables",
    note="Note. N = 400. Reliability coefficients appear on the diagonal in parentheses.\n*p < .05. **p < .01.",
)
```

## APA Formatting Rules Enforced

- **No leading zero** for statistics bounded between -1 and +1: correlations (`.54`), alpha (`.87`), beta (`.34`), p values (`.03`), R-squared (`.22`)
- **Leading zero** for statistics that can exceed +/-1: M (`3.45`), SD (`0.82`), b (`0.25`), t (`2.94`), F (`7.33`)
- **Significance stars**: `*` p < .05, `**` p < .01 (in tables only, never in text)
- **p values**: exact to three decimals; `< .001` for very small values; never `.000`
- **Confidence intervals**: `95% CI [lower, upper]` with square brackets
- **Tables**: horizontal rules only (no vertical lines), three rules (top, below header, bottom)
- **Unstandardised coefficients** when interaction terms are present
- **Mean-centring** of predictors before computing product terms

## Running Tests

```bash
pip install apastats[dev]
pytest tests/ -v
```

191 tests covering all modules, edge cases, and APA formatting rules.

## License

MIT
