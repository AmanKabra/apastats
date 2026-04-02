# apastats

**APA 7th edition compliant statistical analyses for organizational science research.**

Python is increasingly the language of choice for analytical pipelines across the social sciences. Yet no Python package exists for the statistical reporting conventions required by journals in organizational behavior and adjacent fields, such as the *Journal of Applied Psychology*. Researchers who work in Python are forced to piece together output from general purpose libraries and reformat manually, or abandon Python entirely and switch to SPSS or R for final reporting. `apastats` closes this gap: it is a purpose built Python package that runs the standard analyses organizational scholars need and produces publication ready output in a single step.

The tooling gap is compounded by a transparency problem. Statistical software routinely applies consequential settings silently. The number of bootstrap resamples, the type of confidence interval (percentile vs. bias corrected), whether variables are mean centered, which estimator is used: these choices shape results, yet most tools bury them in undocumented defaults. When researchers do not report these settings, readers cannot evaluate whether the reported findings would hold under alternative, equally defensible specifications. Treating one set of assumptions as interchangeable with another is not a minor reporting omission; it is a threat to the credibility of cumulative science. `apastats` addresses this by printing all parameter settings alongside every analysis output, making methods sections complete by construction rather than by the researcher's diligence.

Third, a disproportionate share of researcher time goes to formatting rather than thinking. Every decimal place, leading zero, significance star, and table border in an APA 7th edition manuscript must follow precise rules, and deviations invite revision requests or desk rejection. `apastats` enforces these rules programmatically. Every analysis returns a formatted plain text table, a structured result object with raw statistics, a `.report()` method that generates copy paste in text reporting strings, and direct export to Word or CSV. The goal is simple: researchers should spend their time on content, not on counting decimal places.

## Disclaimer

`apastats` is provided as is, without warranty of any kind. While the package includes an extensive test suite (191 tests at the time of writing) and every effort has been made to ensure correctness, errors in statistical software are always possible. Users are strongly encouraged to cross verify results against at least one independent tool (e.g., PROCESS for SPSS, lavaan for R, jamovi) before relying on any output for publication. This is standard practice in quantitative research, and it is what the author does in his own work. The project is under active development and building rapidly, but it has not yet undergone external audit. By using this software, you accept full responsibility for verifying the accuracy of any results it produces. The author assumes no liability for errors, omissions, or consequences arising from the use of this package.

## Citation

If you use `apastats` in published research, please cite it as:

> Kabra, A. (2026). *apastats: APA 7th edition compliant statistical analyses for organizational science* (Version 0.1.6) [Computer software]. https://github.com/AmanKabra/apastats

BibTeX:

```bibtex
@software{kabra2026apastats,
  author = {Kabra, Aman},
  title = {apastats: APA 7th Edition Compliant Statistical Analyses for Organizational Science},
  year = {2026},
  url = {https://github.com/AmanKabra/apastats},
  version = {0.1.6}
}
```

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
| **Descriptives** | JAP "Table 1": means, SDs, lower triangular correlations, Cronbach's alpha on the diagonal, significance stars |
| **Moderation** | Hierarchical regression, simple slopes at +/-1 SD, Johnson-Neyman regions of significance, interaction plots |
| **Mediation** | Bootstrap indirect effects (10,000 resamples), single and parallel mediators, path diagrams |
| **Conditional Process** | Moderated mediation (PROCESS Models 7, 8, 14, 15), index of moderated mediation with bootstrap CI |
| **CFA** | Confirmatory factor analysis via semopy. Fit indices (chi sq, CFI, TLI, RMSEA with 90% CI, SRMR), standardised loadings, CR, AVE, Fornell Larcker, HTMT |
| **Scale Reliability** | Cronbach's alpha, McDonald's omega, composite reliability, AVE, corrected item-total correlations, alpha-if-deleted |
| **Effect Sizes** | Cohen's d (with CI), f-squared, R-squared interpretation, partial eta-squared |
| **Export** | APA-formatted Word (.docx), CSV |

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
        # age has no alpha, so the diagonal shows an em dash
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
