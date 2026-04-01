# Testing Protocol

## Philosophy

Trust in statistical software is not a feature. It is the product.

A researcher who uses `apastats` to produce a table for a *Journal of Applied Psychology* submission is staking a career decision on the claim that the numbers are right. If a bootstrap confidence interval has incorrect coverage, a manuscript gets published with a conclusion that does not hold. If a p value is formatted with a leading zero, a reviewer questions the author's competence. Neither error is recoverable after publication. The cost of a bug in statistical software is not a crash report; it is a retraction, a failed replication, or a finding that quietly misleads the field.

This is why we test in layers, and why the layers are ordered the way they are.

The progression follows an epistemological logic. Layer 1 asks: does the package compute the right quantity? This is the most fundamental question. If you ask for the indirect effect and the code returns something other than the product of the a and b paths, nothing else matters. We verify this by generating data from a known process where the true parameters are set by the researcher, then checking whether the package recovers them. This is the statistical equivalent of checking that a thermometer reads 100 degrees when placed in boiling water. No comparison to other thermometers, no repeated trials. Just: does this instrument measure what it claims to measure?

Layer 2 asks a subtler question: does the package agree with instruments the field already trusts? A package can recover the right quantity in principle but implement the computation with a numerical shortcut that diverges from established software by the fourth decimal place. For most purposes this is harmless. For some (ill conditioned matrices, near singular designs, very small samples) it compounds. We test this by running the same analysis in both `apastats` and in scipy, statsmodels, or pingouin, and asserting agreement to six or more decimal places. This is not redundant with Layer 1. Layer 1 tests whether the package gets the answer right in a world we constructed. Layer 2 tests whether it gets the same answer as everyone else in the real world.

Layer 3 moves from single datasets to distributions. A single correct answer on a single dataset does not tell you whether the inferential machinery works. A 95% confidence interval that happens to contain the true value on one dataset could have 80% coverage across a thousand datasets. A significance test that correctly fails to reject on one null dataset could have a 12% false positive rate across five hundred. These properties are only visible in aggregate, which is why this layer runs hundreds of replications and checks coverage rates, Type I error rates, and estimator bias. This is the most computationally expensive layer, and the most important for establishing that the package's inferences can be trusted for publication.

Layer 4 asks: what happens when the world is not well behaved? Real data has missing values, zero variance items, variables on wildly different scales, sample sizes of 15, and predictors that are nearly collinear. A package that produces correct results on clean simulated data but crashes on a dataset with 40% missingness is not usable. This layer tests every degenerate, extreme, and unusual input we can construct, and verifies that the package either handles it gracefully or fails with an informative error. The standard is simple: the package should never silently produce a wrong number. A crash is acceptable. A NaN with a warning is acceptable. A confident wrong answer is not.

Layer 5 tests something entirely different from Layers 1 through 4. It tests the presentation, not the computation. APA 7th edition formatting rules are precise and mechanical: no leading zero on correlations, leading zero on means, exact p values to three decimals, significance stars at specific thresholds, confidence intervals in square brackets, tables with exactly three horizontal rules and no vertical lines. These rules have nothing to do with whether the statistics are correct, but violating them signals carelessness to reviewers and can delay publication. Every formatted string the package produces is checked against these rules using pattern matching. If a correlation appears as "0.54" instead of ".54", the test fails.

The layers are ordered by consequence. A wrong number (Layer 1) is worse than a number that disagrees with another package (Layer 2), which is worse than an inferential procedure with incorrect properties (Layer 3), which is worse than a crash on edge case input (Layer 4), which is worse than a formatting error (Layer 5). We build confidence from the inside out: first that the math is right, then that it agrees with the field, then that it works in expectation, then that it survives the mess of real data, and finally that it looks right on the page.

No test suite can guarantee the absence of bugs. But a test suite that verifies correctness at five distinct levels of abstraction, from the algebra of a single computation to the formatting of a single decimal point, is the closest we can come to earning the trust that publication requires.

---

## Layers

### Layer 1: Ground Truth Validation via Simulation

Every statistical computation is verified against data generated from a known data generating process (DGP) where the true parameter values are set by the researcher. If the package cannot recover known parameters within tolerance, the computation is wrong.

Each module has dedicated ground truth tests:
- **Descriptives**: recover known means, SDs, and correlations from multivariate normal data
- **Moderation**: recover known interaction coefficient, verify significance when present and absent, verify simple slope direction
- **Mediation**: recover known a, b, c' paths and indirect effect, verify total effect decomposition (c = c' + ab)
- **Conditional process**: recover known index of moderated mediation for Models 7 and 14, verify conditional indirect effect direction
- **CFA**: correct model yields good fit, wrong model yields poor fit, loadings load on correct factors
- **Reliability**: high loading DGP yields high alpha/omega, low loading DGP yields low alpha/omega, AVE near loading squared
- **Effect sizes**: known group separation yields expected Cohen's d, known R squared yields expected f squared

File: `tests/test_layer1_ground_truth.py`

### Layer 2: Cross Package Concordance

Every computation is run in both `apastats` and an established reference package. Results must agree within tight numerical tolerance (1e-6 or better for deterministic computations). This catches subtle bugs that Layer 1 would miss.

Reference packages used:
- `scipy.stats.pearsonr` for correlations and p values
- `pandas` for means and standard deviations
- `statsmodels.OLS` for regression coefficients, standard errors, R squared
- `pingouin.cronbach_alpha` for Cronbach's alpha
- `pingouin.compute_effsize` for Cohen's d

File: `tests/test_layer2_concordance.py`

### Layer 3: Monte Carlo Statistical Property Tests

These tests verify that inferential machinery has correct properties in expectation across hundreds of replications:
- **Bootstrap CI coverage**: 95% CIs should contain the true value in approximately 95% of replications (acceptance range: 88% to 99%)
- **Type I error**: when the true effect is zero, rejection rate at alpha = .05 should be between 2% and 8%
- **Unbiasedness**: mean parameter estimate across replications should be within 0.03 of the true value

These tests are computationally expensive and marked with `@pytest.mark.slow`. They are excluded from default test runs.

Run them with:
```bash
pytest -m slow tests/test_layer3_montecarlo.py
```

File: `tests/test_layer3_montecarlo.py`

### Layer 4: Edge Cases and Robustness

Tests that the package handles unusual inputs without crashing or silently producing wrong results:
- **Degenerate inputs**: zero variance, perfect collinearity, constant predictors, all identical items
- **Missing data**: 50% missing, patchy missingness, single variable missing
- **Numerical extremes**: values at 1e6, values at 1e-6, mixed scales
- **Boundary conditions**: r = 1.0, r = 0.0, R squared near 1
- **Small samples**: N = 5, 10, 20, 30
- **Large samples**: N = 50,000 and 100,000 (performance sanity)
- **Bootstrap edge cases**: n_boot = 100, ci_level = 0.99

File: `tests/test_layer4_edge_cases.py`

### Layer 5: APA Formatting Compliance

Every formatted string is checked against APA 7th edition rules using regex and string assertions:
- No leading zero on bounded statistics (r, alpha, beta, p, R squared)
- Leading zero on unbounded statistics (M, SD, b, SE, t, F, d)
- p values exact to 3 decimals, "< .001" for very small, never ".000"
- Significance stars: ** for p < .01, * for p < .05, none at boundaries
- CI format: [lower, upper] with square brackets and comma
- Table structure: no vertical lines, exactly 3 horizontal rules
- Table notes: "Note." followed by N, alpha note, probability note
- In text strings: correct stat type formatting, degrees of freedom in parentheses

File: `tests/test_layer5_apa_compliance.py`

---

## Running the Tests

```bash
# Fast suite (Layers 1, 2, 4, 5): ~60 seconds
pytest tests/ -m "not slow"

# Full suite including Monte Carlo (Layer 3): ~10 minutes
pytest tests/

# Individual layers
pytest tests/test_layer1_ground_truth.py -v
pytest tests/test_layer2_concordance.py -v
pytest -m slow tests/test_layer3_montecarlo.py -v
pytest tests/test_layer4_edge_cases.py -v
pytest tests/test_layer5_apa_compliance.py -v
```

## Tolerance Standards

| Test type | Tolerance | Rationale |
|---|---|---|
| Parameter recovery (Layer 1) | 0.10 to 0.15 | Finite sample noise in simulation |
| Cross package concordance (Layer 2) | 1e-6 to 1e-10 | Deterministic computations should agree exactly |
| Bootstrap CI coverage (Layer 3) | 88% to 99% | Sampling variability around 95% nominal |
| Type I error rate (Layer 3) | 2% to 8% | Sampling variability around 5% nominal |
| Unbiasedness (Layer 3) | 0.03 | Mean across 200+ replications |
