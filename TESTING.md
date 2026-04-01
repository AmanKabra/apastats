# Testing Protocol

## Philosophy

A researcher who runs `apastats` and puts the output into a manuscript is trusting that the numbers are correct and the formatting is compliant. If either assumption is wrong, the consequences range from a revision letter to a retraction. Statistical software bugs do not announce themselves; they produce plausible, wrong numbers that survive peer review.

We test in five layers, ordered by the severity of the failure each layer is designed to catch.

**Layer 1** verifies that each computation returns the right quantity. We generate data from a process where the true parameters are known (e.g., the true indirect effect is exactly 0.20) and check whether the package recovers them. If it cannot, the implementation is wrong at the level of algebra, and no other testing matters until that is fixed.

**Layer 2** verifies that the package agrees with established software. A computation can be algebraically correct but numerically divergent from scipy or statsmodels due to implementation differences (matrix decomposition choices, handling of near singular designs, floating point accumulation). We run the same analysis in both `apastats` and a reference package and assert agreement to six or more decimal places. Layer 1 checks correctness against a known truth. Layer 2 checks consistency against the tools reviewers already trust.

**Layer 3** verifies inferential properties across many replications. A correct point estimate on one dataset does not mean the confidence intervals have proper coverage or the significance tests control Type I error at the stated rate. A 95% CI could contain the true value on any single run but have only 80% coverage across a thousand runs. These properties are only visible in aggregate. We run hundreds of replications under known data generating processes and check that coverage rates are near 95%, rejection rates under the null are near 5%, and mean estimates are unbiased.

**Layer 4** verifies that the package handles the inputs researchers actually have. Real datasets contain missing values, zero variance items, variables on different scales, near collinear predictors, and small samples. A package that works on clean simulated data but crashes or silently produces wrong output on messy real data is not usable. The standard is that the package should never return a confident wrong answer. A crash with an informative error is acceptable. A NaN with a warning is acceptable. A wrong number presented as correct is not.

**Layer 5** verifies formatting compliance with APA 7th edition rules. These tests are independent of statistical correctness. They check that correlations have no leading zero, p values never appear as ".000," significance stars follow the correct thresholds, confidence intervals use square brackets, and tables have exactly three horizontal rules with no vertical lines. Formatting errors do not produce wrong science, but they signal carelessness and invite revision requests.

The ordering reflects a simple priority: getting the math right matters more than matching other software, which matters more than correct coverage rates, which matters more than surviving bad input, which matters more than formatting. We verify each level before moving to the next.

No test suite eliminates all bugs. This one verifies correctness at five distinct levels so that the failures it cannot catch are genuinely rare rather than structurally predictable.

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
