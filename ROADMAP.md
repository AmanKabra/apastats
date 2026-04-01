# Roadmap

Planned features for future releases, roughly in priority order.

## Next Up

### Common Method Variance (CMV) Testing
- Harman's single-factor test (EFA, report variance explained by first factor)
- CFA common latent factor comparison (with/without CLF, chi-square difference test)
- Unmeasured latent method construct (ULMC) approach
- Rationale: reviewers request CMV tests in ~76% of JAP submissions with same-source survey data

### Relative Weight / Dominance Analysis
- Relative importance of predictors accounting for multicollinearity (Johnson, 2000)
- Dominance analysis (Budescu, 1993)
- APA-formatted importance tables
- Rationale: frequent reviewer ask, "which predictor matters most?"

### Multilevel / HLM Module
- ICC(1) and ICC(2) calculations
- Null model (unconditional means)
- Random intercept and random slopes models
- Cross-level interactions
- APA-formatted output for all models
- Wraps `statsmodels.MixedLM` with researcher-friendly API
- Rationale: nested data (employees in teams) is ubiquitous in OB research

### Power Analysis
- A priori power for regression R-squared increment
- Monte Carlo power for mediation indirect effects
- Power for interaction detection in moderation
- Multilevel design power calculations
- Rationale: JAP now requires a priori power analyses or sample size justifications

## Future

### Measurement Invariance
- Configural, metric, scalar, strict invariance ladder
- Automated sequential model comparison
- Delta-CFI and chi-square difference tests at each step
- Rationale: required for any JAP paper comparing groups on latent constructs

### Full Structural Equation Modeling (SEM)
- Path models with measurement and structural components
- Standardised and unstandardised solution tables
- APA-formatted path diagrams
- Builds on existing CFA module + semopy backend

### Bayesian Estimation
- Bayesian credible intervals for indirect effects in mediation
- Bayes factors for model comparison
- Rationale: gaining traction in JAP; future differentiator

### Additional Analyses to Investigate
- Latent profile analysis (LPA)
- Polynomial regression with response surface methodology
- Meta-analytic structural equation modeling (MASEM)
- Experience sampling method (ESM) analyses
- Latent growth curve modeling
- Cross-lagged panel models

A systematic audit of JAP, AMJ, and Personnel Psychology papers (2023-2025) is planned to identify any other standard analyses missing from this list.
