"""
Layer 3: Monte Carlo Statistical Property Tests.

These tests verify that the package's inferential machinery has correct
statistical properties *in expectation* across many replications:
  - Bootstrap CIs achieve nominal coverage rates.
  - Significance tests control Type I error at the stated alpha.
  - Parameter estimates are unbiased (mean across replications equals truth).

These tests are computationally expensive (hundreds of replications,
each with bootstrap resampling).  They are marked ``@pytest.mark.slow``
and excluded from the default test run.  Run them with:

    pytest -m slow tests/test_layer3_montecarlo.py

Expected runtime: 5-15 minutes depending on hardware.
"""

import numpy as np
import pandas as pd
import pytest


# All tests in this file are slow
pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coverage_rate(true_value, ci_lowers, ci_uppers):
    """Fraction of CIs that contain the true value."""
    covers = [(lo <= true_value <= hi) for lo, hi in zip(ci_lowers, ci_uppers)]
    return np.mean(covers)


def _rejection_rate(p_values, alpha=0.05):
    """Fraction of p-values below alpha."""
    return np.mean([p < alpha for p in p_values])


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap CI Coverage for Mediation
# ═══════════════════════════════════════════════════════════════════════════

class TestMonteCarloCoverageMediationIndirect:
    """95% bootstrap CI for the indirect effect should contain the true
    value in approximately 95% of replications (92%-98% acceptable)."""

    def test_indirect_effect_coverage(self):
        from apastats import mediation_analysis

        true_a, true_b = 0.5, 0.4
        true_ab = true_a * true_b
        n_reps = 200
        n_obs = 200
        n_boot = 1000

        covers = []
        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            x = rng.normal(0, 1, n_obs)
            m = true_a * x + rng.normal(0, 0.7, n_obs)
            y = true_b * m + 0.2 * x + rng.normal(0, 0.7, n_obs)
            df = pd.DataFrame({"x": x, "m": m, "y": y})

            res = mediation_analysis(df, x="x", m="m", y="y", n_boot=n_boot, seed=rep)
            ie = res.indirect_effects[0]
            covers.append(ie.ci_lower <= true_ab <= ie.ci_upper)

        coverage = np.mean(covers)
        assert 0.88 <= coverage <= 0.99, f"Coverage = {coverage:.3f}, expected ~0.95"


# ═══════════════════════════════════════════════════════════════════════════
# Type I Error for Moderation Interaction Test
# ═══════════════════════════════════════════════════════════════════════════

class TestMonteCarloTypeIErrorModeration:
    """When the true interaction is zero, the interaction test should
    reject at approximately the nominal alpha rate (3%-7% for alpha=.05)."""

    def test_interaction_type1_error(self):
        from apastats import moderation_analysis

        n_reps = 500
        n_obs = 200
        rejections = []

        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            x = rng.normal(0, 1, n_obs)
            w = rng.normal(0, 1, n_obs)
            y = 0.5 * x + 0.3 * w + rng.normal(0, 1, n_obs)  # no interaction
            df = pd.DataFrame({"x": x, "w": w, "y": y})

            res = moderation_analysis(df, x="x", w="w", y="y", jn=False)
            rejections.append(res.interaction_p < 0.05)

        rate = np.mean(rejections)
        assert 0.02 <= rate <= 0.08, f"Type I error = {rate:.3f}, expected ~0.05"


# ═══════════════════════════════════════════════════════════════════════════
# Type I Error for Mediation Indirect Effect
# ═══════════════════════════════════════════════════════════════════════════

class TestMonteCarloTypeIErrorMediation:
    """When a = 0 (no mediation), bootstrap CI should include zero in
    approximately 95% of replications (rejection ~5%)."""

    def test_indirect_type1_error(self):
        from apastats import mediation_analysis

        n_reps = 200
        n_obs = 200
        n_boot = 1000
        rejections = []

        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            x = rng.normal(0, 1, n_obs)
            m = rng.normal(0, 1, n_obs)  # a = 0, no X->M path
            y = 0.4 * m + 0.3 * x + rng.normal(0, 0.7, n_obs)
            df = pd.DataFrame({"x": x, "m": m, "y": y})

            res = mediation_analysis(df, x="x", m="m", y="y", n_boot=n_boot, seed=rep)
            rejections.append(res.indirect_effects[0].significant)

        rate = np.mean(rejections)
        assert 0.01 <= rate <= 0.10, f"Type I error = {rate:.3f}, expected ~0.05"


# ═══════════════════════════════════════════════════════════════════════════
# Parameter Recovery (Unbiasedness) for Moderation
# ═══════════════════════════════════════════════════════════════════════════

class TestMonteCarloUnbiasednessModeration:
    """Mean estimated interaction coefficient across replications should
    be close to the true value (unbiased)."""

    def test_interaction_unbiased(self):
        from apastats import moderation_analysis

        true_bxw = 0.30
        n_reps = 300
        n_obs = 300
        estimates = []

        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            x = rng.normal(0, 1, n_obs)
            w = rng.normal(0, 1, n_obs)
            y = 1 + 0.5 * x + 0.3 * w + true_bxw * x * w + rng.normal(0, 1, n_obs)
            df = pd.DataFrame({"x": x, "w": w, "y": y})

            res = moderation_analysis(df, x="x", w="w", y="y", jn=False)
            estimates.append(res.interaction_b)

        mean_est = np.mean(estimates)
        assert abs(mean_est - true_bxw) < 0.03, f"Mean estimate = {mean_est:.4f}, true = {true_bxw}"


# ═══════════════════════════════════════════════════════════════════════════
# Parameter Recovery (Unbiasedness) for Mediation Paths
# ═══════════════════════════════════════════════════════════════════════════

class TestMonteCarloUnbiasednessMediation:
    """Mean estimated a and b paths across replications should be close
    to the true values."""

    def test_a_path_unbiased(self):
        from apastats import mediation_analysis

        true_a = 0.50
        n_reps = 200
        n_obs = 300
        estimates = []

        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            x = rng.normal(0, 1, n_obs)
            m = true_a * x + rng.normal(0, 0.7, n_obs)
            y = 0.4 * m + 0.2 * x + rng.normal(0, 0.7, n_obs)
            df = pd.DataFrame({"x": x, "m": m, "y": y})

            res = mediation_analysis(df, x="x", m="m", y="y", n_boot=100, seed=rep)
            estimates.append(res.paths["a"].b)

        mean_est = np.mean(estimates)
        assert abs(mean_est - true_a) < 0.03, f"Mean a = {mean_est:.4f}, true = {true_a}"


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap CI Coverage for Conditional Process IMM
# ═══════════════════════════════════════════════════════════════════════════

class TestMonteCarloCoverageIMM:
    """95% bootstrap CI for the index of moderated mediation should
    contain the true IMM in approximately 95% of replications."""

    def test_imm_coverage(self):
        from apastats import conditional_process

        true_a3, true_b = 0.40, 0.30
        true_imm = true_a3 * true_b
        n_reps = 150
        n_obs = 300
        n_boot = 1000

        covers = []
        for rep in range(n_reps):
            rng = np.random.default_rng(rep)
            x = rng.normal(0, 1, n_obs)
            w = rng.normal(0, 1, n_obs)
            m = 0.5 * x + 0.2 * w + true_a3 * x * w + rng.normal(0, 0.7, n_obs)
            y = true_b * m + 0.1 * x + rng.normal(0, 0.7, n_obs)
            df = pd.DataFrame({"x": x, "w": w, "m": m, "y": y})

            res = conditional_process(df, x="x", m="m", y="y", w="w", model=7, n_boot=n_boot, seed=rep)
            covers.append(res.imm_ci_lower <= true_imm <= res.imm_ci_upper)

        coverage = np.mean(covers)
        assert 0.88 <= coverage <= 0.99, f"IMM coverage = {coverage:.3f}, expected ~0.95"
