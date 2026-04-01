"""
Layer 1: Ground Truth Validation via Simulation.

Every statistical computation is verified against data generated from a
known data-generating process (DGP) where the true parameter values are
set by the researcher.  If the package cannot recover known parameters
within tolerance, the computation is wrong.

Tolerances are generous (0.10-0.15 for coefficients) because we use
finite samples and stochastic noise.  The point is not decimal-level
precision (that is Layer 2) but structural correctness: does the
package estimate the right quantity?
"""

import math
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simulate_mediation(n=2000, a=0.5, b=0.4, cp=0.2, seed=42):
    """DGP: X -> M -> Y with known a, b, c' paths."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    m = a * x + rng.normal(0, 0.7, n)
    y = b * m + cp * x + rng.normal(0, 0.7, n)
    return pd.DataFrame({"x": x, "m": m, "y": y}), {"a": a, "b": b, "cp": cp, "ab": a * b}


def _simulate_moderation(n=2000, b_x=0.5, b_w=0.3, b_xw=0.4, seed=42):
    """DGP: Y = b0 + b_x*X + b_w*W + b_xw*X*W + noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    y = 1.0 + b_x * x + b_w * w + b_xw * x * w + rng.normal(0, 0.8, n)
    return pd.DataFrame({"x": x, "w": w, "y": y}), {"b_x": b_x, "b_w": b_w, "b_xw": b_xw}


def _simulate_conditional_process_model7(n=2000, a1=0.5, a3=0.4, b=0.3, cp=0.1, seed=42):
    """DGP for PROCESS Model 7: first-stage moderated mediation."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    m = a1 * x + 0.2 * w + a3 * x * w + rng.normal(0, 0.7, n)
    y = b * m + cp * x + rng.normal(0, 0.7, n)
    return pd.DataFrame({"x": x, "w": w, "m": m, "y": y}), {"a1": a1, "a3": a3, "b": b, "imm": a3 * b}


def _simulate_conditional_process_model14(n=2000, a=0.6, b1=0.4, b3=0.3, cp=0.1, seed=42):
    """DGP for PROCESS Model 14: second-stage moderated mediation."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    m = a * x + rng.normal(0, 0.7, n)
    y = b1 * m + 0.2 * w + b3 * m * w + cp * x + rng.normal(0, 0.7, n)
    return pd.DataFrame({"x": x, "w": w, "m": m, "y": y}), {"a": a, "b1": b1, "b3": b3, "imm": a * b3}


def _simulate_two_factor_cfa(n=1000, seed=42):
    """DGP: two latent factors, 4 indicators each, known loadings ~0.7-0.8."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(0, 1, n)
    f2 = 0.4 * f1 + rng.normal(0, 0.85, n)
    loadings_f1 = [0.80, 0.75, 0.70, 0.72]
    loadings_f2 = [0.78, 0.82, 0.75, 0.70]
    data = {}
    for i, lam in enumerate(loadings_f1):
        data[f"f1_item{i+1}"] = lam * f1 + rng.normal(0, 0.5, n)
    for i, lam in enumerate(loadings_f2):
        data[f"f2_item{i+1}"] = lam * f2 + rng.normal(0, 0.5, n)
    return pd.DataFrame(data), {"loadings_f1": loadings_f1, "loadings_f2": loadings_f2}


def _simulate_reliability_scale(n=1000, k=5, true_loading=0.75, seed=42):
    """DGP: one-factor congeneric model with known loadings."""
    rng = np.random.default_rng(seed)
    true_score = rng.normal(0, 1, n)
    items = {}
    for i in range(k):
        noise_sd = math.sqrt(1 - true_loading ** 2)
        items[f"item{i+1}"] = true_loading * true_score + rng.normal(0, noise_sd, n)
    return pd.DataFrame(items), {"loading": true_loading, "k": k}


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGroundTruthDescriptives:
    """Verify means, SDs, and correlations against known DGP values."""

    def test_mean_recovery(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(5.0, 1.0, 5000)})
        from apastats import descriptives_table
        res = descriptives_table(df, variables=["x"])
        assert abs(res.means["x"] - 5.0) < 0.10

    def test_sd_recovery(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({"x": rng.normal(0, 2.0, 5000)})
        from apastats import descriptives_table
        res = descriptives_table(df, variables=["x"])
        assert abs(res.sds["x"] - 2.0) < 0.10

    def test_correlation_recovery(self):
        """Known bivariate normal with r = 0.6."""
        rng = np.random.default_rng(42)
        cov = [[1.0, 0.6], [0.6, 1.0]]
        data = rng.multivariate_normal([0, 0], cov, size=5000)
        df = pd.DataFrame(data, columns=["x", "y"])
        from apastats import descriptives_table
        res = descriptives_table(df, variables=["x", "y"])
        assert abs(res.correlations.loc["x", "y"] - 0.6) < 0.05

    def test_zero_correlation(self):
        """Independent variables should have r near 0."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 5000),
            "y": rng.normal(0, 1, 5000),
        })
        from apastats import descriptives_table
        res = descriptives_table(df, variables=["x", "y"])
        assert abs(res.correlations.loc["x", "y"]) < 0.05


class TestGroundTruthModeration:
    """Verify moderation coefficients against known DGP."""

    def test_interaction_coefficient_recovery(self):
        df, truth = _simulate_moderation(n=3000, b_xw=0.40)
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y")
        assert abs(res.interaction_b - truth["b_xw"]) < 0.10

    def test_interaction_significance_when_present(self):
        df, _ = _simulate_moderation(n=2000, b_xw=0.40)
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y")
        assert res.interaction_p < 0.01

    def test_no_interaction_when_absent(self):
        df, _ = _simulate_moderation(n=5000, b_xw=0.0, seed=99)
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y")
        assert res.interaction_p > 0.05

    def test_simple_slopes_direction(self):
        """With positive interaction, slope at +1 SD should exceed slope at -1 SD."""
        df, _ = _simulate_moderation(n=3000, b_x=0.5, b_xw=0.4)
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y")
        low = res.simple_slopes[0].b   # -1 SD
        high = res.simple_slopes[2].b  # +1 SD
        assert high > low

    def test_r2_increases_with_interaction(self):
        df, _ = _simulate_moderation(n=2000, b_xw=0.40)
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y")
        assert res.r2[-1] > res.r2[0]


class TestGroundTruthMediation:
    """Verify mediation paths and indirect effect against known DGP."""

    def test_a_path_recovery(self):
        df, truth = _simulate_mediation(n=3000, a=0.50)
        from apastats import mediation_analysis
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=500, seed=42)
        assert abs(res.paths["a"].b - truth["a"]) < 0.10

    def test_b_path_recovery(self):
        df, truth = _simulate_mediation(n=3000, b=0.40)
        from apastats import mediation_analysis
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=500, seed=42)
        assert abs(res.paths["b"].b - truth["b"]) < 0.10

    def test_direct_effect_recovery(self):
        df, truth = _simulate_mediation(n=3000, cp=0.20)
        from apastats import mediation_analysis
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=500, seed=42)
        assert abs(res.direct_effect.b - truth["cp"]) < 0.10

    def test_indirect_effect_recovery(self):
        df, truth = _simulate_mediation(n=3000, a=0.50, b=0.40)
        from apastats import mediation_analysis
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=1000, seed=42)
        assert abs(res.indirect_effects[0].ab - truth["ab"]) < 0.10

    def test_indirect_significant_when_present(self):
        df, _ = _simulate_mediation(n=2000, a=0.50, b=0.40)
        from apastats import mediation_analysis
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=2000, seed=42)
        assert res.indirect_effects[0].significant

    def test_no_indirect_when_a_zero(self):
        df, _ = _simulate_mediation(n=2000, a=0.0, b=0.40)
        from apastats import mediation_analysis
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=2000, seed=42)
        assert not res.indirect_effects[0].significant

    def test_total_effect_decomposition(self):
        """c should equal c' + ab within tolerance."""
        df, truth = _simulate_mediation(n=5000, a=0.50, b=0.40, cp=0.20)
        from apastats import mediation_analysis
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=500, seed=42)
        c = res.total_effect.b
        c_prime = res.direct_effect.b
        ab = res.indirect_effects[0].ab
        assert abs(c - (c_prime + ab)) < 0.05


class TestGroundTruthConditionalProcess:
    """Verify conditional process models against known DGP."""

    def test_model7_imm_recovery(self):
        df, truth = _simulate_conditional_process_model7(n=3000)
        from apastats import conditional_process
        res = conditional_process(df, x="x", m="m", y="y", w="w", model=7, n_boot=1000, seed=42)
        assert abs(res.imm - truth["imm"]) < 0.10

    def test_model7_imm_significant(self):
        df, _ = _simulate_conditional_process_model7(n=2000)
        from apastats import conditional_process
        res = conditional_process(df, x="x", m="m", y="y", w="w", model=7, n_boot=2000, seed=42)
        assert res.imm_significant

    def test_model7_conditional_indirect_direction(self):
        """With positive a3 and positive b, indirect should increase with W."""
        df, _ = _simulate_conditional_process_model7(n=3000, a3=0.40, b=0.30)
        from apastats import conditional_process
        res = conditional_process(df, x="x", m="m", y="y", w="w", model=7, n_boot=500, seed=42)
        low = res.conditional_indirect[0].ab   # -1 SD
        high = res.conditional_indirect[2].ab  # +1 SD
        assert high > low

    def test_model14_imm_recovery(self):
        df, truth = _simulate_conditional_process_model14(n=3000)
        from apastats import conditional_process
        res = conditional_process(df, x="x", m="m", y="y", w="w", model=14, n_boot=1000, seed=42)
        assert abs(res.imm - truth["imm"]) < 0.10

    def test_model14_imm_significant(self):
        df, _ = _simulate_conditional_process_model14(n=2000)
        from apastats import conditional_process
        res = conditional_process(df, x="x", m="m", y="y", w="w", model=14, n_boot=2000, seed=42)
        assert res.imm_significant


class TestGroundTruthCFA:
    """Verify CFA fit and loadings against known factor structure."""

    def test_good_fit_for_correct_model(self):
        df, _ = _simulate_two_factor_cfa(n=1000)
        from apastats import cfa
        res = cfa(df, factors={
            "F1": ["f1_item1", "f1_item2", "f1_item3", "f1_item4"],
            "F2": ["f2_item1", "f2_item2", "f2_item3", "f2_item4"],
        })
        assert res.fit.cfi > 0.90

    def test_poor_fit_for_wrong_model(self):
        """Forcing all items onto one factor should yield worse fit."""
        df, _ = _simulate_two_factor_cfa(n=1000)
        from apastats import cfa
        all_items = [f"f1_item{i}" for i in range(1, 5)] + [f"f2_item{i}" for i in range(1, 5)]
        res_1f = cfa(df, factors={"General": all_items})
        res_2f = cfa(df, factors={
            "F1": ["f1_item1", "f1_item2", "f1_item3", "f1_item4"],
            "F2": ["f2_item1", "f2_item2", "f2_item3", "f2_item4"],
        })
        assert res_2f.fit.cfi > res_1f.fit.cfi

    def test_loadings_on_correct_factor(self):
        df, _ = _simulate_two_factor_cfa(n=1000)
        from apastats import cfa
        res = cfa(df, factors={
            "F1": ["f1_item1", "f1_item2", "f1_item3", "f1_item4"],
            "F2": ["f2_item1", "f2_item2", "f2_item3", "f2_item4"],
        })
        for i in range(1, 5):
            assert abs(res.loadings_df.loc[f"f1_item{i}", "F1"]) > 0.40
            assert abs(res.loadings_df.loc[f"f2_item{i}", "F2"]) > 0.40

    def test_cr_above_threshold(self):
        df, _ = _simulate_two_factor_cfa(n=1000)
        from apastats import cfa
        res = cfa(df, factors={
            "F1": ["f1_item1", "f1_item2", "f1_item3", "f1_item4"],
            "F2": ["f2_item1", "f2_item2", "f2_item3", "f2_item4"],
        })
        assert res.cr["F1"] > 0.60
        assert res.cr["F2"] > 0.60


class TestGroundTruthReliability:
    """Verify reliability statistics against known congeneric model."""

    def test_alpha_high_for_reliable_scale(self):
        df, _ = _simulate_reliability_scale(n=2000, k=5, true_loading=0.75)
        from apastats import scale_reliability
        res = scale_reliability(df, items=[f"item{i}" for i in range(1, 6)])
        assert res.alpha > 0.75

    def test_omega_close_to_alpha_for_congeneric(self):
        df, _ = _simulate_reliability_scale(n=2000, k=5, true_loading=0.75)
        from apastats import scale_reliability
        res = scale_reliability(df, items=[f"item{i}" for i in range(1, 6)])
        assert abs(res.omega_total - res.alpha) < 0.10

    def test_ave_recoverable(self):
        """AVE should be near loading^2 for equal-loading congeneric model."""
        df, truth = _simulate_reliability_scale(n=2000, k=5, true_loading=0.75)
        from apastats import scale_reliability
        res = scale_reliability(df, items=[f"item{i}" for i in range(1, 6)])
        expected_ave = truth["loading"] ** 2
        assert abs(res.ave - expected_ave) < 0.15

    def test_low_reliability_detected(self):
        """Weak loadings should yield low alpha."""
        df, _ = _simulate_reliability_scale(n=2000, k=5, true_loading=0.30)
        from apastats import scale_reliability
        res = scale_reliability(df, items=[f"item{i}" for i in range(1, 6)])
        assert res.alpha < 0.50


class TestGroundTruthEffectSizes:
    """Verify effect sizes against known group differences."""

    def test_cohens_d_recovery(self):
        """Two groups with 1 SD separation should yield d near 1.0."""
        rng = np.random.default_rng(42)
        g1 = rng.normal(1.0, 1.0, 2000)
        g2 = rng.normal(0.0, 1.0, 2000)
        from apastats import cohens_d
        res = cohens_d(g1, g2)
        assert abs(res.value - 1.0) < 0.10

    def test_cohens_d_zero_when_equal(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 2000)
        g2 = rng.normal(0, 1, 2000)
        from apastats import cohens_d
        res = cohens_d(g1, g2)
        assert abs(res.value) < 0.10

    def test_f2_from_known_r2(self):
        """f2 = R2 / (1 - R2). For R2 = 0.20, f2 = 0.25."""
        from apastats import cohens_f2
        res = cohens_f2(0.20)
        assert abs(res.value - 0.25) < 0.01

    def test_f2_incremental(self):
        """f2 = (R2_full - R2_reduced) / (1 - R2_full)."""
        from apastats import cohens_f2
        res = cohens_f2(0.30, 0.20)
        expected = (0.30 - 0.20) / (1 - 0.30)
        assert abs(res.value - expected) < 0.01
