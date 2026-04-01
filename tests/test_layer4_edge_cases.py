"""
Layer 4: Edge Cases and Robustness.

Tests that the package handles unusual, degenerate, and extreme inputs
without crashing or silently producing wrong results.  Every test in
this file should either succeed cleanly or raise an informative error.
"""

import math
import numpy as np
import pandas as pd
import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Degenerate Inputs
# ═══════════════════════════════════════════════════════════════════════════

class TestDegenerateInputs:

    def test_zero_variance_predictor_moderation(self):
        """Constant predictor should not crash (may produce NaN/inf coefficients)."""
        import warnings
        df = pd.DataFrame({
            "x": np.ones(100),
            "w": np.random.default_rng(0).normal(0, 1, 100),
            "y": np.random.default_rng(0).normal(0, 1, 100),
        })
        from apastats import moderation_analysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Should complete without crashing
            res = moderation_analysis(df, x="x", w="w", y="y", jn=False)
            assert res.n == 100

    def test_zero_variance_mediator(self):
        """Constant mediator should not crash."""
        import warnings
        df = pd.DataFrame({
            "x": np.random.default_rng(0).normal(0, 1, 100),
            "m": np.ones(100),
            "y": np.random.default_rng(0).normal(0, 1, 100),
        })
        from apastats import mediation_analysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mediation_analysis(df, x="x", m="m", y="y", n_boot=100, seed=0)
            assert res.n == 100

    def test_perfect_collinearity_moderation(self):
        """x = 2*w: regression may be degenerate but should not crash."""
        import warnings
        rng = np.random.default_rng(0)
        w = rng.normal(0, 1, 100)
        df = pd.DataFrame({
            "x": 2 * w,
            "w": w,
            "y": rng.normal(0, 1, 100),
        })
        from apastats import moderation_analysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = moderation_analysis(df, x="x", w="w", y="y", jn=False)
            assert res.n == 100

    def test_all_identical_items_reliability(self):
        """All identical item responses should return NaN alpha, not crash."""
        df = pd.DataFrame({
            "item1": np.ones(100),
            "item2": np.ones(100),
            "item3": np.ones(100),
        })
        from apastats import scale_reliability
        res = scale_reliability(df, items=["item1", "item2", "item3"])
        assert math.isnan(res.alpha)


# ═══════════════════════════════════════════════════════════════════════════
# Missing Data Patterns
# ═══════════════════════════════════════════════════════════════════════════

class TestMissingDataPatterns:

    def test_50_percent_missing_moderation(self):
        """Half the data missing should still produce results (on remaining half)."""
        rng = np.random.default_rng(42)
        n = 400
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "w": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        })
        df.loc[:199, "x"] = np.nan
        import warnings
        from apastats import moderation_analysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = moderation_analysis(df, x="x", w="w", y="y")
        assert res.n == 200

    def test_missing_on_single_variable(self):
        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "m": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        })
        df.loc[:49, "m"] = np.nan
        import warnings
        from apastats import mediation_analysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mediation_analysis(df, x="x", m="m", y="y", n_boot=100, seed=42)
        assert res.n == 250

    def test_descriptives_with_patchy_missing(self):
        """Different variables with different missing patterns."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "a": rng.normal(0, 1, n),
            "b": rng.normal(0, 1, n),
            "c": rng.normal(0, 1, n),
        })
        df.loc[:19, "a"] = np.nan
        df.loc[50:69, "b"] = np.nan
        import warnings
        from apastats import descriptives_table
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = descriptives_table(df, variables=["a", "b", "c"])
        # Should still produce a table
        assert "Table 1" in res.table_str


# ═══════════════════════════════════════════════════════════════════════════
# Numerical Extremes
# ═══════════════════════════════════════════════════════════════════════════

class TestNumericalExtremes:

    def test_very_large_values(self):
        """Variables with values in the millions should not overflow."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "x": rng.normal(1e6, 1e4, n),
            "w": rng.normal(1e6, 1e4, n),
            "y": rng.normal(1e6, 1e4, n),
        })
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y", jn=False)
        assert not math.isnan(res.interaction_b)

    def test_very_small_values(self):
        """Variables near zero should not underflow."""
        rng = np.random.default_rng(42)
        n = 200
        df = pd.DataFrame({
            "x": rng.normal(0, 1e-6, n),
            "w": rng.normal(0, 1e-6, n),
            "y": rng.normal(0, 1e-6, n),
        })
        from apastats import descriptives_table
        res = descriptives_table(df, variables=["x", "y"])
        assert res.n > 0

    def test_mixed_scales(self):
        """One variable 0-1, another 0-10000 should not break centering."""
        rng = np.random.default_rng(42)
        n = 300
        df = pd.DataFrame({
            "x": rng.uniform(0, 1, n),
            "w": rng.uniform(0, 10000, n),
            "y": rng.normal(50, 10, n),
        })
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y", jn=False)
        assert not math.isnan(res.interaction_b)


# ═══════════════════════════════════════════════════════════════════════════
# Boundary Conditions
# ═══════════════════════════════════════════════════════════════════════════

class TestBoundaryConditions:

    def test_correlation_of_exactly_one(self):
        """Two perfectly correlated variables."""
        x = np.arange(100, dtype=float)
        df = pd.DataFrame({"x": x, "y": x * 2 + 1})
        from apastats import descriptives_table
        res = descriptives_table(df, variables=["x", "y"])
        assert abs(res.correlations.loc["x", "y"] - 1.0) < 1e-10

    def test_correlation_of_exactly_zero(self):
        """Orthogonal variables (constructed, not random)."""
        n = 100
        x = np.sin(np.linspace(0, 4 * np.pi, n))
        y = np.cos(np.linspace(0, 4 * np.pi, n))
        df = pd.DataFrame({"x": x, "y": y})
        from apastats import descriptives_table
        res = descriptives_table(df, variables=["x", "y"])
        assert abs(res.correlations.loc["x", "y"]) < 0.05

    def test_r_squared_near_one(self):
        """Near-perfect prediction should yield R2 close to 1."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        w = rng.normal(0, 1, n)
        y = 3 * x + 2 * w + rng.normal(0, 0.01, n)
        df = pd.DataFrame({"x": x, "w": w, "y": y})
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y", jn=False)
        assert res.r2[-1] > 0.99


# ═══════════════════════════════════════════════════════════════════════════
# Small Samples
# ═══════════════════════════════════════════════════════════════════════════

class TestSmallSamples:

    def test_moderation_n20(self):
        """N=20 is small but should produce results."""
        rng = np.random.default_rng(42)
        n = 20
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "w": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        })
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y", jn=False)
        assert res.n == 20

    def test_mediation_n30(self):
        """N=30 with bootstrap should work."""
        rng = np.random.default_rng(42)
        n = 30
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "m": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        })
        from apastats import mediation_analysis
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=500, seed=42)
        assert res.n == 30

    def test_descriptives_n5(self):
        """Very small sample for descriptives."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, 5),
            "y": rng.normal(0, 1, 5),
        })
        from apastats import descriptives_table
        res = descriptives_table(df, variables=["x", "y"])
        assert res.n > 0

    def test_reliability_n10(self):
        rng = np.random.default_rng(42)
        n = 10
        ts = rng.normal(0, 1, n)
        df = pd.DataFrame({f"item{i}": ts + rng.normal(0, 0.5, n) for i in range(3)})
        from apastats import scale_reliability
        res = scale_reliability(df, items=["item0", "item1", "item2"])
        assert not math.isnan(res.alpha)


# ═══════════════════════════════════════════════════════════════════════════
# Large Samples (performance sanity check)
# ═══════════════════════════════════════════════════════════════════════════

class TestLargeSamples:

    def test_descriptives_n100k(self):
        """N=100,000 should complete without memory issues."""
        rng = np.random.default_rng(42)
        n = 100_000
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        })
        from apastats import descriptives_table
        res = descriptives_table(df, variables=["x", "y"])
        assert res.n == n

    def test_moderation_n50k(self):
        """N=50,000 regression should complete."""
        rng = np.random.default_rng(42)
        n = 50_000
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "w": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
        })
        from apastats import moderation_analysis
        res = moderation_analysis(df, x="x", w="w", y="y", jn=False)
        assert res.n == n


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap Edge Cases
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrapEdgeCases:

    def test_nboot_100_still_produces_ci(self):
        rng = np.random.default_rng(42)
        n = 100
        x = rng.normal(0, 1, n)
        m = 0.5 * x + rng.normal(0, 1, n)
        y = 0.5 * m + rng.normal(0, 1, n)
        df = pd.DataFrame({"x": x, "m": m, "y": y})
        from apastats import mediation_analysis
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=100, seed=42)
        ie = res.indirect_effects[0]
        assert not math.isnan(ie.ci_lower)
        assert not math.isnan(ie.ci_upper)
        assert ie.ci_lower < ie.ci_upper

    def test_ci_level_99(self):
        """99% CI should be wider than 95% CI."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        m = 0.5 * x + rng.normal(0, 1, n)
        y = 0.5 * m + rng.normal(0, 1, n)
        df = pd.DataFrame({"x": x, "m": m, "y": y})
        from apastats import mediation_analysis
        r95 = mediation_analysis(df, x="x", m="m", y="y", n_boot=2000, seed=42, ci_level=0.95)
        r99 = mediation_analysis(df, x="x", m="m", y="y", n_boot=2000, seed=42, ci_level=0.99)
        w95 = r95.indirect_effects[0].ci_upper - r95.indirect_effects[0].ci_lower
        w99 = r99.indirect_effects[0].ci_upper - r99.indirect_effects[0].ci_lower
        assert w99 > w95
