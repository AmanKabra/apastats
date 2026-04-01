"""
Layer 2: Cross-Package Concordance.

Every computation is run in both apastats and an established reference
package (statsmodels, scipy, pingouin).  Results must agree within
tight numerical tolerance.  This catches subtle bugs that Layer 1
(parameter recovery from simulations) would miss because both the
true value and the estimate could be wrong in the same direction.

Tolerances here are strict: 1e-6 for deterministic computations,
looser for bootstrap-based quantities (which are stochastic by design).
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

# ---------------------------------------------------------------------------
# Shared simulated data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ols_data():
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.5 + 0.8 * x1 + 0.3 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture(scope="module")
def corr_data():
    rng = np.random.default_rng(42)
    cov = [[1.0, 0.5, -0.3], [0.5, 1.0, 0.2], [-0.3, 0.2, 1.0]]
    data = rng.multivariate_normal([3.0, 2.0, 4.0], cov, size=500)
    return pd.DataFrame(data, columns=["a", "b", "c"])


@pytest.fixture(scope="module")
def reliability_data():
    rng = np.random.default_rng(42)
    true_score = rng.normal(0, 1, 500)
    items = {}
    for i in range(5):
        items[f"item{i+1}"] = true_score + rng.normal(0, 0.4, 500)
    return pd.DataFrame(items)


# ═══════════════════════════════════════════════════════════════════════════
# Correlations: apastats vs scipy
# ═══════════════════════════════════════════════════════════════════════════

class TestConcordanceCorrelations:

    def test_pearson_r_matches_scipy(self, corr_data):
        from scipy.stats import pearsonr
        from apastats import descriptives_table

        res = descriptives_table(corr_data, variables=["a", "b", "c"])

        for i, v1 in enumerate(["a", "b", "c"]):
            for j, v2 in enumerate(["a", "b", "c"]):
                if i == j:
                    continue
                r_apastats = res.correlations.loc[v1, v2]
                r_scipy, _ = pearsonr(corr_data[v1], corr_data[v2])
                assert_allclose(r_apastats, r_scipy, atol=1e-10)

    def test_pearson_p_matches_scipy(self, corr_data):
        from scipy.stats import pearsonr
        from apastats import descriptives_table

        res = descriptives_table(corr_data, variables=["a", "b", "c"])

        for i, v1 in enumerate(["a", "b", "c"]):
            for j, v2 in enumerate(["a", "b", "c"]):
                if i >= j:
                    continue
                p_apastats = res.p_values.loc[v1, v2]
                _, p_scipy = pearsonr(corr_data[v1], corr_data[v2])
                assert_allclose(p_apastats, p_scipy, atol=1e-10)

    def test_means_match_pandas(self, corr_data):
        from apastats import descriptives_table
        res = descriptives_table(corr_data, variables=["a", "b", "c"])
        for v in ["a", "b", "c"]:
            assert_allclose(res.means[v], corr_data[v].mean(), atol=1e-10)

    def test_sds_match_pandas(self, corr_data):
        from apastats import descriptives_table
        res = descriptives_table(corr_data, variables=["a", "b", "c"])
        for v in ["a", "b", "c"]:
            assert_allclose(res.sds[v], corr_data[v].std(ddof=1), atol=1e-10)


# ═══════════════════════════════════════════════════════════════════════════
# OLS Regression: apastats moderation internals vs statsmodels
# ═══════════════════════════════════════════════════════════════════════════

class TestConcordanceOLS:

    def test_coefficients_match_statsmodels(self, ols_data):
        import statsmodels.api as sm
        from apastats import moderation_analysis

        # apastats: use x1 as predictor, x2 as moderator
        res = moderation_analysis(ols_data, x="x1", w="x2", y="y")

        # statsmodels: replicate the final model (mean-centred, with interaction)
        df = ols_data.copy()
        df["x1_c"] = df["x1"] - df["x1"].mean()
        df["x2_c"] = df["x2"] - df["x2"].mean()
        df["x1x2"] = df["x1_c"] * df["x2_c"]
        X = sm.add_constant(df[["x1_c", "x2_c", "x1x2"]])
        sm_model = sm.OLS(df["y"], X).fit()

        # Compare final step coefficients
        final_summary = res.step_summaries[-1]
        for j, name in enumerate(["const", "x1_c", "x2_c", "x1_c \u00d7 x2_c"]):
            apastats_b = final_summary.loc[name, "b"] if name != "x1_c \u00d7 x2_c" else final_summary.loc["x1_c \u00d7 x2_c", "b"]
            sm_b = sm_model.params.iloc[j]
            assert_allclose(apastats_b, sm_b, atol=1e-8)

    def test_r_squared_matches_statsmodels(self, ols_data):
        import statsmodels.api as sm
        from apastats import moderation_analysis

        res = moderation_analysis(ols_data, x="x1", w="x2", y="y")

        df = ols_data.copy()
        df["x1_c"] = df["x1"] - df["x1"].mean()
        df["x2_c"] = df["x2"] - df["x2"].mean()
        df["x1x2"] = df["x1_c"] * df["x2_c"]
        X = sm.add_constant(df[["x1_c", "x2_c", "x1x2"]])
        sm_model = sm.OLS(df["y"], X).fit()

        assert_allclose(res.r2[-1], sm_model.rsquared, atol=1e-10)

    def test_se_matches_statsmodels(self, ols_data):
        import statsmodels.api as sm
        from apastats import moderation_analysis

        res = moderation_analysis(ols_data, x="x1", w="x2", y="y")

        df = ols_data.copy()
        df["x1_c"] = df["x1"] - df["x1"].mean()
        df["x2_c"] = df["x2"] - df["x2"].mean()
        df["x1x2"] = df["x1_c"] * df["x2_c"]
        X = sm.add_constant(df[["x1_c", "x2_c", "x1x2"]])
        sm_model = sm.OLS(df["y"], X).fit()

        final_summary = res.step_summaries[-1]
        for j, name in enumerate(["const", "x1_c", "x2_c", "x1_c \u00d7 x2_c"]):
            apastats_se = final_summary.loc[name, "se"]
            sm_se = sm_model.bse.iloc[j]
            assert_allclose(apastats_se, sm_se, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════
# Cronbach's alpha: apastats vs pingouin
# ═══════════════════════════════════════════════════════════════════════════

class TestConcordanceAlpha:

    def test_alpha_matches_pingouin(self, reliability_data):
        try:
            import pingouin as pg
        except ImportError:
            pytest.skip("pingouin not installed")

        from apastats import scale_reliability
        items = [f"item{i}" for i in range(1, 6)]

        res = scale_reliability(reliability_data, items=items)
        pg_alpha, _ = pg.cronbach_alpha(reliability_data[items])

        assert_allclose(res.alpha, pg_alpha, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Cohen's d: apastats vs pingouin
# ═══════════════════════════════════════════════════════════════════════════

class TestConcordanceCohensD:

    def test_d_matches_pingouin(self):
        try:
            import pingouin as pg
        except ImportError:
            pytest.skip("pingouin not installed")

        from apastats import cohens_d

        rng = np.random.default_rng(42)
        g1 = rng.normal(1.0, 1.0, 200)
        g2 = rng.normal(0.0, 1.0, 200)

        res = cohens_d(g1, g2)
        pg_d = pg.compute_effsize(g1, g2, eftype="cohen")

        assert_allclose(res.value, pg_d, atol=1e-4)


# ═══════════════════════════════════════════════════════════════════════════
# Mediation: apastats path coefficients vs raw statsmodels OLS
# ═══════════════════════════════════════════════════════════════════════════

class TestConcordanceMediation:

    def test_a_path_matches_statsmodels(self):
        import statsmodels.api as sm
        from apastats import mediation_analysis

        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        m = 0.5 * x + rng.normal(0, 0.7, n)
        y = 0.4 * m + 0.2 * x + rng.normal(0, 0.7, n)
        df = pd.DataFrame({"x": x, "m": m, "y": y})

        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=100, seed=42)

        # Direct statsmodels: M = a*X
        X_a = sm.add_constant(df["x"])
        sm_a = sm.OLS(df["m"], X_a).fit()
        assert_allclose(res.paths["a"].b, sm_a.params.iloc[1], atol=1e-8)

    def test_b_path_matches_statsmodels(self):
        import statsmodels.api as sm
        from apastats import mediation_analysis

        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        m = 0.5 * x + rng.normal(0, 0.7, n)
        y = 0.4 * m + 0.2 * x + rng.normal(0, 0.7, n)
        df = pd.DataFrame({"x": x, "m": m, "y": y})

        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=100, seed=42)

        # Direct statsmodels: Y = c'*X + b*M
        X_b = sm.add_constant(df[["x", "m"]])
        sm_b = sm.OLS(df["y"], X_b).fit()
        assert_allclose(res.paths["b"].b, sm_b.params.iloc[2], atol=1e-8)  # M coeff
        assert_allclose(res.direct_effect.b, sm_b.params.iloc[1], atol=1e-8)  # X coeff (c')

    def test_total_effect_matches_statsmodels(self):
        import statsmodels.api as sm
        from apastats import mediation_analysis

        rng = np.random.default_rng(42)
        n = 500
        x = rng.normal(0, 1, n)
        m = 0.5 * x + rng.normal(0, 0.7, n)
        y = 0.4 * m + 0.2 * x + rng.normal(0, 0.7, n)
        df = pd.DataFrame({"x": x, "m": m, "y": y})

        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=100, seed=42)

        # Direct statsmodels: Y = c*X (no mediator)
        X_c = sm.add_constant(df["x"])
        sm_c = sm.OLS(df["y"], X_c).fit()
        assert_allclose(res.total_effect.b, sm_c.params.iloc[1], atol=1e-8)
