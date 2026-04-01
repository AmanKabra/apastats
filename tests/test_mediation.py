"""Tests for mediation analysis."""

import numpy as np
import pandas as pd
import pytest
from apastats.mediation import mediation_analysis


@pytest.fixture
def single_mediator_data():
    """Data with a known mediation effect: X -> M -> Y."""
    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(0, 1, n)
    m = 0.6 * x + rng.normal(0, 0.8, n)       # a = 0.6
    y = 0.5 * m + 0.2 * x + rng.normal(0, 0.8, n)  # b = 0.5, c' = 0.2
    # indirect = a*b = 0.3, total = c' + ab = 0.5
    return pd.DataFrame({"x": x, "m": m, "y": y})


@pytest.fixture
def parallel_mediator_data():
    """Data with two parallel mediators."""
    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(0, 1, n)
    m1 = 0.5 * x + rng.normal(0, 0.7, n)
    m2 = 0.3 * x + rng.normal(0, 0.7, n)
    y = 0.4 * m1 + 0.3 * m2 + 0.1 * x + rng.normal(0, 0.7, n)
    return pd.DataFrame({"x": x, "m1": m1, "m2": m2, "y": y})


class TestSingleMediator:
    def test_returns_result(self, single_mediator_data):
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=1000, seed=42,
        )
        assert res.n == 500
        assert res.n_boot == 1000
        assert len(res.m_names) == 1

    def test_paths_exist(self, single_mediator_data):
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=1000, seed=42,
        )
        assert "a" in res.paths
        assert "b" in res.paths

    def test_a_path_recoverable(self, single_mediator_data):
        """a path should be near 0.6."""
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=1000, seed=42,
        )
        assert abs(res.paths["a"].b - 0.6) < 0.15

    def test_b_path_recoverable(self, single_mediator_data):
        """b path should be near 0.5."""
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=1000, seed=42,
        )
        assert abs(res.paths["b"].b - 0.5) < 0.15

    def test_indirect_significant(self, single_mediator_data):
        """Indirect effect should be significant (CI excludes zero)."""
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=2000, seed=42,
        )
        ie = res.indirect_effects[0]
        assert ie.significant

    def test_indirect_near_true(self, single_mediator_data):
        """Indirect ab should be near 0.3."""
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=1000, seed=42,
        )
        assert abs(res.indirect_effects[0].ab - 0.3) < 0.15

    def test_total_effect(self, single_mediator_data):
        """Total effect c ≈ c' + ab."""
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=1000, seed=42,
        )
        c = res.total_effect.b
        c_prime = res.direct_effect.b
        ab = res.total_indirect.ab
        assert abs(c - (c_prime + ab)) < 0.05

    def test_no_t_or_p_for_indirect(self, single_mediator_data):
        """Table should show em-dashes for t and p of indirect effect."""
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=1000, seed=42,
        )
        assert "\u2014" in res.table_str  # em-dash present


class TestParallelMediators:
    def test_two_mediators(self, parallel_mediator_data):
        res = mediation_analysis(
            parallel_mediator_data, x="x", m=["m1", "m2"], y="y",
            n_boot=1000, seed=42,
        )
        assert len(res.indirect_effects) == 2
        assert len(res.m_names) == 2

    def test_specific_indirect_keys(self, parallel_mediator_data):
        res = mediation_analysis(
            parallel_mediator_data, x="x", m=["m1", "m2"], y="y",
            n_boot=1000, seed=42,
        )
        assert "a1" in res.paths
        assert "a2" in res.paths
        assert "b1" in res.paths
        assert "b2" in res.paths

    def test_total_indirect_sum(self, parallel_mediator_data):
        """Total indirect ≈ sum of specific indirect effects."""
        res = mediation_analysis(
            parallel_mediator_data, x="x", m=["m1", "m2"], y="y",
            n_boot=1000, seed=42,
        )
        specific_sum = sum(ie.ab for ie in res.indirect_effects)
        assert abs(res.total_indirect.ab - specific_sum) < 0.01


class TestMediationTable:
    def test_table_str_nonempty(self, single_mediator_data):
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=1000, seed=42,
        )
        assert len(res.table_str) > 100

    def test_table_has_structure(self, single_mediator_data):
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=1000, seed=42,
        )
        text = res.table_str
        assert "Table 3" in text
        assert "Bootstrap" in text or "bootstrap" in text
        assert "CI" in text

    def test_bootstrap_count_in_note(self, single_mediator_data):
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=5000, seed=42,
        )
        assert "5,000" in res.table_str


class TestMediationPlot:
    def test_plot_returns_figure(self, single_mediator_data):
        import matplotlib
        matplotlib.use("Agg")
        res = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=500, seed=42,
        )
        fig = res.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")


class TestMediationWithCovariates:
    def test_covariates(self, single_mediator_data):
        df = single_mediator_data.copy()
        rng = np.random.default_rng(99)
        df["age"] = rng.normal(35, 8, len(df))
        res = mediation_analysis(
            df, x="x", m="m", y="y", covariates=["age"],
            n_boot=500, seed=42,
        )
        assert res.n == len(df)


class TestMediationEdgeCases:
    def test_missing_column(self, single_mediator_data):
        with pytest.raises(KeyError):
            mediation_analysis(
                single_mediator_data, x="x", m="missing", y="y",
                n_boot=100, seed=42,
            )

    def test_reproducibility(self, single_mediator_data):
        """Same seed should give identical results."""
        r1 = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=500, seed=123,
        )
        r2 = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=500, seed=123,
        )
        assert r1.indirect_effects[0].ab == r2.indirect_effects[0].ab
        assert r1.indirect_effects[0].ci_lower == r2.indirect_effects[0].ci_lower

    def test_duplicate_mediators_raises(self, single_mediator_data):
        """Duplicate mediator names should be rejected."""
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            mediation_analysis(
                single_mediator_data, x="x", m=["m", "m"], y="y",
                n_boot=100, seed=42,
            )

    def test_overlapping_variables_raises(self, single_mediator_data):
        """x, y, m must all be distinct."""
        with pytest.raises(ValueError, match="distinct"):
            mediation_analysis(
                single_mediator_data, x="x", m="x", y="y",
                n_boot=100, seed=42,
            )

    def test_ci_level_90(self, single_mediator_data):
        """90% CI should be narrower than 95% CI."""
        r95 = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=2000, seed=42, ci_level=0.95,
        )
        r90 = mediation_analysis(
            single_mediator_data, x="x", m="m", y="y",
            n_boot=2000, seed=42, ci_level=0.90,
        )
        width_95 = r95.indirect_effects[0].ci_upper - r95.indirect_effects[0].ci_lower
        width_90 = r90.indirect_effects[0].ci_upper - r90.indirect_effects[0].ci_lower
        assert width_90 < width_95

    def test_nan_drops_with_warning(self, single_mediator_data):
        """Missing values should be dropped with a warning."""
        import warnings
        df = single_mediator_data.copy()
        df.loc[0, "x"] = np.nan
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = mediation_analysis(
                df, x="x", m="m", y="y",
                n_boot=100, seed=42,
            )
        assert res.n == 499
        drop_warnings = [x for x in w if "dropped" in str(x.message)]
        assert len(drop_warnings) >= 1
