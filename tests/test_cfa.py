"""Tests for confirmatory factor analysis."""

import math
import numpy as np
import pandas as pd
import pytest
from japtools.cfa import cfa, FitIndices


@pytest.fixture
def two_factor_data():
    """Simulate two-factor CFA data with clear factor structure."""
    rng = np.random.default_rng(42)
    n = 400
    # Factor 1: POS
    f1 = rng.normal(0, 1, n)
    pos1 = 0.8 * f1 + rng.normal(0, 0.5, n)
    pos2 = 0.75 * f1 + rng.normal(0, 0.5, n)
    pos3 = 0.7 * f1 + rng.normal(0, 0.5, n)
    pos4 = 0.72 * f1 + rng.normal(0, 0.5, n)

    # Factor 2: Commitment
    f2 = 0.4 * f1 + rng.normal(0, 0.8, n)  # moderately correlated factors
    com1 = 0.78 * f2 + rng.normal(0, 0.5, n)
    com2 = 0.82 * f2 + rng.normal(0, 0.5, n)
    com3 = 0.75 * f2 + rng.normal(0, 0.5, n)
    com4 = 0.70 * f2 + rng.normal(0, 0.5, n)

    return pd.DataFrame({
        "pos1": pos1, "pos2": pos2, "pos3": pos3, "pos4": pos4,
        "com1": com1, "com2": com2, "com3": com3, "com4": com4,
    })


@pytest.fixture
def two_factor_spec():
    return {
        "POS": ["pos1", "pos2", "pos3", "pos4"],
        "Commit": ["com1", "com2", "com3", "com4"],
    }


class TestCFAFitIndices:
    def test_fit_indices_present(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        assert not math.isnan(res.fit.chi2)
        assert res.fit.df > 0
        # CFI can slightly exceed 1.0 for very well-fitting models
        assert res.fit.cfi > 0.5

    def test_good_fit_on_clean_data(self, two_factor_data, two_factor_spec):
        """Clean two-factor data should yield good fit."""
        res = cfa(two_factor_data, two_factor_spec)
        assert res.fit.cfi > 0.90

    def test_rmsea_ci(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        assert res.fit.rmsea_ci_lower <= res.fit.rmsea <= res.fit.rmsea_ci_upper

    def test_fit_report_string(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        report = res.fit.report()
        assert "\u03c7\u00b2" in report  # chi-squared symbol
        assert "CFI" in report
        assert "RMSEA" in report
        assert "SRMR" in report

    def test_fit_interpretation(self):
        fit = FitIndices(
            chi2=100, df=50, p=0.001,
            cfi=0.97, tli=0.96, rmsea=0.04,
            rmsea_ci_lower=0.02, rmsea_ci_upper=0.06,
            srmr=0.03,
        )
        assert fit.interpretation() == "good"

    def test_fit_poor(self):
        fit = FitIndices(
            chi2=500, df=50, p=0.001,
            cfi=0.80, tli=0.78, rmsea=0.12,
            rmsea_ci_lower=0.10, rmsea_ci_upper=0.14,
            srmr=0.15,
        )
        assert fit.interpretation() == "poor"


class TestCFALoadings:
    def test_loadings_shape(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        assert res.loadings_df.shape == (8, 2)  # 8 items x 2 factors

    def test_loadings_on_correct_factor(self, two_factor_data, two_factor_spec):
        """POS items should load on POS, not Commit."""
        res = cfa(two_factor_data, two_factor_spec)
        for item in ["pos1", "pos2", "pos3", "pos4"]:
            assert abs(res.loadings_df.loc[item, "POS"]) > 0.3
            assert abs(res.loadings_df.loc[item, "Commit"]) < 0.01


class TestCFAValidity:
    def test_cr_per_factor(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        assert "POS" in res.cr
        assert "Commit" in res.cr
        assert res.cr["POS"] > 0.5
        assert res.cr["Commit"] > 0.5

    def test_ave_per_factor(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        assert "POS" in res.ave
        assert res.ave["POS"] > 0.0

    def test_fornell_larcker_shape(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        assert res.fornell_larcker_df.shape == (2, 2)

    def test_fornell_larcker_diagonal(self, two_factor_data, two_factor_spec):
        """Diagonal should be sqrt(AVE)."""
        res = cfa(two_factor_data, two_factor_spec)
        for i, factor in enumerate(res.factor_names):
            expected = math.sqrt(res.ave[factor])
            actual = res.fornell_larcker_df.iloc[i, i]
            assert abs(actual - expected) < 0.01

    def test_htmt_shape(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        assert res.htmt_df.shape == (2, 2)

    def test_htmt_diagonal_is_one(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        for i in range(len(res.factor_names)):
            assert abs(res.htmt_df.iloc[i, i] - 1.0) < 0.01


class TestCFAModelComparison:
    def test_compare_returns_dict(self, two_factor_data, two_factor_spec):
        res1 = cfa(two_factor_data, two_factor_spec)
        # Single factor model for comparison
        one_factor = {"General": ["pos1", "pos2", "pos3", "pos4", "com1", "com2", "com3", "com4"]}
        res2 = cfa(two_factor_data, one_factor)
        comp = res1.compare(res2)
        assert "delta_chi2" in comp
        assert "delta_df" in comp
        assert "p" in comp
        assert "significant" in comp


class TestCFAOutput:
    def test_table_str(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        text = res.table_str
        assert "Confirmatory Factor Analysis" in text
        assert "CFI" in text
        assert "CR" in text
        assert "AVE" in text

    def test_report_method(self, two_factor_data, two_factor_spec):
        res = cfa(two_factor_data, two_factor_spec)
        report = res.report()
        assert "CFI" in report
        assert "POS" in report
        assert "CR" in report


class TestCFAValidation:
    def test_missing_column(self, two_factor_data):
        with pytest.raises(KeyError):
            cfa(two_factor_data, {"F1": ["missing1", "missing2", "missing3"]})

    def test_shared_items(self, two_factor_data):
        with pytest.raises(ValueError, match="shared"):
            cfa(two_factor_data, {
                "F1": ["pos1", "pos2", "pos3"],
                "F2": ["pos3", "com1", "com2"],  # pos3 shared
            })

    def test_too_few_indicators(self, two_factor_data):
        with pytest.raises(ValueError, match="2 indicators"):
            cfa(two_factor_data, {"F1": ["pos1"]})
