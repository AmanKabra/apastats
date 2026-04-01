"""Tests for scale reliability analysis."""

import math
import numpy as np
import pandas as pd
import pytest
from japtools.reliability import scale_reliability


@pytest.fixture
def survey_items():
    """Simulate 5-item scale with known reliability structure."""
    rng = np.random.default_rng(42)
    n = 300
    true_score = rng.normal(3.5, 1.0, n)
    items = {}
    for i in range(1, 6):
        items[f"item{i}"] = true_score + rng.normal(0, 0.4, n)
    return pd.DataFrame(items)


@pytest.fixture
def bad_item_data():
    """Scale with one item that hurts reliability."""
    rng = np.random.default_rng(42)
    n = 200
    true_score = rng.normal(3.0, 1.0, n)
    df = pd.DataFrame({
        "good1": true_score + rng.normal(0, 0.3, n),
        "good2": true_score + rng.normal(0, 0.3, n),
        "good3": true_score + rng.normal(0, 0.3, n),
        "bad": rng.normal(3.0, 1.0, n),  # uncorrelated noise
    })
    return df


class TestScaleReliabilityBasic:
    def test_returns_result(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert res.n_items == 5
        assert res.n_obs == 300

    def test_alpha_high(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert res.alpha > 0.85

    def test_omega_close_to_alpha(self, survey_items):
        """For tau-equivalent items, omega ≈ alpha."""
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert abs(res.omega_total - res.alpha) < 0.10

    def test_cr_computed(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert 0.5 < res.cr < 1.0

    def test_ave_computed(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert 0.0 < res.ave < 1.0


class TestCITC:
    def test_citc_shape(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert len(res.citc) == 5

    def test_citc_positive(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert all(res.citc > 0.3)

    def test_bad_item_low_citc(self, bad_item_data):
        res = scale_reliability(bad_item_data, items=["good1", "good2", "good3", "bad"])
        assert res.citc["bad"] < res.citc["good1"]


class TestAlphaIfDeleted:
    def test_shape(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert len(res.alpha_if_deleted) == 5

    def test_bad_item_increases_alpha(self, bad_item_data):
        """Removing the bad item should increase alpha."""
        res = scale_reliability(bad_item_data, items=["good1", "good2", "good3", "bad"])
        assert res.alpha_if_deleted["bad"] > res.alpha


class TestFactorLoadings:
    def test_loadings_positive(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert all(res.factor_loadings > 0)

    def test_loadings_bounded(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        assert all(res.factor_loadings <= 1.0)


class TestReliabilityOutput:
    def test_table_str(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        text = res.table_str
        assert "Reliability" in text
        assert "\u03b1" in text  # alpha symbol
        assert "\u03c9" in text  # omega symbol
        assert "CITC" in text

    def test_report_method(self, survey_items):
        res = scale_reliability(survey_items, items=["item1", "item2", "item3", "item4", "item5"])
        report = res.report()
        assert "\u03b1 =" in report
        assert "\u03c9 =" in report
        assert "CR =" in report
        assert "AVE =" in report


class TestReliabilityValidation:
    def test_missing_column(self, survey_items):
        with pytest.raises(KeyError):
            scale_reliability(survey_items, items=["missing"])

    def test_too_few_items(self, survey_items):
        with pytest.raises(ValueError, match="2 items"):
            scale_reliability(survey_items, items=["item1"])
