"""Tests for effect size calculations."""

import math
import numpy as np
import pytest
from japtools.effect_sizes import cohens_d, cohens_f2, r2_effect, partial_eta_squared


class TestCohensD:
    def test_known_effect(self):
        """Two groups with known mean difference."""
        rng = np.random.default_rng(42)
        g1 = rng.normal(1.0, 1.0, 200)
        g2 = rng.normal(0.0, 1.0, 200)
        result = cohens_d(g1, g2)
        # d should be near 1.0
        assert abs(result.value - 1.0) < 0.25

    def test_zero_effect(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0, 1, 500)
        g2 = rng.normal(0, 1, 500)
        result = cohens_d(g1, g2)
        assert abs(result.value) < 0.2

    def test_ci_contains_estimate(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0.5, 1, 100)
        g2 = rng.normal(0.0, 1, 100)
        result = cohens_d(g1, g2)
        assert result.ci_lower < result.value < result.ci_upper

    def test_interpretation_large(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(2.0, 1, 200)
        g2 = rng.normal(0.0, 1, 200)
        result = cohens_d(g1, g2)
        assert result.interpretation == "large"

    def test_interpretation_small(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0.3, 1, 500)
        g2 = rng.normal(0.0, 1, 500)
        result = cohens_d(g1, g2)
        assert result.interpretation == "small"

    def test_handles_nan(self):
        g1 = np.array([1.0, 2.0, np.nan, 3.0])
        g2 = np.array([0.0, 1.0, 2.0])
        result = cohens_d(g1, g2)
        assert not math.isnan(result.value)

    def test_repr(self):
        rng = np.random.default_rng(42)
        result = cohens_d(rng.normal(0.5, 1, 50), rng.normal(0, 1, 50))
        text = repr(result)
        assert "Cohen's d" in text
        assert "CI" in text


class TestCohensF2:
    def test_overall_model(self):
        result = cohens_f2(0.25)
        # f2 = 0.25 / 0.75 = 0.333
        assert abs(result.value - 0.333) < 0.01
        assert result.interpretation == "medium"

    def test_incremental(self):
        result = cohens_f2(0.30, 0.25)
        # f2 = 0.05 / 0.70 = 0.071
        assert abs(result.value - 0.071) < 0.01
        assert result.interpretation == "small"

    def test_zero_r2(self):
        result = cohens_f2(0.0)
        assert result.value == 0.0
        assert result.interpretation == "negligible"


class TestR2Effect:
    def test_small(self):
        assert r2_effect(0.05).interpretation == "small"

    def test_medium(self):
        assert r2_effect(0.15).interpretation == "medium"

    def test_large(self):
        assert r2_effect(0.30).interpretation == "large"

    def test_negligible(self):
        assert r2_effect(0.01).interpretation == "negligible"


class TestPartialEtaSquared:
    def test_basic(self):
        result = partial_eta_squared(100, 400)
        assert abs(result.value - 0.20) < 0.01
        assert result.interpretation == "medium"

    def test_zero(self):
        result = partial_eta_squared(0, 100)
        assert result.value == 0.0
