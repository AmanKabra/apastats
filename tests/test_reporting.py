"""Tests for in-text APA reporting strings."""

import numpy as np
import pandas as pd
import pytest
from apastats.formatting import (
    report_regression_coeff,
    report_model_fit,
    report_r2_change,
    report_indirect_effect,
    report_simple_slope,
)
from apastats.moderation import moderation_analysis
from apastats.mediation import mediation_analysis


class TestReportRegressionCoeff:
    def test_basic(self):
        s = report_regression_coeff(0.25, 0.08, 3.13, 0.002, 225, 0.09, 0.41)
        assert "b = 0.25" in s
        assert "SE = 0.08" in s
        assert "t(225)" in s
        assert ".002" in s
        assert "[0.09, 0.41]" in s

    def test_no_ci(self):
        s = report_regression_coeff(0.25, 0.08, 3.13, 0.002, 225)
        assert "CI" not in s

    def test_standardised(self):
        s = report_regression_coeff(0.34, 0.05, 6.53, 0.0001, 225,
                                     standardised=True)
        assert "\u03b2 = .34" in s  # β = .34 (no leading zero)
        assert "b =" not in s

    def test_p_less_than_001(self):
        s = report_regression_coeff(0.50, 0.05, 10.0, 0.00001, 300)
        assert "< .001" in s


class TestReportModelFit:
    def test_basic(self):
        s = report_model_fit(0.22, 21.15, 3, 225, 0.0001)
        assert "R\u00b2 = .22" in s
        assert "F(3, 225)" in s
        assert "< .001" in s


class TestReportR2Change:
    def test_basic(self):
        s = report_r2_change(0.03, 8.92, 1, 224, 0.003)
        assert "\u0394R\u00b2 = .03" in s
        assert "\u0394F(1, 224)" in s
        assert ".003" in s


class TestReportIndirectEffect:
    def test_basic(self):
        s = report_indirect_effect(0.17, 0.07, 0.04, 0.32)
        assert "ab = 0.17" in s
        assert "[0.04, 0.32]" in s

    def test_with_mediator(self):
        s = report_indirect_effect(0.17, 0.07, 0.04, 0.32, mediator="engagement")
        assert "engagement" in s


class TestReportSimpleSlope:
    def test_basic(self):
        s = report_simple_slope("+1 SD", 0.45, 0.06, 7.20, 394, 0.0001, 0.33, 0.57)
        assert "At +1 SD" in s
        assert "b = 0.45" in s
        assert "t(394)" in s
        assert "< .001" in s


class TestModerationReport:
    def test_report_method(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        w = rng.normal(0, 1, n)
        y = 1 + 0.5 * x + 0.3 * w + 0.4 * x * w + rng.normal(0, 0.8, n)
        df = pd.DataFrame({"x": x, "w": w, "y": y})
        res = moderation_analysis(df, x="x", w="w", y="y")
        report = res.report()
        assert "Model fit:" in report
        assert "Interaction:" in report
        assert "Simple slopes:" in report
        assert "R\u00b2" in report


class TestMediationReport:
    def test_report_method(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        m = 0.6 * x + rng.normal(0, 0.8, n)
        y = 0.5 * m + 0.2 * x + rng.normal(0, 0.8, n)
        df = pd.DataFrame({"x": x, "m": m, "y": y})
        res = mediation_analysis(df, x="x", m="m", y="y", n_boot=500, seed=42)
        report = res.report()
        assert "Path a:" in report
        assert "Path b:" in report
        assert "Direct effect" in report
        assert "Total effect" in report
        assert "ab =" in report
        assert "Bootstrap samples:" in report
