"""Tests for conditional process analysis (moderated mediation)."""

import numpy as np
import pandas as pd
import pytest
from japtools.conditional_process import conditional_process


@pytest.fixture
def model7_data():
    """Data with a known first-stage moderated mediation.

    DGP: M = 0.5*X + 0.3*W + 0.4*X*W + noise
         Y = 0.3*M + 0.1*X + noise
    Conditional indirect = (0.5 + 0.4*w) * 0.3
    IMM = 0.4 * 0.3 = 0.12
    """
    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    m = 0.5 * x + 0.3 * w + 0.4 * x * w + rng.normal(0, 0.7, n)
    y = 0.3 * m + 0.1 * x + rng.normal(0, 0.7, n)
    return pd.DataFrame({"x": x, "w": w, "m": m, "y": y})


@pytest.fixture
def model14_data():
    """Data with a known second-stage moderated mediation.

    DGP: M = 0.6*X + noise
         Y = 0.4*M + 0.2*W + 0.3*M*W + 0.1*X + noise
    Conditional indirect = 0.6 * (0.4 + 0.3*w)
    IMM = 0.6 * 0.3 = 0.18
    """
    rng = np.random.default_rng(42)
    n = 500
    x = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    m = 0.6 * x + rng.normal(0, 0.7, n)
    y = 0.4 * m + 0.2 * w + 0.3 * m * w + 0.1 * x + rng.normal(0, 0.7, n)
    return pd.DataFrame({"x": x, "w": w, "m": m, "y": y})


class TestModel7:
    def test_returns_result(self, model7_data):
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=500, seed=42,
        )
        assert res.model == 7
        assert res.n == 500

    def test_imm_near_true(self, model7_data):
        """IMM should be near 0.12 (= a3 * b = 0.4 * 0.3)."""
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=1000, seed=42,
        )
        assert abs(res.imm - 0.12) < 0.08

    def test_imm_significant(self, model7_data):
        """With a true IMM of 0.12 and N=500, should be significant."""
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=2000, seed=42,
        )
        assert res.imm_significant

    def test_conditional_indirect_at_three_levels(self, model7_data):
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=500, seed=42,
        )
        assert len(res.conditional_indirect) == 3
        labels = [c.w_label for c in res.conditional_indirect]
        assert "\u22121 SD" in labels
        assert "Mean" in labels
        assert "+1 SD" in labels

    def test_indirect_differs_across_w(self, model7_data):
        """The indirect effect should be stronger at higher W."""
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=500, seed=42,
        )
        low = res.conditional_indirect[0].ab
        high = res.conditional_indirect[2].ab
        assert high > low

    def test_direct_unmoderated(self, model7_data):
        """Model 7: direct effect is not moderated."""
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=500, seed=42,
        )
        assert res.conditional_direct is None


class TestModel8:
    def test_conditional_direct(self, model7_data):
        """Model 8: direct effect is moderated by W."""
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=8, n_boot=500, seed=42,
        )
        assert res.conditional_direct is not None
        assert len(res.conditional_direct) == 3

    def test_model_number(self, model7_data):
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=8, n_boot=500, seed=42,
        )
        assert res.model == 8


class TestModel14:
    def test_imm_near_true(self, model14_data):
        """IMM should be near 0.18 (= a * b3 = 0.6 * 0.3)."""
        res = conditional_process(
            model14_data, x="x", m="m", y="y", w="w",
            model=14, n_boot=1000, seed=42,
        )
        assert abs(res.imm - 0.18) < 0.10

    def test_imm_significant(self, model14_data):
        res = conditional_process(
            model14_data, x="x", m="m", y="y", w="w",
            model=14, n_boot=2000, seed=42,
        )
        assert res.imm_significant


class TestModel15:
    def test_runs(self, model14_data):
        res = conditional_process(
            model14_data, x="x", m="m", y="y", w="w",
            model=15, n_boot=500, seed=42,
        )
        assert res.model == 15
        assert res.conditional_direct is not None


class TestConditionalProcessTable:
    def test_table_str(self, model7_data):
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=500, seed=42,
        )
        text = res.table_str
        assert "PROCESS Model 7" in text
        assert "Index of moderated mediation" in text
        assert "Conditional indirect" in text
        assert "mean-centred" in text

    def test_report_method(self, model7_data):
        res = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=500, seed=42,
        )
        report = res.report()
        assert "Index of moderated mediation" in report
        assert "Conditional indirect" in report


class TestConditionalProcessValidation:
    def test_invalid_model(self, model7_data):
        with pytest.raises(ValueError, match="Supported"):
            conditional_process(
                model7_data, x="x", m="m", y="y", w="w",
                model=99, n_boot=100, seed=42,
            )

    def test_duplicate_variables(self, model7_data):
        with pytest.raises(ValueError, match="distinct"):
            conditional_process(
                model7_data, x="x", m="x", y="y", w="w",
                model=7, n_boot=100, seed=42,
            )

    def test_missing_column(self, model7_data):
        with pytest.raises(KeyError):
            conditional_process(
                model7_data, x="x", m="m", y="y", w="missing",
                model=7, n_boot=100, seed=42,
            )

    def test_reproducibility(self, model7_data):
        r1 = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=500, seed=123,
        )
        r2 = conditional_process(
            model7_data, x="x", m="m", y="y", w="w",
            model=7, n_boot=500, seed=123,
        )
        assert r1.imm == r2.imm
        assert r1.imm_ci_lower == r2.imm_ci_lower


class TestConditionalProcessCovariates:
    def test_with_covariates(self, model7_data):
        df = model7_data.copy()
        rng = np.random.default_rng(99)
        df["age"] = rng.normal(35, 8, len(df))
        res = conditional_process(
            df, x="x", m="m", y="y", w="w",
            model=7, covariates=["age"], n_boot=500, seed=42,
        )
        assert res.n == len(df)
        assert "age" in str(res.mediator_coeffs)
