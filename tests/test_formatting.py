"""Tests for APA 7th edition formatting utilities."""

import math
import pytest
from apastats.formatting import (
    fmt_number,
    fmt_p,
    significance_stars,
    fmt_ci,
    fmt_r2,
    fmt_correlation,
    table_note,
)


class TestFmtNumber:
    """APA number formatting rules."""

    # --- Bounded statistics: no leading zero ---
    def test_correlation_positive(self):
        assert fmt_number(0.54, stat_type="r") == ".54"

    def test_correlation_negative(self):
        assert fmt_number(-0.42, stat_type="r") == "-.42"

    def test_alpha_no_leading_zero(self):
        assert fmt_number(0.87, stat_type="alpha") == ".87"

    def test_beta_no_leading_zero(self):
        assert fmt_number(0.34, stat_type="beta") == ".34"

    def test_r2_no_leading_zero(self):
        assert fmt_number(0.24, stat_type="r2") == ".24"

    def test_bounded_shorthand(self):
        assert fmt_number(0.50, stat_type="bounded") == ".50"

    # --- Unbounded statistics: leading zero ---
    def test_mean_leading_zero(self):
        assert fmt_number(3.45, stat_type="m") == "3.45"

    def test_sd_leading_zero(self):
        assert fmt_number(0.82, stat_type="sd") == "0.82"

    def test_b_leading_zero(self):
        assert fmt_number(0.25, stat_type="b") == "0.25"

    def test_se_leading_zero(self):
        assert fmt_number(0.08, stat_type="se") == "0.08"

    def test_t_leading_zero(self):
        assert fmt_number(2.94, stat_type="t") == "2.94"

    def test_f_leading_zero(self):
        assert fmt_number(7.33, stat_type="f") == "7.33"

    def test_unbounded_default(self):
        assert fmt_number(0.50, stat_type="unbounded") == "0.50"

    # --- Decimal places ---
    def test_three_decimals(self):
        assert fmt_number(0.123, stat_type="r", decimals=3) == ".123"

    # --- NaN ---
    def test_nan_returns_empty(self):
        assert fmt_number(float("nan"), stat_type="r") == ""


class TestFmtP:
    """p value formatting per APA 7th."""

    def test_normal_p(self):
        assert fmt_p(0.032) == ".032"

    def test_very_small_p(self):
        assert fmt_p(0.0001) == "< .001"

    def test_exact_boundary(self):
        assert fmt_p(0.001) == ".001"

    def test_p_no_leading_zero(self):
        result = fmt_p(0.500)
        assert not result.startswith("0")

    def test_nan(self):
        assert fmt_p(float("nan")) == ""


class TestSignificanceStars:
    """Star convention: * p < .05, ** p < .01."""

    def test_double_star(self):
        assert significance_stars(0.005) == "**"

    def test_single_star(self):
        assert significance_stars(0.03) == "*"

    def test_no_star(self):
        assert significance_stars(0.10) == ""

    def test_boundary_01(self):
        # p = .01 is NOT < .01, so single star
        assert significance_stars(0.01) == "*"

    def test_boundary_05(self):
        # p = .05 is NOT < .05, so no star
        assert significance_stars(0.05) == ""

    def test_nan(self):
        assert significance_stars(float("nan")) == ""


class TestFmtNumberEdgeCases:
    """Edge cases added after audit."""

    def test_infinity(self):
        assert fmt_number(float("inf"), stat_type="m") == "\u221e"

    def test_negative_infinity(self):
        assert fmt_number(float("-inf"), stat_type="m") == "-\u221e"

    def test_negative_zero(self):
        result = fmt_number(-0.0, stat_type="r")
        assert result == ".00"  # no negative sign for -0.0

    def test_negative_decimals_raises(self):
        with pytest.raises(ValueError, match="decimals"):
            fmt_number(0.5, stat_type="r", decimals=-1)

    def test_very_small_bounded(self):
        # 0.0001 rounds to .00 at 2 decimals
        assert fmt_number(0.0001, stat_type="r", decimals=2) == ".00"

    def test_large_unbounded(self):
        assert fmt_number(12345.67, stat_type="m") == "12345.67"


class TestFmtCi:
    def test_basic(self):
        assert fmt_ci(0.12, 0.56) == "[0.12, 0.56]"

    def test_bounded(self):
        assert fmt_ci(0.04, 0.32, stat_type="bounded") == "[.04, .32]"


class TestFmtR2:
    def test_basic(self):
        assert fmt_r2(0.24) == ".24"


class TestFmtCorrelation:
    def test_significant(self):
        assert fmt_correlation(0.42, 0.001) == ".42**"

    def test_not_significant(self):
        assert fmt_correlation(0.09, 0.20) == ".09"


class TestTableNote:
    def test_basic(self):
        note = table_note(350)
        assert "N = 350" in note
        assert "Cronbach's alpha" in note
        assert "*p < .05" in note

    def test_no_alpha(self):
        note = table_note(200, alpha_on_diagonal=False)
        assert "alpha" not in note
