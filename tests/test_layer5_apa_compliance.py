"""
Layer 5: APA 7th Edition Formatting Compliance.

Every formatted string the package produces is checked against APA rules
using regex and string assertions.  This layer is independent of
statistical correctness (that is Layers 1-3) and focuses entirely on
whether the output *looks* right.

Rules verified:
  - No leading zero on bounded statistics (r, alpha, beta, p, R2)
  - Leading zero on unbounded statistics (M, SD, b, SE, t, F, d)
  - p values: exact to 3 decimals, "< .001" for very small, never ".000"
  - Significance stars: ** for p < .01, * for p < .05, none otherwise
  - CI format: [lower, upper] with square brackets
  - Table structure: horizontal rules only, no vertical lines
  - Table notes: "Note." followed by N, then alpha note, then stars
"""

import re
import math
import numpy as np
import pandas as pd
import pytest

from apastats.formatting import (
    fmt_number,
    fmt_p,
    fmt_ci,
    fmt_r2,
    fmt_correlation,
    significance_stars,
    report_regression_coeff,
    report_model_fit,
    report_indirect_effect,
)


# ═══════════════════════════════════════════════════════════════════════════
# No Leading Zero on Bounded Statistics
# ═══════════════════════════════════════════════════════════════════════════

class TestNoLeadingZeroBounded:
    """APA 7th: statistics that cannot exceed |1| have no leading zero."""

    @pytest.mark.parametrize("stat_type", ["r", "alpha", "beta", "p", "r2", "delta_r2", "sr2"])
    def test_positive_bounded_no_leading_zero(self, stat_type):
        result = fmt_number(0.54, stat_type=stat_type)
        assert not result.startswith("0"), f"{stat_type}: got '{result}', should not start with 0"
        assert result.startswith("."), f"{stat_type}: got '{result}', should start with ."

    @pytest.mark.parametrize("stat_type", ["r", "alpha", "beta"])
    def test_negative_bounded_no_leading_zero(self, stat_type):
        result = fmt_number(-0.42, stat_type=stat_type)
        assert result.startswith("-."), f"{stat_type}: got '{result}', should start with -."
        assert "-0." not in result

    @pytest.mark.parametrize("value", [0.01, 0.10, 0.50, 0.99])
    def test_p_value_no_leading_zero(self, value):
        result = fmt_p(value)
        assert "0." not in result.replace("< ", ""), f"p={value}: got '{result}'"

    def test_r2_no_leading_zero(self):
        assert fmt_r2(0.24) == ".24"
        assert fmt_r2(0.05) == ".05"


# ═══════════════════════════════════════════════════════════════════════════
# Leading Zero on Unbounded Statistics
# ═══════════════════════════════════════════════════════════════════════════

class TestLeadingZeroUnbounded:
    """APA 7th: statistics that can exceed |1| have a leading zero."""

    @pytest.mark.parametrize("stat_type", ["m", "sd", "b", "se", "t", "f", "d"])
    def test_positive_unbounded_has_leading_zero(self, stat_type):
        result = fmt_number(0.54, stat_type=stat_type)
        assert result.startswith("0."), f"{stat_type}: got '{result}', should start with 0."

    @pytest.mark.parametrize("stat_type", ["m", "sd", "b", "se"])
    def test_negative_unbounded_has_leading_zero(self, stat_type):
        result = fmt_number(-0.54, stat_type=stat_type)
        assert result.startswith("-0."), f"{stat_type}: got '{result}', should start with -0."


# ═══════════════════════════════════════════════════════════════════════════
# P Value Formatting
# ═══════════════════════════════════════════════════════════════════════════

class TestPValueFormatting:

    def test_exact_three_decimals(self):
        result = fmt_p(0.032)
        assert result == ".032"

    def test_very_small_uses_less_than(self):
        result = fmt_p(0.0001)
        assert result == "< .001"

    def test_never_dot_zero_zero_zero(self):
        """APA: never report p = .000."""
        result = fmt_p(0.00001)
        assert ".000" not in result or "< .001" in result

    def test_p_001_is_exact(self):
        result = fmt_p(0.001)
        assert result == ".001"

    def test_p_at_boundary(self):
        result = fmt_p(0.050)
        assert result == ".050"


# ═══════════════════════════════════════════════════════════════════════════
# Significance Stars
# ═══════════════════════════════════════════════════════════════════════════

class TestSignificanceStarsCompliance:

    def test_double_star_strictly_below_01(self):
        assert significance_stars(0.009) == "**"
        assert significance_stars(0.01) == "*"   # NOT **, because .01 is not < .01

    def test_single_star_strictly_below_05(self):
        assert significance_stars(0.049) == "*"
        assert significance_stars(0.05) == ""    # NOT *, because .05 is not < .05

    def test_no_star_above_05(self):
        assert significance_stars(0.10) == ""
        assert significance_stars(0.50) == ""
        assert significance_stars(0.99) == ""

    def test_stars_in_correlation_table(self):
        """Correlation with p < .01 should have ** appended."""
        result = fmt_correlation(0.42, 0.001)
        assert result.endswith("**")
        assert result == ".42**"

    def test_no_stars_in_nonsig_correlation(self):
        result = fmt_correlation(0.05, 0.30)
        assert "*" not in result


# ═══════════════════════════════════════════════════════════════════════════
# Confidence Interval Format
# ═══════════════════════════════════════════════════════════════════════════

class TestCIFormat:

    def test_square_brackets(self):
        result = fmt_ci(0.12, 0.56)
        assert result.startswith("[") and result.endswith("]")

    def test_comma_separator(self):
        result = fmt_ci(0.12, 0.56)
        assert ", " in result

    def test_bounded_ci_no_leading_zero(self):
        result = fmt_ci(0.04, 0.32, stat_type="bounded")
        assert result == "[.04, .32]"

    def test_unbounded_ci_has_leading_zero(self):
        result = fmt_ci(0.12, 0.56, stat_type="unbounded")
        assert result == "[0.12, 0.56]"


# ═══════════════════════════════════════════════════════════════════════════
# Table Structure Compliance
# ═══════════════════════════════════════════════════════════════════════════

class TestTableStructure:

    @pytest.fixture
    def desc_table(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "x": rng.normal(3, 1, 200),
            "y": rng.normal(2, 1, 200),
        })
        from apastats import descriptives_table
        return descriptives_table(df, variables=["x", "y"])

    def test_no_vertical_lines(self, desc_table):
        assert "|" not in desc_table.table_str

    def test_has_horizontal_rules(self, desc_table):
        """Should have box-drawing horizontal rules."""
        assert "\u2500" in desc_table.table_str

    def test_three_horizontal_rules(self, desc_table):
        """APA tables have exactly 3 horizontal rules: top, below header, bottom."""
        lines = desc_table.table_str.split("\n")
        rule_lines = [l for l in lines if l.strip() and all(c == "\u2500" for c in l.strip())]
        assert len(rule_lines) == 3

    def test_table_note_starts_with_note(self, desc_table):
        text = desc_table.table_str
        assert "Note." in text

    def test_table_note_has_n(self, desc_table):
        text = desc_table.table_str
        assert "N = " in text

    def test_table_note_ends_with_stars(self, desc_table):
        text = desc_table.table_str
        assert "*p < .05. **p < .01." in text

    def test_table_has_title(self, desc_table):
        text = desc_table.table_str
        assert "Table 1" in text


# ═══════════════════════════════════════════════════════════════════════════
# In-Text Reporting String Compliance
# ═══════════════════════════════════════════════════════════════════════════

class TestInTextCompliance:

    def test_regression_coeff_format(self):
        s = report_regression_coeff(0.25, 0.08, 3.13, 0.002, 225, 0.09, 0.41)
        # Should have: b = 0.25, SE = 0.08, t(225) = 3.13, p = .002, 95% CI [0.09, 0.41]
        assert "b = 0.25" in s
        assert "SE = 0.08" in s
        assert "t(225)" in s
        assert ".002" in s
        # p should NOT have leading zero
        assert "p 0." not in s and "p = 0." not in s
        assert "95% CI" in s
        assert "[0.09, 0.41]" in s

    def test_standardised_coeff_no_leading_zero(self):
        s = report_regression_coeff(0.34, 0.05, 6.53, 0.001, 225, standardised=True)
        # beta = .34 (no leading zero)
        assert "\u03b2 = .34" in s
        assert "\u03b2 = 0.34" not in s

    def test_model_fit_format(self):
        s = report_model_fit(0.22, 21.15, 3, 225, 0.0001)
        assert "R\u00b2 = .22" in s  # no leading zero on R2
        assert "F(3, 225)" in s
        assert "< .001" in s

    def test_indirect_effect_format(self):
        s = report_indirect_effect(0.17, 0.07, 0.04, 0.32)
        assert "ab = 0.17" in s  # leading zero on b (unbounded)
        assert "[0.04, 0.32]" in s


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline: Moderation Table APA Compliance
# ═══════════════════════════════════════════════════════════════════════════

class TestModerationTableCompliance:

    @pytest.fixture
    def mod_result(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        w = rng.normal(0, 1, n)
        y = 1 + 0.5 * x + 0.3 * w + 0.3 * x * w + rng.normal(0, 0.8, n)
        df = pd.DataFrame({"x": x, "w": w, "y": y})
        from apastats import moderation_analysis
        return moderation_analysis(df, x="x", w="w", y="y")

    def test_no_vertical_lines(self, mod_result):
        assert "|" not in mod_result.table_str

    def test_has_r_squared(self, mod_result):
        assert "R\u00b2" in mod_result.table_str

    def test_has_delta_r_squared(self, mod_result):
        assert "\u0394R\u00b2" in mod_result.table_str

    def test_star_note_present(self, mod_result):
        assert "*p < .05. **p < .01." in mod_result.table_str

    def test_mean_centred_note(self, mod_result):
        text = mod_result.table_str.lower()
        assert "mean-centred" in text or "mean centred" in text


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline: Mediation Table APA Compliance
# ═══════════════════════════════════════════════════════════════════════════

class TestMediationTableCompliance:

    @pytest.fixture
    def med_result(self):
        rng = np.random.default_rng(42)
        n = 300
        x = rng.normal(0, 1, n)
        m = 0.5 * x + rng.normal(0, 0.8, n)
        y = 0.4 * m + 0.2 * x + rng.normal(0, 0.8, n)
        df = pd.DataFrame({"x": x, "m": m, "y": y})
        from apastats import mediation_analysis
        return mediation_analysis(df, x="x", m="m", y="y", n_boot=1000, seed=42)

    def test_no_vertical_lines(self, med_result):
        assert "|" not in med_result.table_str

    def test_bootstrap_n_reported(self, med_result):
        assert "1,000" in med_result.table_str or "10,000" in med_result.table_str

    def test_ci_label_present(self, med_result):
        assert "CI" in med_result.table_str

    def test_indirect_has_em_dash_for_t_and_p(self, med_result):
        """Indirect effect row should show em dash for t and p (no test stat)."""
        assert "\u2014" in med_result.table_str

    def test_percentile_bootstrap_noted(self, med_result):
        text = med_result.table_str.lower()
        assert "percentile" in text or "bootstrap" in text
