"""Tests for the descriptives + correlations table."""

import numpy as np
import pandas as pd
import pytest
from apastats.descriptives import descriptives_table, cronbach_alpha


@pytest.fixture
def sample_data():
    """Create reproducible sample data mimicking OB survey scales."""
    rng = np.random.default_rng(42)
    n = 300
    # Correlated variables
    cov = [[1.0, 0.5, -0.3],
           [0.5, 1.0, -0.4],
           [-0.3, -0.4, 1.0]]
    raw = rng.multivariate_normal([3.5, 3.8, 2.1], cov, size=n)
    df = pd.DataFrame(raw, columns=["satisfaction", "commitment", "turnover"])
    df["age"] = rng.normal(34, 8, n)
    # Item-level data for alpha computation
    for var in ["satisfaction", "commitment", "turnover"]:
        for i in range(4):
            df[f"{var}_item{i+1}"] = df[var] + rng.normal(0, 0.3, n)
    return df


class TestCronbachAlpha:
    def test_known_value(self):
        """High inter-item correlation should yield high alpha."""
        rng = np.random.default_rng(0)
        true_score = rng.normal(0, 1, 500)
        items = np.column_stack([true_score + rng.normal(0, 0.3, 500) for _ in range(5)])
        a = cronbach_alpha(items)
        assert 0.85 < a < 1.0

    def test_two_items_minimum(self):
        with pytest.raises(ValueError, match="at least 2"):
            cronbach_alpha(np.array([[1], [2], [3]]))

    def test_handles_nan(self):
        """Rows with NaN should be dropped."""
        arr = np.array([[1, 2], [3, 4], [np.nan, 5], [6, 7]])
        a = cronbach_alpha(arr)
        assert 0 <= a <= 1


class TestDescriptivesTable:
    def test_basic_output(self, sample_data):
        res = descriptives_table(
            sample_data,
            variables=["satisfaction", "commitment", "turnover", "age"],
            labels=["Job satisfaction", "Org. commitment", "Turnover intent", "Age"],
        )
        # Check structure
        assert res.n > 0
        assert len(res.means) == 4
        assert len(res.sds) == 4
        assert res.correlations.shape == (4, 4)

    def test_table_str_format(self, sample_data):
        res = descriptives_table(
            sample_data,
            variables=["satisfaction", "commitment", "turnover"],
        )
        text = res.table_str
        # Must contain Table 1 header
        assert "Table 1" in text
        # Must contain horizontal rules (box-drawing chars)
        assert "\u2500" in text
        # Must have significance note
        assert "*p < .05" in text

    def test_alpha_on_diagonal(self, sample_data):
        """Pre-computed alpha should appear in parentheses on diagonal."""
        res = descriptives_table(
            sample_data,
            variables=["satisfaction", "commitment"],
            alphas={"satisfaction": 0.87, "commitment": 0.91},
        )
        df = res.table_df
        assert "(.87)" in df["1"].values[0]
        assert "(.91)" in df["2"].values[1]

    def test_alpha_from_items(self, sample_data):
        """Alpha computed from item columns."""
        items = ["satisfaction_item1", "satisfaction_item2",
                 "satisfaction_item3", "satisfaction_item4"]
        res = descriptives_table(
            sample_data,
            variables=["satisfaction", "commitment"],
            alphas={"satisfaction": items},
        )
        a = res.alphas["satisfaction"]
        assert a is not None and 0 < a < 1

    def test_em_dash_for_no_alpha(self, sample_data):
        """Variables without alpha get an em-dash on diagonal."""
        res = descriptives_table(
            sample_data,
            variables=["satisfaction", "age"],
            alphas={"satisfaction": 0.87},
        )
        df = res.table_df
        assert df["2"].values[1] == "\u2014"

    def test_lower_triangle_only(self, sample_data):
        """Upper triangle should be empty."""
        res = descriptives_table(
            sample_data,
            variables=["satisfaction", "commitment", "turnover"],
        )
        df = res.table_df
        # Row 0 (first variable): columns 2, 3 should be empty
        assert df["2"].values[0] == ""
        assert df["3"].values[0] == ""

    def test_no_leading_zero_in_correlations(self, sample_data):
        """Correlation values must not have leading zeros."""
        res = descriptives_table(
            sample_data,
            variables=["satisfaction", "commitment"],
        )
        df = res.table_df
        r_val = df["1"].values[1]  # row 1, col 1 = correlation
        # Should start with . or -. or stars, not 0.
        clean = r_val.replace("*", "")
        assert not clean.startswith("0.")

    def test_leading_zero_in_means(self, sample_data):
        """M and SD should have leading zeros when < 1."""
        rng = np.random.default_rng(99)
        df_small = pd.DataFrame({"x": rng.normal(0.5, 0.2, 100)})
        res = descriptives_table(df_small, variables=["x"])
        m_str = res.table_df["M"].values[0]
        assert m_str == "0.50" or m_str.startswith("0.")

    def test_missing_column_raises(self, sample_data):
        with pytest.raises(KeyError):
            descriptives_table(sample_data, variables=["nonexistent"])

    def test_label_mismatch_raises(self, sample_data):
        with pytest.raises(ValueError, match="labels"):
            descriptives_table(
                sample_data,
                variables=["satisfaction"],
                labels=["Too", "Many"],
            )

    def test_spearman(self, sample_data):
        res = descriptives_table(
            sample_data,
            variables=["satisfaction", "commitment"],
            method="spearman",
        )
        assert res.correlations.shape == (2, 2)


class TestCronbachAlphaEdgeCases:
    def test_zero_variance_returns_nan(self):
        """All identical responses -> alpha is undefined (NaN)."""
        import math
        arr = np.ones((100, 4))  # all items = 1
        a = cronbach_alpha(arr)
        assert math.isnan(a)

    def test_single_variable_table(self):
        """1-variable table should still work (1x1 matrix)."""
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(3, 1, 50)})
        res = descriptives_table(df, variables=["x"])
        assert res.correlations.shape == (1, 1)
        assert "Table 1" in res.table_str

    def test_pairwise_n_warning(self):
        """Unequal missing data should produce a pairwise-N warning."""
        import warnings
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "a": rng.normal(0, 1, 100),
            "b": rng.normal(0, 1, 100),
        })
        # Add missing values only to 'b'
        df.loc[90:, "b"] = np.nan
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = descriptives_table(df, variables=["a", "b"])
            pairwise_warnings = [x for x in w if "Pairwise N" in str(x.message)]
            assert len(pairwise_warnings) >= 1
