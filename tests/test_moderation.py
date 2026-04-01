"""Tests for moderation analysis."""

import numpy as np
import pandas as pd
import pytest
from apastats.moderation import moderation_analysis


@pytest.fixture
def interaction_data():
    """Data with a known interaction effect."""
    rng = np.random.default_rng(42)
    n = 400
    x = rng.normal(0, 1, n)
    w = rng.normal(0, 1, n)
    # Y = 1 + 0.5*X + 0.3*W + 0.4*X*W + noise
    y = 1 + 0.5 * x + 0.3 * w + 0.4 * x * w + rng.normal(0, 0.8, n)
    return pd.DataFrame({"x": x, "w": w, "y": y})


@pytest.fixture
def controlled_data():
    """Data with controls + interaction."""
    rng = np.random.default_rng(42)
    n = 300
    age = rng.normal(35, 8, n)
    gender = rng.binomial(1, 0.5, n).astype(float)
    x = rng.normal(3, 1, n)
    w = rng.normal(3, 1, n)
    y = 2 + 0.1 * age + 0.3 * gender + 0.5 * x + 0.2 * w + 0.3 * (x - 3) * (w - 3) + rng.normal(0, 1, n)
    return pd.DataFrame({"age": age, "gender": gender, "x": x, "w": w, "y": y})


class TestModerationBasic:
    def test_returns_result(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        assert res.n == 400
        assert res.x_name == "x"
        assert res.w_name == "w"

    def test_interaction_significant(self, interaction_data):
        """With true interaction of 0.4, should be significant."""
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        assert res.interaction_p < 0.01

    def test_interaction_coefficient_recoverable(self, interaction_data):
        """Estimated interaction b should be near 0.4."""
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        assert abs(res.interaction_b - 0.4) < 0.15

    def test_step_count_no_controls(self, interaction_data):
        """Without controls: 2 steps (main effects, interaction)."""
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        assert len(res.steps) == 2

    def test_step_count_with_controls(self, controlled_data):
        """With controls: 3 steps."""
        res = moderation_analysis(
            controlled_data, x="x", w="w", y="y",
            controls=["age", "gender"],
        )
        assert len(res.steps) == 3

    def test_r2_increases(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        assert res.r2[-1] >= res.r2[0]

    def test_delta_r2_positive_for_interaction(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        assert res.delta_r2[-1] > 0


class TestSimpleSlopes:
    def test_three_slopes(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        assert len(res.simple_slopes) == 3

    def test_slope_labels(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        labels = [s.w_label for s in res.simple_slopes]
        assert "\u22121 SD" in labels
        assert "Mean" in labels
        assert "+1 SD" in labels

    def test_slopes_differ(self, interaction_data):
        """With a real interaction, slopes at different W levels should differ."""
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        slopes_b = [s.b for s in res.simple_slopes]
        assert slopes_b[0] != slopes_b[2]  # -1 SD != +1 SD


class TestJohnsonNeyman:
    def test_jn_computed(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y", jn=True)
        assert res.jn is not None

    def test_jn_has_conditional_effects(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y", jn=True)
        assert len(res.jn.conditional_effects) == 1000

    def test_jn_skipped(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y", jn=False)
        assert res.jn is None


class TestModerationTable:
    def test_table_str_nonempty(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        assert len(res.table_str) > 100

    def test_table_has_structure(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        text = res.table_str
        assert "Table 2" in text
        assert "R\u00b2" in text
        assert "\u0394R\u00b2" in text
        assert "mean-centred" in text
        assert "*p < .05" in text


class TestModerationPlot:
    def test_plot_returns_figure(self, interaction_data):
        import matplotlib
        matplotlib.use("Agg")
        res = moderation_analysis(interaction_data, x="x", w="w", y="y")
        fig = res.plot()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_jn_plot_returns_figure(self, interaction_data):
        import matplotlib
        matplotlib.use("Agg")
        res = moderation_analysis(interaction_data, x="x", w="w", y="y", jn=True)
        fig = res.plot_jn()
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close("all")

    def test_jn_plot_raises_without_jn(self, interaction_data):
        res = moderation_analysis(interaction_data, x="x", w="w", y="y", jn=False)
        with pytest.raises(ValueError, match="Johnson"):
            res.plot_jn()


class TestModerationEdgeCases:
    def test_missing_column(self, interaction_data):
        with pytest.raises(KeyError):
            moderation_analysis(interaction_data, x="missing", w="w", y="y")

    def test_nan_handling(self, interaction_data):
        """NaN rows should be dropped."""
        import warnings
        df = interaction_data.copy()
        df.loc[0, "x"] = np.nan
        df.loc[1, "y"] = np.nan
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = moderation_analysis(df, x="x", w="w", y="y")
        assert res.n == 398
        drop_warnings = [x for x in w if "dropped" in str(x.message)]
        assert len(drop_warnings) >= 1

    def test_same_x_w_raises(self, interaction_data):
        """x and w must be different variables."""
        with pytest.raises(ValueError, match="different"):
            moderation_analysis(interaction_data, x="x", w="x", y="y")

    def test_controls_centred(self, controlled_data):
        """Continuous controls should be mean-centred."""
        res = moderation_analysis(
            controlled_data, x="x", w="w", y="y",
            controls=["age", "gender"],
        )
        # age is continuous -> should appear as age_c in step summaries
        step1_preds = list(res.step_summaries[0].index)
        assert "age_c" in step1_preds
        # gender is binary -> should stay as gender (not gender_c)
        assert "gender" in step1_preds
