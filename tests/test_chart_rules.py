"""
Tests for app.utils.chart_rules
Covers all rule branches and the auto-insight suggestion function.
"""

from app.utils.chart_rules import recommend_chart, suggest_best_insights


# ── recommend_chart tests ─────────────────────────────────────────────────────


class TestRecommendChart:

    def test_datetime_and_numeric_returns_line(self):
        cols = [
            {"name": "date", "semantic": "datetime"},
            {"name": "revenue", "semantic": "numeric"},
        ]
        assert recommend_chart(cols, "Show revenue over time") == "line"

    def test_distribution_keyword_returns_histogram(self):
        cols = [{"name": "salary", "semantic": "numeric"}]
        assert recommend_chart(cols, "Show distribution of salaries") == "histogram"

    def test_spread_keyword_returns_histogram(self):
        cols = [{"name": "age", "semantic": "numeric"}]
        assert recommend_chart(cols, "Show the spread of ages") == "histogram"

    def test_proportion_with_low_cardinality_returns_pie(self):
        cols = [
            {"name": "category", "semantic": "categorical", "unique": 5},
            {"name": "revenue", "semantic": "numeric"},
        ]
        assert recommend_chart(cols, "Show proportion of revenue by category") == "pie"

    def test_proportion_with_high_cardinality_does_not_pie(self):
        """High cardinality categorical should not produce pie."""
        cols = [
            {"name": "product", "semantic": "categorical", "unique": 20},
            {"name": "sales", "semantic": "numeric"},
        ]
        result = recommend_chart(cols, "Show proportion of sales by product")
        assert result != "pie"

    def test_correlation_with_few_numerics_returns_scatter(self):
        cols = [
            {"name": "height", "semantic": "numeric"},
            {"name": "weight", "semantic": "numeric"},
        ]
        assert recommend_chart(cols, "Show correlation between height and weight") == "scatter"

    def test_correlation_with_many_numerics_returns_heatmap(self):
        cols = [{"name": f"col{i}", "semantic": "numeric"} for i in range(6)]
        assert recommend_chart(cols, "Show correlation matrix") == "heatmap"

    def test_categorical_and_numeric_returns_bar(self):
        cols = [
            {"name": "region", "semantic": "categorical", "unique": 5},
            {"name": "sales", "semantic": "numeric"},
        ]
        assert recommend_chart(cols, "Compare sales across regions") == "bar"

    def test_categorical_high_cardinality_returns_none(self):
        """Over 15 categories → ambiguous → return None (LLM should decide)."""
        cols = [
            {"name": "product", "semantic": "categorical", "unique": 20},
            {"name": "revenue", "semantic": "numeric"},
        ]
        result = recommend_chart(cols, "Show revenue by product")
        assert result is None

    def test_two_numerics_no_categorical_returns_scatter(self):
        cols = [
            {"name": "x", "semantic": "numeric"},
            {"name": "y", "semantic": "numeric"},
        ]
        assert recommend_chart(cols, "plot x vs y") == "scatter"

    def test_fallback_returns_bar(self):
        cols = [{"name": "name", "semantic": "other"}]
        assert recommend_chart(cols, "show something") == "bar"

    def test_trend_keyword_with_datetime_returns_line(self):
        cols = [
            {"name": "month", "semantic": "datetime"},
            {"name": "profit", "semantic": "numeric"},
        ]
        assert recommend_chart(cols, "Show profit trend over time") == "line"

    def test_comparison_keyword_returns_bar(self):
        cols = [
            {"name": "team", "semantic": "categorical", "unique": 4},
            {"name": "score", "semantic": "numeric"},
        ]
        assert recommend_chart(cols, "Compare scores across teams") == "bar"


# ── suggest_best_insights tests ───────────────────────────────────────────────


class TestSuggestBestInsights:

    def test_returns_list(self):
        cols = [
            {"name": "date", "semantic": "datetime"},
            {"name": "revenue", "semantic": "numeric"},
            {"name": "region", "semantic": "categorical", "unique": 4},
        ]
        result = suggest_best_insights(cols)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_each_insight_has_required_keys(self):
        cols = [
            {"name": "month", "semantic": "datetime"},
            {"name": "sales", "semantic": "numeric"},
            {"name": "category", "semantic": "categorical", "unique": 3},
        ]
        result = suggest_best_insights(cols)
        for item in result:
            assert "chart_type" in item
            assert "task" in item

    def test_datetime_numeric_produces_line_insight(self):
        cols = [
            {"name": "date", "semantic": "datetime"},
            {"name": "value", "semantic": "numeric"},
        ]
        result = suggest_best_insights(cols)
        types = [i["chart_type"] for i in result]
        assert "line" in types

    def test_respects_max_insights(self):
        cols = [
            {"name": f"num{i}", "semantic": "numeric"} for i in range(10)
        ] + [
            {"name": "date", "semantic": "datetime"},
            {"name": "cat", "semantic": "categorical", "unique": 3},
        ]
        result = suggest_best_insights(cols, max_insights=3)
        assert len(result) <= 3

    def test_unique_chart_types(self):
        """Each returned insight should have a distinct chart_type."""
        cols = [
            {"name": "date", "semantic": "datetime"},
            {"name": "revenue", "semantic": "numeric"},
            {"name": "cost", "semantic": "numeric"},
            {"name": "region", "semantic": "categorical", "unique": 4},
        ]
        result = suggest_best_insights(cols)
        types = [i["chart_type"] for i in result]
        assert len(types) == len(set(types)), "Chart types should be unique"
